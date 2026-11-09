"""Local, bounded FFmpeg operations; output publication never replaces a file."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from mcsfx.errors import McsfxError

DATA_URI_MAX_BYTES = 45 * 1024 * 1024
SEEDANCE_MAX_BYTES = 200_000_000
VERSION_TIMEOUT = 10
PROBE_TIMEOUT = 30
MEDIA_TIMEOUT = 300


def keyframe_duration(seconds: float) -> int:
    """Reject overlong footage instead of clamping or modifying its timeline."""
    if not math.isfinite(seconds) or not 0 < seconds <= 15:
        raise McsfxError(
            "MEDIA_CONSTRAINT", "grok-keyframe video must be over 0 and at most 15 seconds."
        )
    return math.ceil(seconds)


@dataclass(frozen=True)
class MediaInfo:
    """Duration is the video track duration, not a longer container/audio duration."""

    duration: float
    width: int
    height: int
    fps: float
    video_codec: str
    has_audio: bool
    size: int
    format_name: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Probe:
    info: MediaInfo
    video: dict[str, Any] | None
    audio: dict[str, Any] | None


def _number(value: Any) -> Fraction | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        return None


def _duration(stream: dict[str, Any]) -> Fraction | None:
    ticks, base = _number(stream.get("duration_ts")), _number(stream.get("time_base"))
    if ticks is not None and base is not None and ticks > 0 and base > 0:
        return ticks * base
    seconds = _number(stream.get("duration"))
    return seconds if seconds is not None and seconds > 0 else None


def _start(stream: dict[str, Any]) -> Fraction:
    ticks, base = _number(stream.get("start_pts")), _number(stream.get("time_base"))
    if ticks is not None and base is not None and base > 0:
        return ticks * base
    return _number(stream.get("start_time")) or Fraction(0)


def _integer(value: Any) -> int:
    number = _number(value)
    return int(number) if number is not None and number.denominator == 1 else 0


def _seconds(value: Fraction) -> str:
    return f"{float(value):.12f}"


class MediaTools:
    """Executable overrides are paths or PATH names, never shell command strings."""

    def __init__(self, ffmpeg: str | None = None, ffprobe: str | None = None) -> None:
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self._resolved: dict[str, str] = {}

    def _executable(self, name: str) -> str:
        if name in self._resolved:
            return self._resolved[name]
        configured = getattr(self, name)
        candidate = os.path.expanduser(configured) if configured else name
        found = shutil.which(candidate)
        if not found:
            raise McsfxError(
                "TOOL_NOT_FOUND",
                f"Cannot find {name}; configure its executable path.",
                details={"tool": name},
            )
        resolved = str(Path(found).resolve())
        self._resolved[name] = resolved
        return resolved

    def _run(
        self, tool: str, args: list[str], *, timeout: int, code: str = "MEDIA_PROCESS_FAILED"
    ) -> subprocess.CompletedProcess[str]:
        command = [self._executable(tool), *args]
        try:
            result = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                shell=False,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise McsfxError(
                "MEDIA_TIMEOUT",
                f"{tool} exceeded its {timeout}-second limit.",
                details={"tool": tool, "timeout_seconds": timeout},
            ) from exc
        except OSError as exc:
            raise McsfxError(
                "TOOL_EXECUTION_FAILED", f"Cannot execute {tool}.", details={"tool": tool}
            ) from exc
        if result.returncode:
            # Do not expose raw tool output, which can contain arbitrary input metadata.
            raise McsfxError(
                code,
                f"{tool} could not process the local media.",
                details={"tool": tool, "returncode": result.returncode},
            )
        return result

    def check(self) -> dict[str, str]:
        """Return ffmpeg/ffprobe paths and ffmpeg_version/ffprobe_version banners."""
        result = {}
        for name in ("ffmpeg", "ffprobe"):
            completed = self._run(name, ["-version"], timeout=VERSION_TIMEOUT)
            lines = completed.stdout.splitlines()
            if not lines or not lines[0].lower().startswith(f"{name} version "):
                raise McsfxError("TOOL_EXECUTION_FAILED", f"Invalid {name} version response.")
            result[name] = self._executable(name)
            result[f"{name}_version"] = lines[0].strip()
        return result

    @staticmethod
    def _input_path(path: Path) -> Path:
        path = Path(path).expanduser().absolute()
        try:
            valid = path.is_file() and path.stat().st_size > 0
        except OSError as exc:
            raise McsfxError("MEDIA_INVALID", "Cannot read the local media file.") from exc
        if not valid:
            raise McsfxError("MEDIA_INVALID", "Media must be a nonempty local file.")
        return path

    def _json(self, args: list[str]) -> dict[str, Any]:
        result = self._run("ffprobe", args, timeout=PROBE_TIMEOUT, code="MEDIA_INVALID")
        try:
            data = json.loads(result.stdout)
        except (ValueError, TypeError) as exc:
            raise McsfxError("MEDIA_INVALID", "ffprobe returned invalid media metadata.") from exc
        if not isinstance(data, dict) or data.get("error"):
            raise McsfxError("MEDIA_INVALID", "ffprobe returned invalid media metadata.")
        return data

    def _read(self, path: Path) -> _Probe:
        path = self._input_path(path)
        data = self._json(
            [
                "-v",
                "error",
                "-protocol_whitelist",
                "file",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ]
        )
        streams, container = data.get("streams"), data.get("format", {})
        if (
            not isinstance(streams, list)
            or not streams
            or not all(isinstance(stream, dict) for stream in streams)
            or not isinstance(container, dict)
        ):
            raise McsfxError("MEDIA_INVALID", "Missing or malformed stream metadata.")
        video, audio = None, None
        for position, raw in enumerate(streams):
            stream = dict(raw)
            index = _number(stream.get("index", position))
            if index is None or index < 0 or index.denominator != 1:
                raise McsfxError("MEDIA_INVALID", "Invalid media stream index.")
            stream["index"] = int(index)
            disposition = stream.get("disposition") or {}
            if not isinstance(disposition, dict):
                raise McsfxError("MEDIA_INVALID", "Invalid stream disposition metadata.")
            if (
                stream.get("codec_type") == "video"
                and video is None
                and not _integer(disposition.get("attached_pic", 0))
            ):
                video = stream
            elif stream.get("codec_type") == "audio" and audio is None:
                audio = stream
        selected = video or audio or {}
        duration = _duration(selected) or _number(container.get("duration")) or Fraction(0)
        fps = _number((video or {}).get("avg_frame_rate"))
        if fps is None or fps <= 0:
            fps = _number((video or {}).get("r_frame_rate")) or Fraction(0)
        codec = (video or {}).get("codec_name", "")
        format_name = container.get("format_name", "")
        if not isinstance(codec, str) or not isinstance(format_name, str):
            raise McsfxError("MEDIA_INVALID", "Invalid codec/container metadata.")
        try:
            duration_float, fps_float = float(duration), float(fps)
            size = path.stat().st_size
        except (OverflowError, OSError) as exc:
            raise McsfxError("MEDIA_INVALID", "Invalid or unreadable media metadata.") from exc
        if not math.isfinite(duration_float) or not math.isfinite(fps_float):
            raise McsfxError("MEDIA_INVALID", "Media timing must be finite.")
        info = MediaInfo(
            duration_float,
            _integer((video or {}).get("width")),
            _integer((video or {}).get("height")),
            fps_float,
            codec,
            audio is not None,
            size,
            format_name,
        )
        return _Probe(info, video, audio)

    def probe(self, path: Path) -> MediaInfo:
        return self._read(path).info

    @staticmethod
    def _validate_common(info: MediaInfo) -> None:
        if not {"mp4", "mov"}.intersection(info.format_name.lower().split(",")):
            raise McsfxError("MEDIA_INVALID", "Input must be an MP4 or MOV container.")
        if info.video_codec.lower() not in {"h264", "hevc"}:
            raise McsfxError("MEDIA_INVALID", "Input must contain H.264 or HEVC video.")
        if (
            not math.isfinite(info.duration)
            or info.duration <= 0
            or not math.isfinite(info.fps)
            or info.fps <= 0
            or info.width <= 0
            or info.height <= 0
            or info.size <= 0
        ):
            raise McsfxError("MEDIA_INVALID", "Video dimensions, size and timing must be positive.")

    def validate_input(self, path: Path, preset: str, transport: str) -> MediaInfo:
        if preset not in {"grok-edit", "grok-keyframe", "seedance-2.5-edit"}:
            raise McsfxError("CONFIG_ERROR", "Unknown media capability preset.")
        if transport not in {"data_uri", "tos"}:
            raise McsfxError("CONFIG_ERROR", "Unknown media transport.")
        if preset == "seedance-2.5-edit" and transport != "tos":
            raise McsfxError("CONFIG_ERROR", "Seedance editing requires TOS transport.")
        if preset == "grok-keyframe" and transport != "data_uri":
            raise McsfxError("CONFIG_ERROR", "grok-keyframe requires PNG data_uri transport.")
        info = self.probe(path)
        self._validate_common(info)
        failures = []
        if transport == "data_uri" and info.size > DATA_URI_MAX_BYTES:
            failures.append("data_uri input exceeds the 45 MiB raw-file safety limit")
        if preset == "grok-edit":
            if info.duration > 8.7:
                failures.append("grok-edit video must be at most 8.7 seconds")
        elif preset == "grok-keyframe":
            keyframe_duration(info.duration)
        else:
            if not 4 <= info.duration <= 30:
                failures.append("Seedance video must be 4 to 30 seconds")
            if info.size > SEEDANCE_MAX_BYTES:
                failures.append("Seedance input must be at most 200 MB")
            if not 24 <= info.fps <= 60:
                failures.append("Seedance video must be 24 to 60 fps")
            if not (300 <= info.width <= 6000 and 300 <= info.height <= 6000):
                failures.append("Seedance dimensions must each be 300 to 6000 pixels")
            if not Fraction(2, 5) <= Fraction(info.width, info.height) <= Fraction(5, 2):
                failures.append("Seedance aspect ratio must be 0.4 to 2.5")
            if not 407696 <= info.width * info.height <= 8295044:
                failures.append("Seedance pixel count must be 407696 to 8295044")
        if failures:
            raise McsfxError(
                "MEDIA_CONSTRAINT",
                "; ".join(failures) + ".",
                details={"preset": preset, "violations": failures},
            )
        return info

    @contextmanager
    def _output(self, destination: Path, *, suffix: str = ".part.mp4") -> Iterator[Path]:
        destination = Path(destination).expanduser().absolute()
        temporary = None
        try:
            if os.path.lexists(destination):
                raise McsfxError("OUTPUT_EXISTS", "The destination already exists.")
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(
                prefix=f".{destination.stem}.", suffix=suffix, dir=destination.parent
            )
            temporary = Path(name)
            os.close(fd)
            yield temporary
            # Same-directory hard links atomically fail if another writer won the race.
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise McsfxError("OUTPUT_EXISTS", "The destination already exists.") from exc
        except OSError as exc:
            raise McsfxError("OUTPUT_ERROR", "Cannot create or publish the media output.") from exc
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    logging.getLogger(__name__).warning("Cannot remove temporary media output.")

    @staticmethod
    def _video_options(video: dict[str, Any]) -> list[str]:
        base = _number(video.get("time_base"))
        if base is None or base <= 0 or base.denominator > 2**31 - 1:
            raise McsfxError("MEDIA_INVALID", "Video has no usable track time base.")
        return [
            "-c:v",
            "copy",
            "-copytb",
            "1",
            "-fps_mode:v",
            "passthrough",
            "-video_track_timescale",
            str(base.denominator),
            "-avoid_negative_ts",
            "disabled",
            "-use_editlist",
            "1",
            "-movflags",
            "+faststart",
        ]

    def _decode(self, path: Path, data: _Probe, *, audio: bool) -> None:
        if data.video is None:
            raise McsfxError("MEDIA_NO_VIDEO", "Media does not contain a video track.")
        args = [
            "-hide_banner",
            "-nostdin",
            "-nostats",
            "-v",
            "info" if audio else "error",
            "-xerror",
            "-err_detect",
            "explode",
            "-protocol_whitelist",
            "file",
            "-i",
            str(path),
            "-map",
            f"0:{data.video['index']}",
        ]
        if audio:
            if data.audio is None:
                raise McsfxError("MEDIA_NO_AUDIO", "Generated media has no audio track.")
            args += ["-map", f"0:{data.audio['index']}", "-af", "astats=metadata=0:reset=0"]
        else:
            args += ["-an"]
        args += ["-fps_mode:v", "passthrough", "-progress", "pipe:1", "-f", "null", "-"]
        result = self._run("ffmpeg", args, timeout=MEDIA_TIMEOUT, code="MEDIA_DECODE_FAILED")
        frames = re.findall(r"^frame=\s*(\d+)\s*$", result.stdout, flags=re.MULTILINE)
        if not frames or max(map(int, frames)) == 0:
            raise McsfxError("MEDIA_DECODE_FAILED", "No video frames were decoded.")
        if audio:
            prefix = r"^\[Parsed_astats_\d+ @ [^\]]+\] "
            summaries = list(re.finditer(prefix + r"Overall\s*$", result.stderr, re.MULTILINE))
            if not summaries:
                raise McsfxError("MEDIA_DECODE_FAILED", "Missing final audio statistics.")
            summary = result.stderr[summaries[-1].end() :]
            peaks = re.findall(prefix + r"Peak level dB:\s*([^\s]+)", summary, re.MULTILINE)
            samples = re.findall(prefix + r"Number of samples:\s*([^\s]+)", summary, re.MULTILINE)
            try:
                levels = [float(peak) for peak in peaks]
                counts = [float(count) for count in samples]
            except ValueError as exc:
                raise McsfxError("MEDIA_DECODE_FAILED", "Cannot verify decoded audio.") from exc
            if (
                not levels
                or not counts
                or max(counts) <= 0
                or any(not math.isfinite(count) for count in counts)
                or any(math.isnan(level) or level == math.inf for level in levels)
            ):
                raise McsfxError("MEDIA_DECODE_FAILED", "No valid decoded audio samples.")
            if all(level == -math.inf for level in levels):
                raise McsfxError("MEDIA_SILENT", "Generated audio is wholly digitally silent.")

    def verify_generated(self, path: Path) -> MediaInfo:
        path = self._input_path(path)
        data = self._read(path)
        if data.video is None:
            raise McsfxError("MEDIA_NO_VIDEO", "Generated media has no video track.")
        if data.audio is None:
            raise McsfxError("MEDIA_NO_AUDIO", "Generated media has no audio track.")
        if data.info.duration <= 0 or not _duration(data.audio):
            raise McsfxError("MEDIA_INVALID", "Generated tracks must have positive timing.")
        self._decode(path, data, audio=True)
        return data.info

    def _packets(self, path: Path, video: dict[str, Any]) -> list[tuple[Any, ...]]:
        base = _number(video.get("time_base"))
        if base is None or base <= 0:
            raise McsfxError("MEDIA_INVALID", "Missing video packet time base.")
        data = self._json(
            [
                "-v",
                "error",
                "-protocol_whitelist",
                "file",
                "-select_streams",
                str(video["index"]),
                "-show_packets",
                "-show_data_hash",
                "sha256",
                "-show_entries",
                "packet=pts,dts,duration,data_hash",
                "-of",
                "json",
                str(path),
            ]
        )
        packets = data.get("packets")
        if not isinstance(packets, list) or not packets:
            raise McsfxError("MEDIA_INVALID", "Video contains no readable packets.")
        signature = []
        for packet in packets:
            if not isinstance(packet, dict):
                raise McsfxError("MEDIA_INVALID", "Malformed video packet metadata.")
            times = [_number(packet.get(key)) for key in ("pts", "dts", "duration")]
            digest = packet.get("data_hash")
            if any(value is None for value in times) or not isinstance(digest, str):
                raise McsfxError("MEDIA_INVALID", "Cannot verify video packet timing/content.")
            signature.append((*[value * base for value in times], digest))
        return signature

    def _verify_copy(self, source: Path, original: _Probe, output: Path, copied: _Probe) -> None:
        if (
            original.video is None
            or copied.video is None
            or copied.info.video_codec != original.info.video_codec
            or _duration(original.video) != _duration(copied.video)
            or self._packets(source, original.video) != self._packets(output, copied.video)
        ):
            raise McsfxError("VIDEO_CHANGED", "Output video frames or timestamps changed.")

    def prepare(self, source: Path, dest: Path) -> MediaInfo:
        """Stream-copy the video only; fully decode and verify it before publication."""
        source = self._input_path(source)
        original = self._read(source)
        self._validate_common(original.info)
        if original.video is None:
            raise McsfxError("MEDIA_NO_VIDEO", "Source has no video track.")
        with self._output(dest) as temporary:
            self._run(
                "ffmpeg",
                [
                    "-hide_banner",
                    "-nostdin",
                    "-v",
                    "error",
                    "-xerror",
                    "-copyts",
                    "-protocol_whitelist",
                    "file",
                    "-i",
                    str(source),
                    "-map",
                    f"0:{original.video['index']}",
                    "-an",
                    "-sn",
                    "-dn",
                    *self._video_options(original.video),
                    "-y",
                    str(temporary),
                ],
                timeout=MEDIA_TIMEOUT,
            )
            copied = self._read(temporary)
            self._validate_common(copied.info)
            if copied.info.has_audio:
                raise McsfxError("MEDIA_INVALID", "Prepared output unexpectedly contains audio.")
            self._decode(temporary, copied, audio=False)
            self._verify_copy(source, original, temporary, copied)
        return copied.info

    def extract_keyframe(self, source: Path, dest: Path) -> MediaInfo:
        """Extract the first decoded frame as a validated, lossless PNG, without seeking."""
        source = self._input_path(source)
        original = self._read(source)
        self._validate_common(original.info)
        if original.video is None:
            raise McsfxError("MEDIA_NO_VIDEO", "Media does not contain a video track.")
        with self._output(dest, suffix=".part.png") as temporary:
            self._run(
                "ffmpeg",
                [
                    "-hide_banner",
                    "-nostdin",
                    "-v",
                    "error",
                    "-xerror",
                    "-noautorotate",
                    "-protocol_whitelist",
                    "file",
                    "-i",
                    str(source),
                    "-map",
                    f"0:{original.video['index']}",
                    "-frames:v",
                    "1",
                    "-an",
                    "-sn",
                    "-dn",
                    "-c:v",
                    "png",
                    "-pix_fmt",
                    "rgb24",
                    "-f",
                    "image2",
                    "-update",
                    "1",
                    "-y",
                    str(temporary),
                ],
                timeout=MEDIA_TIMEOUT,
            )
            frame = self._read(temporary)
            with temporary.open("rb") as stream:
                signature = stream.read(8)
            if (
                signature != b"\x89PNG\r\n\x1a\n"
                or frame.info.video_codec != "png"
                or frame.info.has_audio
                or (frame.info.width, frame.info.height)
                != (original.info.width, original.info.height)
                or frame.info.size > DATA_URI_MAX_BYTES
            ):
                raise McsfxError("MEDIA_INVALID", "Extracted keyframe is not a valid bounded PNG.")
            self._decode(temporary, frame, audio=False)
        return frame.info

    def preserve(self, source: Path, generated: Path, dest: Path) -> dict[str, Any]:
        """Keep video packets/timing; only pad or trim the generated audio tail.

        duration_delta_seconds is OUTPUT video minus SOURCE video (normally zero).
        audio_adjustment_seconds is SOURCE video minus GENERATED audio: positive
        means padding, negative means trimming. No content-based alignment occurs.
        """
        source, generated = self._input_path(source), self._input_path(generated)
        original, model = self._read(source), self._read(generated)
        self._validate_common(original.info)
        if original.video is None or model.video is None:
            raise McsfxError("MEDIA_NO_VIDEO", "Both inputs must contain a video track.")
        if model.audio is None:
            raise McsfxError("MEDIA_NO_AUDIO", "Generated media has no audio track.")
        video_length, audio_length = _duration(original.video), _duration(model.audio)
        if video_length is None or audio_length is None:
            raise McsfxError("MEDIA_INVALID", "Track durations are required for preservation.")
        adjustment = video_length - audio_length
        if abs(adjustment) > Fraction(1, 2):
            raise McsfxError(
                "TIMELINE_MISMATCH",
                "Generated audio differs from source video by over 0.5 seconds.",
                details={
                    "source_video_duration_seconds": float(video_length),
                    "generated_audio_duration_seconds": float(audio_length),
                    "duration_delta_seconds": float(-adjustment),
                },
            )
        self._decode(generated, model, audio=True)
        warnings = []
        if adjustment:
            verb = "Padded" if adjustment > 0 else "Trimmed"
            warnings.append(f"{verb} only the audio tail by {float(abs(adjustment)):.6f} seconds.")
        # Reset only the audio timestamp origin, then place it at the source video's
        # existing origin. Do not constrain the muxer: -t/-shortest can drop video.
        audio_filter = (
            f"asetpts=PTS-STARTPTS,apad,atrim=duration={_seconds(video_length)},"
            f"asetpts=PTS+({_seconds(_start(original.video))})/TB"
        )
        with self._output(dest) as temporary:
            self._run(
                "ffmpeg",
                [
                    "-hide_banner",
                    "-nostdin",
                    "-v",
                    "error",
                    "-xerror",
                    "-copyts",
                    "-protocol_whitelist",
                    "file",
                    "-i",
                    str(source),
                    "-protocol_whitelist",
                    "file",
                    "-i",
                    str(generated),
                    "-map",
                    f"0:{original.video['index']}",
                    "-map",
                    f"1:{model.audio['index']}",
                    "-sn",
                    "-dn",
                    *self._video_options(original.video),
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-af",
                    audio_filter,
                    "-y",
                    str(temporary),
                ],
                timeout=MEDIA_TIMEOUT,
            )
            output_info = self.verify_generated(temporary)
            copied = self._read(temporary)
            self._verify_copy(source, original, temporary, copied)
            output_audio = _duration(copied.audio or {})
            sample_rate = _integer((copied.audio or {}).get("sample_rate"))
            # Allow one AAC frame for encoder padding, never a video-frame tolerance.
            tolerance = Fraction(1025, sample_rate) if sample_rate > 0 else Fraction(1, 20)
            if (
                output_audio is None
                or abs(output_audio - video_length) > tolerance
                or abs(_start(copied.audio or {}) - _start(original.video)) > tolerance
            ):
                raise McsfxError(
                    "TIMELINE_MISMATCH", "Output audio does not cover the video timeline."
                )
        return {
            "path": str(Path(dest).expanduser().absolute()),
            "media": output_info.as_dict(),
            "duration_delta_seconds": float(_duration(copied.video) - video_length),
            "audio_adjustment_seconds": float(adjustment),
            "warnings": warnings,
        }
