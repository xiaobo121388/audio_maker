"""Pure metadata/failure tests plus optional real local FFmpeg integration tests."""

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from mcsfx.errors import McsfxError
from mcsfx.media import DATA_URI_MAX_BYTES, SEEDANCE_MAX_BYTES, MediaTools


def payload(*, duration=6, audio=6, container_duration=None, **video_fields):
    video = {
        "index": 0,
        "codec_type": "video",
        "codec_name": "h264",
        "width": 1280,
        "height": 720,
        "avg_frame_rate": "30/1",
        "r_frame_rate": "30/1",
        "time_base": "1/15360",
        "duration_ts": round(duration * 15360),
        "duration": str(duration),
        "start_pts": 0,
        "start_time": "0",
    }
    video.update(video_fields)
    streams = [video]
    if audio is not None:
        streams.append(
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "time_base": "1/48000",
                "sample_rate": "48000",
                "duration_ts": round(audio * 48000),
                "duration": str(audio),
                "start_pts": 0,
                "start_time": "0",
            }
        )
    return {
        "streams": streams,
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "size": "99999",
            "duration": str(container_duration or max(duration, audio or 0)),
        },
    }


def completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(["mock"], returncode, stdout, stderr)


def stats_log(peaks, samples):
    lines = [
        "Overall",
        *(f"Peak level dB: {peak}" for peak in peaks),
        f"Number of samples: {samples}",
    ]
    return "\n".join(f"[Parsed_astats_0 @ 0000] {line}" for line in lines) + "\n"


@pytest.fixture
def local_file(tmp_path):
    path = tmp_path / "\u4e2d\u6587 source clip.mov"
    path.write_bytes(b"mock media content")
    return path


@pytest.fixture
def metadata(monkeypatch):
    tools = MediaTools()
    data = payload()

    def run(tool, args, **kwargs):
        assert tool == "ffprobe"
        assert kwargs["timeout"] > 0
        return completed(json.dumps(data))

    monkeypatch.setattr(tools, "_run", run)
    return tools, data


def assert_code(code, callable_, *args, **kwargs):
    with pytest.raises(McsfxError) as caught:
        callable_(*args, **kwargs)
    assert caught.value.code == code
    return caught.value


def test_probe_uses_video_ticks_and_actual_size(metadata, local_file):
    tools, data = metadata
    data["streams"][0]["duration"] = "800"
    data["format"]["duration"] = "900"
    data["streams"][0]["avg_frame_rate"] = "30000/1001"
    info = tools.probe(local_file)
    assert info.duration == 6
    assert info.fps == pytest.approx(30000 / 1001)
    assert info.size == local_file.stat().st_size
    assert info.video_codec == "h264"
    assert info.has_audio
    assert info.as_dict() == {
        "duration": 6,
        "width": 1280,
        "height": 720,
        "fps": 30000 / 1001,
        "video_codec": "h264",
        "has_audio": True,
        "size": local_file.stat().st_size,
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
    }


def test_probe_fallbacks_and_attached_picture(metadata, local_file):
    tools, data = metadata
    video = data["streams"][0]
    video.update(
        duration_ts="N/A", duration="N/A", avg_frame_rate="0/0", r_frame_rate="24000/1001", index=2
    )
    data["streams"].insert(
        0,
        {
            "index": 0,
            "codec_type": "video",
            "codec_name": "mjpeg",
            "disposition": {"attached_pic": 1},
        },
    )
    info = tools.probe(local_file)
    assert info.video_codec == "h264"
    assert info.fps == pytest.approx(24000 / 1001)
    assert info.duration == 6


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        {},
        {"streams": []},
        {"streams": [None]},
        {"streams": [{}], "format": []},
        {"error": {"code": -1}},
    ],
)
def test_malformed_probe_json(monkeypatch, local_file, data):
    tools = MediaTools()
    monkeypatch.setattr(tools, "_run", lambda *a, **kw: completed(json.dumps(data)))
    assert_code("MEDIA_INVALID", tools.probe, local_file)


def test_invalid_json(monkeypatch, local_file):
    tools = MediaTools()
    monkeypatch.setattr(tools, "_run", lambda *a, **kw: completed("not json"))
    assert_code("MEDIA_INVALID", tools.probe, local_file)


@pytest.mark.parametrize("kind", ["missing", "empty", "directory"])
def test_invalid_local_files(tmp_path, kind):
    path = tmp_path / kind
    if kind == "empty":
        path.touch()
    elif kind == "directory":
        path.mkdir()
    assert_code("MEDIA_INVALID", MediaTools().probe, path)


@pytest.mark.parametrize("codec", ["h264", "hevc"])
@pytest.mark.parametrize("container", ["mov", "mp4", "mov,mp4,m4a,3gp,3g2,mj2"])
def test_supported_containers_and_codecs(metadata, local_file, codec, container):
    tools, data = metadata
    data["streams"][0]["codec_name"] = codec
    data["format"]["format_name"] = container
    assert tools.validate_input(local_file, "grok-edit", "data_uri").video_codec == codec
    assert tools.validate_input(local_file, "seedance-2.5-edit", "tos").duration == 6


@pytest.mark.parametrize(
    "field,value",
    [
        ("codec_name", "vp9"),
        ("width", 0),
        ("height", -3),
        ("avg_frame_rate", "0/0"),
        ("avg_frame_rate", "NaN"),
        ("avg_frame_rate", "inf"),
        ("duration", "NaN"),
        ("duration", "-2"),
    ],
)
def test_rejects_invalid_video_metadata(metadata, local_file, field, value):
    tools, data = metadata
    video = data["streams"][0]
    video[field] = value
    if field == "avg_frame_rate":
        video["r_frame_rate"] = "0/0"
    if field == "duration":
        video["duration_ts"] = "N/A"
        data["format"]["duration"] = value
    assert_code("MEDIA_INVALID", tools.validate_input, local_file, "grok-edit", "tos")


def test_extension_does_not_override_container(metadata, local_file):
    tools, data = metadata
    data["format"]["format_name"] = "matroska,webm"
    assert_code("MEDIA_INVALID", tools.validate_input, local_file, "grok-edit", "tos")


@pytest.mark.parametrize("duration,valid", [(8.7, True), (8.7001, False), (0, False)])
def test_grok_duration_boundary(metadata, local_file, duration, valid):
    tools, data = metadata
    data.update(payload(duration=duration, audio=None))
    if valid:
        assert tools.validate_input(local_file, "grok-edit", "tos").duration == duration
    else:
        assert_code(
            "MEDIA_INVALID" if duration == 0 else "MEDIA_CONSTRAINT",
            tools.validate_input,
            local_file,
            "grok-edit",
            "tos",
        )


@pytest.mark.parametrize(
    "size,transport,valid",
    [
        (DATA_URI_MAX_BYTES, "data_uri", True),
        (DATA_URI_MAX_BYTES + 1, "data_uri", False),
        (DATA_URI_MAX_BYTES + 1, "tos", True),
    ],
)
def test_data_uri_raw_byte_boundary(metadata, local_file, monkeypatch, size, transport, valid):
    tools, _ = metadata
    info = replace(tools.probe(local_file), size=size)
    monkeypatch.setattr(tools, "probe", lambda path: info)
    if valid:
        assert tools.validate_input(local_file, "grok-edit", transport).size == size
    else:
        assert_code("MEDIA_CONSTRAINT", tools.validate_input, local_file, "grok-edit", transport)


@pytest.mark.parametrize(
    "changes,valid",
    [
        ({"duration": 4}, True),
        ({"duration": 30}, True),
        ({"duration": 3.999}, False),
        ({"duration": 30.001}, False),
        ({"fps": 24}, True),
        ({"fps": 60}, True),
        ({"fps": 24000 / 1001}, False),
        ({"fps": 60.001}, False),
        ({"width": 300, "height": 1400}, False),
        ({"width": 299, "height": 1500}, False),
        ({"width": 6001, "height": 2500}, False),
        ({"width": 640, "height": 1600}, True),
        ({"width": 1600, "height": 640}, True),
        ({"width": 1601, "height": 640}, False),
        ({"width": 639, "height": 1600}, False),
        ({"width": 638, "height": 639}, False),
        ({"width": 640, "height": 640}, True),
        ({"width": 3840, "height": 2160}, True),
        ({"width": 4096, "height": 2160}, False),
        ({"size": SEEDANCE_MAX_BYTES}, True),
        ({"size": SEEDANCE_MAX_BYTES + 1}, False),
    ],
)
def test_seedance_boundaries(metadata, local_file, monkeypatch, changes, valid):
    tools, _ = metadata
    info = replace(tools.probe(local_file), **changes)
    monkeypatch.setattr(tools, "probe", lambda path: info)
    if valid:
        assert tools.validate_input(local_file, "seedance-2.5-edit", "tos") == info
    else:
        assert_code(
            "MEDIA_CONSTRAINT", tools.validate_input, local_file, "seedance-2.5-edit", "tos"
        )


@pytest.mark.parametrize(
    "preset,transport",
    [
        ("unknown", "tos"),
        ("grok-edit", "unknown"),
        ("seedance-2.5-edit", "data_uri"),
    ],
)
def test_unknown_contract_values(local_file, preset, transport):
    assert_code("CONFIG_ERROR", MediaTools().validate_input, local_file, preset, transport)


def test_check_resolves_paths_and_versions(monkeypatch, tmp_path):
    executables = {tool: str(tmp_path / f"{tool} space.exe") for tool in ("ffmpeg", "ffprobe")}
    monkeypatch.setattr(shutil, "which", lambda name: executables[name])
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        name = Path(command[0]).stem.split()[0]
        return completed(f"{name} version test-build\nconfiguration: test")

    monkeypatch.setattr(subprocess, "run", run)
    result = MediaTools().check()
    assert result["ffmpeg"] == executables["ffmpeg"]
    assert result["ffprobe"] == executables["ffprobe"]
    assert result["ffmpeg_version"] == "ffmpeg version test-build"
    assert result["ffprobe_version"] == "ffprobe version test-build"
    for command, kwargs in calls:
        assert isinstance(command, list)
        assert kwargs["shell"] is False
        assert kwargs["timeout"] == 10
        assert kwargs["stdin"] == subprocess.DEVNULL


@pytest.mark.parametrize("tool", ["ffmpeg", "ffprobe"])
def test_check_missing_executable(monkeypatch, tool):
    monkeypatch.setattr(shutil, "which", lambda name: None if name == tool else name)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: completed("ffmpeg version test"))
    error = assert_code("TOOL_NOT_FOUND", MediaTools().check)
    assert error.details["tool"] == tool


@pytest.mark.parametrize(
    "failure,code",
    [
        (subprocess.TimeoutExpired("ffprobe", 30), "MEDIA_TIMEOUT"),
        (OSError("access denied"), "TOOL_EXECUTION_FAILED"),
        (None, "MEDIA_INVALID"),
    ],
)
def test_subprocess_failures_are_safe(monkeypatch, local_file, failure, code):
    tools = MediaTools()
    monkeypatch.setattr(tools, "_executable", lambda name: name)

    def run(command, **kwargs):
        if failure:
            raise failure
        return completed(stderr="unsafe embedded metadata", returncode=1)

    monkeypatch.setattr(subprocess, "run", run)
    error = assert_code(code, tools.probe, local_file)
    assert "unsafe embedded metadata" not in json.dumps(error.as_dict())


def test_probe_paths_are_one_subprocess_argument(monkeypatch, local_file):
    tools = MediaTools()
    monkeypatch.setattr(tools, "_executable", lambda name: name)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs["shell"] is False
        assert kwargs["timeout"] == 30
        return completed(json.dumps(payload()))

    monkeypatch.setattr(subprocess, "run", run)
    tools.probe(local_file)
    assert commands[0][-1] == str(local_file)
    assert commands[0].count(str(local_file)) == 1


@pytest.mark.parametrize(
    "peaks,samples,frames,code",
    [
        (["-inf"], "288000", 180, "MEDIA_SILENT"),
        (["-inf", "-inf"], "288000", 180, "MEDIA_SILENT"),
        (["-inf", "-50.1"], "288000", 180, None),
        (["-900.0"], "288000", 180, None),
        (["-12.5"], "288000", 180, None),
        (["nan"], "288000", 180, "MEDIA_DECODE_FAILED"),
        (["inf"], "288000", 180, "MEDIA_DECODE_FAILED"),
        (["-12.5"], "0", 180, "MEDIA_DECODE_FAILED"),
        (["-12.5"], "nan", 180, "MEDIA_DECODE_FAILED"),
        ([], "288000", 180, "MEDIA_DECODE_FAILED"),
        (["-12.5"], "288000", 0, "MEDIA_DECODE_FAILED"),
    ],
)
def test_full_decode_and_digital_silence(monkeypatch, local_file, peaks, samples, frames, code):
    tools = MediaTools()
    calls = []

    def run(tool, args, **kwargs):
        calls.append((tool, args))
        if tool == "ffprobe":
            return completed(json.dumps(payload()))
        assert kwargs["timeout"] == 300
        assert "-xerror" in args
        assert "-t" not in args and "-shortest" not in args
        assert args[args.index("-af") + 1] == "astats=metadata=0:reset=0"
        return completed(f"frame={frames}\nprogress=end\n", stats_log(peaks, samples))

    monkeypatch.setattr(tools, "_run", run)
    if code:
        assert_code(code, tools.verify_generated, local_file)
    else:
        assert tools.verify_generated(local_file).has_audio
    assert calls[-1][0] == "ffmpeg"
    assert calls[-1][1].count("-map") == 2


def test_verify_requires_both_tracks(metadata, local_file):
    tools, data = metadata
    data.update(payload(audio=None))
    assert_code("MEDIA_NO_AUDIO", tools.verify_generated, local_file)
    data.update(payload())
    data["streams"].pop(0)
    assert_code("MEDIA_NO_VIDEO", tools.verify_generated, local_file)


def test_silence_checks_final_astats_not_metadata(monkeypatch, local_file):
    tools = MediaTools()

    def run(tool, args, **kwargs):
        if tool == "ffprobe":
            return completed(json.dumps(payload()))
        return completed(
            "frame=180\nprogress=end\n",
            "  title: Peak level dB: -1\n  title: Number of samples: 100\n"
            + stats_log(["-inf"], "288000"),
        )

    monkeypatch.setattr(tools, "_run", run)
    assert_code("MEDIA_SILENT", tools.verify_generated, local_file)


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    """Model a muxer, but exercise real probing, guards, temp files and publication."""
    tools = MediaTools()
    source, generated = tmp_path / "source.mov", tmp_path / "model.mp4"
    source.write_bytes(b"original source")
    generated.write_bytes(b"original model")
    state = {
        "source": payload(duration=6, audio=12),
        "generated": payload(duration=3, audio=6),
        "output": payload(duration=6, audio=6),
        "commands": [],
    }
    packet = {"pts": 0, "dts": -1024, "duration": 512, "data_hash": "SHA256:mock"}

    def run(tool, args, **kwargs):
        state["commands"].append((tool, args))
        if tool == "ffprobe":
            if "-show_packets" in args:
                return completed(json.dumps({"packets": [packet]}))
            key = (
                "source"
                if args[-1] == str(source)
                else ("generated" if args[-1] == str(generated) else "output")
            )
            return completed(json.dumps(state[key]))
        if "-f" in args and args[args.index("-f") + 1] == "null":
            return completed("frame=180\nprogress=end\n", stats_log(["-20.0"], "288000"))
        Path(args[-1]).write_bytes(b"muxed result")
        return completed()

    monkeypatch.setattr(tools, "_run", run)
    return tools, source, generated, state


@pytest.mark.parametrize("adjustment", [-0.5, -0.2, 0, 0.2, 0.5])
def test_preserve_uses_track_timing_and_tail_only(pipeline, tmp_path, adjustment):
    tools, source, generated, state = pipeline
    state["generated"] = payload(duration=3, audio=6 - adjustment, container_duration=90)
    output = tmp_path / "preserved.mp4"
    metrics = tools.preserve(source, generated, output)
    assert output.read_bytes() == b"muxed result"
    assert metrics["duration_delta_seconds"] == 0
    assert metrics["audio_adjustment_seconds"] == pytest.approx(adjustment)
    assert bool(metrics["warnings"]) == bool(adjustment)
    assert source.read_bytes() == b"original source"
    assert generated.read_bytes() == b"original model"
    assert not list(tmp_path.glob("*.part.mp4"))
    mux = next(args for tool, args in state["commands"] if "-c:a" in args)
    assert "-t" not in mux and "-shortest" not in mux
    assert "-copyts" in mux
    assert mux[mux.index("-c:v") + 1] == "copy"
    assert mux[mux.index("-c:a") + 1] == "aac"
    audio_filter = mux[mux.index("-af") + 1]
    assert "apad,atrim=duration=6.000000000000" in audio_filter
    assert "atempo" not in audio_filter and "async" not in audio_filter


@pytest.mark.parametrize("audio_length", [5.499, 6.501])
def test_mismatch_keeps_model_and_never_creates_output(pipeline, tmp_path, audio_length):
    tools, source, generated, state = pipeline
    state["generated"] = payload(duration=6, audio=audio_length)
    output = tmp_path / "output.mp4"
    assert_code("TIMELINE_MISMATCH", tools.preserve, source, generated, output)
    assert generated.read_bytes() == b"original model"
    assert source.read_bytes() == b"original source"
    assert not output.exists()
    assert all(tool == "ffprobe" for tool, args in state["commands"])


def test_missing_track_duration_does_not_use_container(pipeline, tmp_path):
    tools, source, generated, state = pipeline
    state["source"]["streams"][0].update(duration_ts="N/A", duration="N/A")
    assert_code("MEDIA_INVALID", tools.preserve, source, generated, tmp_path / "out.mp4")


def test_prepare_removes_audio_and_keeps_original(pipeline, tmp_path):
    tools, source, _, state = pipeline
    state["output"] = payload(duration=6, audio=None)
    output = tmp_path / "silent.mp4"
    assert not tools.prepare(source, output).has_audio
    assert source.read_bytes() == b"original source"
    mux = next(args for tool, args in state["commands"] if "-y" in args)
    assert "-an" in mux and "-copyts" in mux
    assert mux[mux.index("-c:v") + 1] == "copy"


@pytest.mark.parametrize("operation", ["prepare", "preserve"])
def test_existing_destination_is_never_replaced(pipeline, tmp_path, operation):
    tools, source, generated, _ = pipeline
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"keep destination")
    inputs = (source, generated, output) if operation == "preserve" else (source, output)
    assert_code("OUTPUT_EXISTS", getattr(tools, operation), *inputs)
    assert output.read_bytes() == b"keep destination"
    assert not list(tmp_path.glob("*.part.mp4"))


def test_atomic_link_race_and_cleanup(monkeypatch, pipeline, tmp_path):
    tools, source, generated, _ = pipeline
    output = tmp_path / "raced.mp4"
    link = os.link

    def racing_link(temporary, destination):
        assert str(temporary).endswith(".part.mp4")
        Path(destination).write_bytes(b"concurrent writer")
        link(temporary, destination)

    monkeypatch.setattr(os, "link", racing_link)
    assert_code("OUTPUT_EXISTS", tools.preserve, source, generated, output)
    assert output.read_bytes() == b"concurrent writer"
    assert not list(tmp_path.glob("*.part.mp4"))


def test_concurrent_outputs_reserve_unique_temporary_files(tmp_path):
    tools = MediaTools()
    destination = tmp_path / "same.mp4"
    with (
        pytest.raises(McsfxError, match="already exists"),
        tools._output(destination) as first,
        tools._output(destination) as second,
    ):
        assert first != second
        first.write_bytes(b"loser")
        second.write_bytes(b"winner")
    assert destination.read_bytes() == b"winner"
    assert not list(tmp_path.glob("*.part.mp4"))


def test_unsupported_link_does_not_fall_back_to_overwrite(monkeypatch, pipeline, tmp_path):
    tools, source, generated, _ = pipeline

    def fail(*args):
        raise OSError("hard links unsupported")

    monkeypatch.setattr(os, "link", fail)
    output = tmp_path / "output.mp4"
    assert_code("OUTPUT_ERROR", tools.preserve, source, generated, output)
    assert not output.exists()
    assert not list(tmp_path.glob("*.part.mp4"))


@pytest.mark.parametrize("stage", ["mux", "decode", "packet_hash", "packet_pts", "audio_length"])
def test_failure_before_publication_keeps_inputs(monkeypatch, pipeline, tmp_path, stage):
    tools, source, generated, state = pipeline
    run = tools._run
    output = tmp_path / "output.mp4"

    def fail(tool, args, **kwargs):
        if stage == "mux" and "-y" in args:
            Path(args[-1]).write_bytes(b"partial")
            raise McsfxError("MEDIA_PROCESS_FAILED", "failed")
        if stage == "decode" and "-progress" in args and str(generated) not in args:
            raise McsfxError("MEDIA_DECODE_FAILED", "failed")
        result = run(tool, args, **kwargs)
        if stage.startswith("packet_") and "-show_packets" in args and args[-1] != str(source):
            data = json.loads(result.stdout)
            field = "data_hash" if stage == "packet_hash" else "pts"
            data["packets"][0][field] = "SHA256:changed" if field == "data_hash" else 1024
            return completed(json.dumps(data))
        return result

    if stage == "audio_length":
        state["output"] = payload(duration=6, audio=8)
    monkeypatch.setattr(tools, "_run", fail)
    with pytest.raises(McsfxError):
        tools.preserve(source, generated, output)
    assert source.read_bytes() == b"original source"
    assert generated.read_bytes() == b"original model"
    assert not output.exists()
    assert not list(tmp_path.glob("*.part.mp4"))


def test_packet_comparison_uses_rational_time_bases(monkeypatch, pipeline, tmp_path):
    tools, source, generated, state = pipeline
    state["output"]["streams"][0].update(time_base="1/30720", duration_ts=184320)
    run = tools._run

    def scaled(tool, args, **kwargs):
        result = run(tool, args, **kwargs)
        if "-show_packets" in args and args[-1] != str(source):
            data = json.loads(result.stdout)
            for key in ("pts", "dts", "duration"):
                data["packets"][0][key] *= 2
            return completed(json.dumps(data))
        return result

    monkeypatch.setattr(tools, "_run", scaled)
    assert tools.preserve(source, generated, tmp_path / "output.mp4")["duration_delta_seconds"] == 0


# These tests run without network access and never download/install executables.
@pytest.fixture(scope="module")
def real_tools():
    tools = MediaTools(os.environ.get("MCSFX_FFMPEG"), os.environ.get("MCSFX_FFPROBE"))
    try:
        tools.check()
    except McsfxError as exc:
        if exc.code == "TOOL_NOT_FOUND":
            pytest.skip("Local ffmpeg AND ffprobe required; set MCSFX_FFMPEG/MCSFX_FFPROBE.")
        raise
    return tools


@pytest.fixture
def make_media(real_tools, tmp_path):
    def create(name, *, video=2, audio=2, silence=False, offset=0, vfr=False):
        path = tmp_path / name
        args = [
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=640x640:rate=30:duration={video}",
        ]
        if audio is not None:
            signal = (
                f"anullsrc=r=48000:cl=stereo:d={audio}"
                if silence
                else f"sine=frequency=880:sample_rate=48000:duration={audio}"
            )
            args += ["-f", "lavfi", "-i", signal]
        args += ["-map", "0:v:0"]
        if audio is not None:
            args += ["-map", "1:a:0", "-c:a", "aac"]
        filters = []
        if vfr:
            filters.append(r"select=not(mod(n\,3))+not(mod(n\,5))")
        if offset:
            filters.append(f"setpts=PTS+{offset}/TB")
        if filters:
            args += ["-vf", ",".join(filters)]
        args += [
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "30",
            "-bf",
            "2",
            "-pix_fmt",
            "yuv420p",
            "-fps_mode:v",
            "passthrough",
            "-movflags",
            "+faststart",
            "-n",
            str(path),
        ]
        real_tools._run("ffmpeg", args, timeout=90)
        return path

    return create


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_md5(tools, path):
    result = tools._run(
        "ffmpeg",
        [
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-copyts",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "rawvideo",
            "-fps_mode:v",
            "passthrough",
            "-f",
            "framemd5",
            "-",
        ],
        timeout=90,
    )
    return result.stdout


@pytest.mark.media
def test_real_prepare_chinese_spaces_sha_and_frames(real_tools, make_media, tmp_path):
    source = make_media("\u4e2d\u6587 source video.mov", video=2, audio=3)
    before = sha(source)
    output = tmp_path / "\u4e2d\u6587 silent video.mp4"
    info = real_tools.prepare(source, output)
    assert not info.has_audio
    assert info.duration == pytest.approx(2)
    assert info.size == output.stat().st_size
    assert sha(source) == before
    assert frame_md5(real_tools, source) == frame_md5(real_tools, output)
    assert not list(tmp_path.glob("*.part.mp4"))


@pytest.mark.media
@pytest.mark.parametrize("adjustment", [-0.5, -0.2, 0, 0.2, 0.5])
def test_real_preserve_padding_trimming_and_frames(real_tools, make_media, tmp_path, adjustment):
    source = make_media("source.mov", video=2, audio=3.5)
    model = make_media("model.mp4", video=1, audio=2 - adjustment)
    before = sha(source), sha(model)
    output = tmp_path / "\u4e2d\u6587 preserved clip.mp4"
    result = real_tools.preserve(source, model, output)
    assert (sha(source), sha(model)) == before
    assert result["duration_delta_seconds"] == 0
    assert result["audio_adjustment_seconds"] == pytest.approx(adjustment, abs=1 / 48000)
    assert frame_md5(real_tools, source) == frame_md5(real_tools, output)
    data = real_tools._read(output)
    assert data.audio["codec_name"] == "aac"
    audio_seconds = int(data.audio["duration_ts"]) * Fraction(data.audio["time_base"])
    assert float(audio_seconds) == pytest.approx(2, abs=0.023)
    assert real_tools.verify_generated(output).has_audio


@pytest.mark.media
@pytest.mark.parametrize("vfr,offset", [(True, 0), (False, 0.4), (True, 0.4)])
def test_real_preserve_vfr_and_video_start_offset(real_tools, make_media, tmp_path, vfr, offset):
    source = make_media("source.mov", video=2, audio=4, vfr=vfr, offset=offset)
    duration = real_tools.probe(source).duration
    model = make_media("model.mp4", video=1, audio=duration)
    output = tmp_path / "preserved.mp4"
    real_tools.preserve(source, model, output)
    assert frame_md5(real_tools, source) == frame_md5(real_tools, output)


@pytest.mark.media
@pytest.mark.parametrize(
    "audio,silence,code",
    [
        (None, False, "MEDIA_NO_AUDIO"),
        (2, True, "MEDIA_SILENT"),
    ],
)
def test_real_no_audio_and_silence(real_tools, make_media, audio, silence, code):
    path = make_media("generated.mp4", audio=audio, silence=silence)
    assert_code(code, real_tools.verify_generated, path)


@pytest.mark.media
def test_real_mismatch_retains_model(real_tools, make_media, tmp_path):
    source = make_media("source.mov", audio=4)
    model = make_media("generated.mp4", audio=2.8)
    before = sha(model)
    output = tmp_path / "output.mp4"
    assert_code("TIMELINE_MISMATCH", real_tools.preserve, source, model, output)
    assert sha(model) == before
    assert not output.exists()
    assert not list(tmp_path.glob("*.part.mp4"))


@pytest.mark.media
def test_real_malformed_and_truncated_decode(real_tools, make_media, tmp_path):
    malformed = tmp_path / "bad.mp4"
    malformed.write_bytes(b"not an mp4\x00\xff")
    assert_code("MEDIA_INVALID", real_tools.verify_generated, malformed)
    valid = make_media("valid.mp4")
    truncated = tmp_path / "truncated.mp4"
    content = valid.read_bytes()
    truncated.write_bytes(content[: len(content) // 2])
    with pytest.raises(McsfxError):
        real_tools.verify_generated(truncated)


@pytest.mark.media
def test_real_no_overwrite(real_tools, make_media, tmp_path):
    source = make_media("source.mov")
    before = sha(source)
    assert_code("OUTPUT_EXISTS", real_tools.prepare, source, source)
    assert sha(source) == before
    destination = tmp_path / "existing.mp4"
    destination.write_bytes(b"keep")
    assert_code("OUTPUT_EXISTS", real_tools.preserve, source, source, destination)
    assert destination.read_bytes() == b"keep"


@pytest.mark.media
def test_real_keyframe_extraction_and_dry_run(real_tools, make_media, tmp_path):
    from mcsfx.config import Settings
    from mcsfx.jobs import JobRunner

    source = make_media("keyframe source.mp4", video=2, audio=3)
    before = sha(source)
    silent, frame = tmp_path / "silent.mp4", tmp_path / "first.png"
    real_tools.prepare(source, silent)
    info = real_tools.extract_keyframe(silent, frame)
    assert info.video_codec == "png" and not info.has_audio
    assert (info.width, info.height) == (640, 640)
    assert frame.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not real_tools.probe(silent).has_audio
    assert real_tools.probe(silent).duration == 2
    assert sha(source) == before
    hashes = []
    for path in (source, frame):
        decoded = real_tools._run(
            "ffmpeg",
            [
                "-hide_banner",
                "-nostdin",
                "-v",
                "error",
                "-noautorotate",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-frames:v",
                "1",
                "-an",
                "-pix_fmt",
                "rgb24",
                "-c:v",
                "rawvideo",
                "-f",
                "hash",
                "-hash",
                "sha256",
                "-",
            ],
            timeout=30,
        )
        hashes.append(decoded.stdout.strip())
    assert hashes[0] == hashes[1]
    assert_code("OUTPUT_EXISTS", real_tools.extract_keyframe, silent, frame)
    settings = Settings(state_dir=tmp_path / "state")
    runner = JobRunner(
        settings,
        media=real_tools,
        provider_factory=lambda _: pytest.fail("Network provider created"),
    )
    result = runner.generate(
        source,
        profile_name="grok-sfx",
        mode="regenerate",
        prompt="A stationary still, action described only.",
        out=tmp_path / "outputs",
        dry_run=True,
    )
    assert result["request"]["duration"] == 2
    assert result["request"]["generate_audio"] is True
    assert result["request"]["resolution"] == "480p"
    assert "image" in result["request"] and "video" not in result["request"]
    assert "keyframe_guided_motion_unverified" in result["warnings"]
    assert not settings.state_dir.exists()
    assert not (tmp_path / "outputs").exists()


@pytest.mark.media
def test_real_keyframe_job_mock_api_and_recovery(real_tools, make_media, tmp_path, monkeypatch):
    import base64

    import httpx

    from mcsfx.config import BUILTINS, Settings
    from mcsfx.jobs import JobRunner
    from mcsfx.providers import Provider, download_video

    monkeypatch.setenv("XAI_API_KEY", "mock-integration-secret")
    profile = replace(
        BUILTINS["grok-sfx"],
        base_url="https://relay.example/v1",
        download_mode="same_origin_content",
    )
    settings = Settings(profiles={"grok-sfx": profile}, state_dir=tmp_path / "state")
    source = make_media("input.mp4", video=2, audio=3)
    model = make_media("model.mp4", video=2, audio=2)
    calls = []

    def handle(request):
        calls.append(request)
        assert request.headers["Authorization"] == "Bearer mock-integration-secret"
        if request.method == "POST":
            assert request.url.path == "/v1/videos/generations"
            body = json.loads(request.content)
            assert body["duration"] == 2 and body["generate_audio"] is True
            raw = base64.b64decode(body["image"]["url"].split(",", 1)[1], validate=True)
            assert raw.startswith(b"\x89PNG\r\n\x1a\n")
            assert "video" not in body
            return httpx.Response(200, json={"request_id": "saved-123"})
        if request.url.path == "/v1/videos/saved-123":
            return httpx.Response(200, json={"video": {"url": "/v1/videos/saved-123/content"}})
        assert request.url.path == "/v1/videos/saved-123/content"
        return httpx.Response(200, content=model.read_bytes())

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:

        def new_runner(active_settings):
            return JobRunner(
                active_settings,
                media=real_tools,
                provider_factory=lambda p: Provider(p, client=client),
                downloader=lambda url, dest, **kw: download_video(url, dest, client=client, **kw),
            )

        first = new_runner(settings)
        submitted = first.generate(
            source,
            profile_name="grok-sfx",
            mode="regenerate",
            prompt="Stationary reference, generate an impact.",
            out=tmp_path / "outputs",
        )
        record = first.store.load(submitted["job_id"])
        assert record["generation"] == {"duration": 2, "resolution": "480p", "generate_audio": True}
        assert Path(record["artifacts"]["keyframe"]).is_file()
        assert record["keyframe"]["sha256"] == sha(Path(record["artifacts"]["keyframe"]))
        # Resume with different active config: persisted profile/remote ID must win.
        resumed = new_runner(Settings(state_dir=settings.state_dir))
        result = resumed.fetch(submitted["job_id"])
        assert result["status"] == "completed"
        assert result["artifacts"]["result"] == result["artifacts"]["generated"]
        assert resumed.fetch(submitted["job_id"]) == result
        assert len(calls) == 3
        manifest = Path(result["artifacts"]["manifest"]).read_text(encoding="utf-8")
        assert "mock-integration-secret" not in manifest
        assert "data:image" not in manifest
        assert "keyframe_guided_motion_unverified" in result["warnings"]


@pytest.mark.parametrize("duration,valid", [(0.1, True), (2, True), (15, True), (15.01, False)])
def test_keyframe_input_duration_boundary(metadata, local_file, duration, valid):
    tools, data = metadata
    data.update(payload(duration=duration))
    if valid:
        assert tools.validate_input(
            local_file, "grok-keyframe", "data_uri"
        ).duration == pytest.approx(duration)
    else:
        assert_code(
            "MEDIA_CONSTRAINT", tools.validate_input, local_file, "grok-keyframe", "data_uri"
        )


@pytest.mark.media
@pytest.mark.parametrize("failure", ["decode", "size", "empty"])
def test_real_keyframe_failure_does_not_publish(
    real_tools, make_media, tmp_path, monkeypatch, failure
):
    source = make_media("input.mp4")
    dest = tmp_path / "keyframe.png"
    before = sha(source)
    if failure == "decode":

        def fail(*args, **kwargs):
            raise McsfxError("MEDIA_DECODE_FAILED", "Cannot decode PNG.")

        monkeypatch.setattr(real_tools, "_decode", fail)
    elif failure == "size":
        monkeypatch.setattr("mcsfx.media.DATA_URI_MAX_BYTES", 8)
    else:
        run = real_tools._run

        def empty_frame(tool, args, **kwargs):
            if tool == "ffmpeg" and "image2" in args:
                return completed()
            return run(tool, args, **kwargs)

        monkeypatch.setattr(real_tools, "_run", empty_frame)
    with pytest.raises(McsfxError):
        real_tools.extract_keyframe(source, dest)
    assert not dest.exists()
    assert not list(tmp_path.glob("*.part.png"))
    assert sha(source) == before


@pytest.mark.media
def test_real_keyframe_cli_dry_run_uses_configured_tools(
    real_tools, make_media, tmp_path, monkeypatch, capsys
):
    from mcsfx.cli import main

    source = make_media("cli reference.mp4", video=2, audio=3)
    config = tmp_path / "mcsfx.toml"
    config.write_text(
        '[profiles.grok-sfx]\nbase_url="https://relay.example/v1"\n'
        'download_mode="same_origin_content"\n[tools]\n'
        f"ffmpeg={json.dumps(real_tools._executable('ffmpeg'))}\n"
        f"ffprobe={json.dumps(real_tools._executable('ffprobe'))}\n",
        encoding="utf-8",
    )
    for variable in ("XAI_API_KEY", "MCSFX_FFMPEG", "MCSFX_FFPROBE"):
        monkeypatch.delenv(variable, raising=False)
    output, state = tmp_path / "outputs", tmp_path / "state"
    assert (
        main(
            [
                "generate",
                str(source),
                "--config",
                str(config),
                "--profile",
                "grok-sfx",
                "--mode",
                "regenerate",
                "--out",
                str(output),
                "--state-dir",
                str(state),
                "--dry-run",
                "--json",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["request"]["download_mode"] == "same_origin_content"
    assert result["request"]["duration"] == 2
    assert result["request"]["preset"] == "grok-keyframe"
    assert not output.exists() and not state.exists()
