"""Durable local jobs. A submission is never repeated by recovery commands."""

import base64
import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from mcsfx.config import Profile, Settings, TosConfig, validate_profile
from mcsfx.errors import McsfxError
from mcsfx.media import MediaTools, keyframe_duration
from mcsfx.providers import Provider, download_video
from mcsfx.storage import TosStorage, object_key, validate_tos

TERMINAL = {"succeeded", "failed", "cancelled", "expired", "not_submitted"}
DEFAULT_PROMPT = (
    "Edit the reference video. Preserve the camera, action order, motion timing and scene. "
    "Add synchronized game action sound effects only. No music, speech, narration or vocals. "
    "Do not add unrelated ambience. Match the visible materials, impacts and energy effects."
)
KEYFRAME_PROMPT = (
    "Generate a game action video guided by this first frame and the described action. "
    "Only the first frame is supplied, not the original video motion. "
    "Add synchronized game action sound effects only. No music, speech, narration or vocals. "
    "Do not add unrelated ambience. Match the visible materials, impacts and energy effects."
)


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".state-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class JobStore:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def directory(self, job_id: str) -> Path:
        try:
            if str(uuid.UUID(job_id)) != job_id:
                raise ValueError
        except (ValueError, AttributeError) as exc:
            raise McsfxError("INVALID_JOB_ID", "Expected a canonical UUID job ID.") from exc
        return self.root / job_id

    def create(self, job_id: str) -> None:
        self.directory(job_id).mkdir(parents=True, exist_ok=False)

    def load(self, job_id: str) -> dict:
        try:
            record = json.loads((self.directory(job_id) / "job.json").read_text(encoding="utf-8"))
            if record["schema_version"] != 1 or record["job_id"] != job_id:
                raise ValueError
            if not Path(record["output_dir"]).is_absolute():
                raise ValueError
            Profile(**record["profile"])
            return record
        except FileNotFoundError as exc:
            raise McsfxError(
                "JOB_NOT_FOUND", "No saved job with this ID in the state directory."
            ) from exc
        except (ValueError, TypeError, KeyError) as exc:
            raise McsfxError(
                "JOB_CORRUPT", "The saved job record is invalid; no action was taken."
            ) from exc

    def save(self, record: dict) -> None:
        record["updated_at"] = timestamp()
        atomic_json(self.directory(record["job_id"]) / "job.json", record)
        output = Path(record["output_dir"])
        if output.is_dir():
            atomic_json(output / "manifest.json", record)

    @contextmanager
    def lock(self, job_id: str):
        directory = self.directory(job_id)
        if not directory.is_dir():
            raise McsfxError("JOB_NOT_FOUND", "No saved job with this ID in the state directory.")
        with (directory / "job.lock").open("a+b") as stream:
            if os.fstat(stream.fileno()).st_size == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise McsfxError(
                    "JOB_BUSY", "Another process is handling this job.", retryable=True
                ) from exc
            try:
                yield
            finally:
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def envelope(record: dict | None = None, *, error: McsfxError | None = None) -> dict:
    record = record or {}
    return {
        "schema_version": 1,
        "ok": error is None,
        "job_id": record.get("job_id"),
        "remote_id": record.get("remote_id"),
        "status": record.get("status", "error" if error else "ready"),
        "remote_status": record.get("remote_status"),
        "artifacts": record.get("artifacts", {}),
        "warnings": record.get("warnings", []),
        "metrics": record.get("metrics", {}),
        "error": error.as_dict() if error else None,
    }


class JobRunner:
    def __init__(
        self,
        settings: Settings,
        *,
        media=None,
        provider_factory=Provider,
        storage_factory=TosStorage,
        downloader=download_video,
        sleep=time.sleep,
        clock=time.monotonic,
        progress=lambda message: None,
    ):
        self.settings = settings
        self.store = JobStore(settings.state_dir)
        self.media = media or MediaTools(settings.ffmpeg, settings.ffprobe)
        self.provider_factory = provider_factory
        self.storage_factory = storage_factory
        self.downloader = downloader
        self.sleep = sleep
        self.clock = clock
        self.progress = progress
        self.current: dict | None = None

    def _save(self, record: dict) -> None:
        self.current = record
        self.store.save(record)

    def _remember_error(self, record: dict, error: McsfxError) -> None:
        record["last_error"] = error.as_dict()
        self._save(record)

    def generate(
        self,
        source: Path,
        *,
        profile_name: str,
        mode: str,
        prompt: str,
        out: Path,
        dry_run: bool = False,
    ) -> dict:
        profile = self.settings.profile(profile_name)
        validate_profile(profile)
        source = source.expanduser().resolve()
        info = self.media.validate_input(source, profile.preset, profile.transport)
        keyframe = profile.preset == "grok-keyframe"
        generation = (
            {
                "duration": keyframe_duration(info.duration),
                "resolution": "480p",
                "generate_audio": True,
            }
            if keyframe
            else None
        )
        effective_prompt = (KEYFRAME_PROMPT if keyframe else DEFAULT_PROMPT) + (
            "\nSpecific sounds and action: " + prompt
            if prompt and keyframe
            else "\nSpecific sounds: " + prompt
            if prompt
            else ""
        )
        warnings = ["alignment_unverified"]
        if keyframe:
            warnings.append("keyframe_guided_motion_unverified")
            if mode == "preserve":
                warnings.append("regenerate_recommended")
        if mode not in {"preserve", "regenerate"}:
            raise McsfxError("INVALID_MODE", "Expected preserve or regenerate.")
        if profile.transport == "tos":
            validate_tos(self.settings.tos, require_credentials=not dry_run)
        if dry_run:
            result = envelope({"status": "validated"})
            result["request"] = {
                "profile": profile.name,
                "protocol": profile.protocol,
                "model": profile.model,
                "preset": profile.preset,
                "mode": mode,
                "transport": profile.transport,
                "download_mode": profile.download_mode,
                "prompt": effective_prompt,
            }
            if keyframe:
                # Exercise actual extraction offline, but leave no durable jobs or artifacts.
                with tempfile.TemporaryDirectory(prefix="mcsfx-dryrun-") as temporary:
                    prepared = Path(temporary) / "input.mp4"
                    self.media.prepare(source, prepared)
                    self.media.validate_input(prepared, profile.preset, profile.transport)
                    frame = self.media.extract_keyframe(prepared, Path(temporary) / "keyframe.png")
                    result["keyframe"] = frame.as_dict()
                result["request"].update(
                    endpoint=profile.base_url.rstrip("/") + "/videos/generations",
                    image={"url": "<data:image/png;base64,... from validated first frame>"},
                    **generation,
                )
            else:
                result["request"]["video"] = "<silent MP4 prepared from local input>"
            result["input"] = {"path": str(source), **info.as_dict()}
            result["warnings"] = ["dry_run_no_upload_or_generation", *warnings]
            return result

        if not os.environ.get(profile.api_key_env, "").strip():
            raise McsfxError(
                "AUTH_ERROR", "The configured API key environment variable is missing."
            )

        provider = self.provider_factory(profile)
        job_id = str(uuid.uuid4())
        output = out.expanduser().resolve() / job_id
        record = {
            "schema_version": 1,
            "job_id": job_id,
            "created_at": timestamp(),
            "status": "preparing",
            "remote_status": "not_submitted",
            "remote_id": None,
            "profile": profile.snapshot(),
            "tos": self.settings.tos.snapshot(),
            "mode": mode,
            "prompt": effective_prompt,
            "output_dir": str(output),
            "state_dir": str(self.store.root),
            "input": {"path": str(source), **info.as_dict()},
            "artifacts": {},
            "warnings": warnings,
            "metrics": {},
            "upload": None,
            "downloaded": False,
            "last_error": None,
        }
        try:
            self.store.create(job_id)
            with self.store.lock(job_id):
                output.mkdir(parents=True, exist_ok=False)
                record["artifacts"]["manifest"] = str(output / "manifest.json")
                self._save(record)
                try:
                    original = output / ("input.original" + source.suffix.lower())
                    with source.open("rb") as src, original.open("xb") as dest:
                        shutil.copyfileobj(src, dest)
                    record["source_copy"] = str(original)
                    record["input"]["sha256"] = digest(original)
                    snapshot_info = self.media.validate_input(
                        original, profile.preset, profile.transport
                    )
                    record["input"].update(snapshot_info.as_dict())
                    prepared = output / "input.mp4"
                    self.media.prepare(original, prepared)
                    self.media.validate_input(prepared, profile.preset, profile.transport)
                    record["prepared"] = str(prepared)
                    if keyframe:
                        generation["duration"] = keyframe_duration(snapshot_info.duration)
                        frame_path = output / "keyframe.png"
                        frame = self.media.extract_keyframe(prepared, frame_path)
                        record["generation"] = generation
                        record["keyframe"] = {**frame.as_dict(), "sha256": digest(frame_path)}
                        record["artifacts"]["keyframe"] = str(frame_path)
                    self._save(record)
                    if profile.transport == "tos":
                        storage = self.storage_factory(self.settings.tos)
                        key = object_key(self.settings.tos, job_id)
                        record["upload"] = {"key": key, "version_id": None, "phase": "uploading"}
                        self._save(record)
                        upload = storage.upload(prepared, key)
                        record["upload"].update(
                            version_id=upload.get("version_id"), phase="uploaded"
                        )
                        self._save(record)
                        reference = storage.signed_url(key, record["upload"]["version_id"])
                    else:
                        reference = (
                            "data:image/png;base64," if keyframe else "data:video/mp4;base64,"
                        ) + base64.b64encode(
                            (frame_path if keyframe else prepared).read_bytes()
                        ).decode("ascii")
                    record.update(status="submitting", remote_status="unknown")
                    self._save(record)
                    self.progress("Submitting once; recovery will not repeat this request.")
                    if keyframe:
                        remote_id = provider.submit(
                            reference, effective_prompt, duration=generation["duration"]
                        )
                    else:
                        remote_id = provider.submit(reference, effective_prompt)
                    record.update(remote_id=remote_id, status="queued", remote_status="queued")
                    self._save(record)
                except BaseException as exc:
                    if record["status"] == "submitting":
                        unknown = (
                            not isinstance(exc, McsfxError) or exc.code == "SUBMISSION_UNKNOWN"
                        )
                        record["status"] = "submission_unknown" if unknown else "failed"
                        record["remote_status"] = "unknown" if unknown else "not_submitted"
                    elif record["remote_id"] is None:
                        record["status"] = "failed"
                    if isinstance(exc, McsfxError):
                        self._remember_error(record, exc)
                    else:
                        self._save(record)
                    self._auto_cleanup(record)
                    raise
                return envelope(record)
        finally:
            provider.close()

    def _load(self, job_id: str) -> dict:
        record = self.store.load(job_id)
        self.current = record
        if record["status"] == "submitting":
            record.update(status="submission_unknown", remote_status="unknown")
            self._save(record)
        if record["status"] == "preparing":
            record.update(status="failed", remote_status="not_submitted")
            self._save(record)
        return record

    def _query(self, record: dict, provider) -> str | None:
        result = provider.query(record["remote_id"])
        record["remote_status"] = result.status
        record["status"] = "ready" if result.status == "succeeded" else result.status
        record["last_error"] = None
        self._save(record)
        if result.status in {"failed", "cancelled", "expired"}:
            self._auto_cleanup(record)
            raise McsfxError(
                "REMOTE_" + result.status.upper(), "Remote generation ended without a usable video."
            )
        return result.video_url

    def _assert_submitted(self, record: dict) -> None:
        if record["remote_status"] == "unknown" or record["status"] == "submission_unknown":
            raise McsfxError(
                "SUBMISSION_UNKNOWN",
                "Submission outcome is unknown. Check the provider account; do not blindly resubmit.",
            )
        if not record["remote_id"]:
            raise McsfxError(
                "NOT_SUBMITTED", "This job was not submitted. Inspect last_error in its manifest."
            )
        if record["remote_status"] in {"failed", "cancelled", "expired"}:
            raise McsfxError(
                "REMOTE_" + record["remote_status"].upper(),
                "Remote generation has ended without a usable video.",
            )

    def status(self, job_id: str) -> dict:
        with self.store.lock(job_id):
            record = self._load(job_id)
            self._assert_submitted(record)
            if record["remote_status"] not in TERMINAL:
                provider = self.provider_factory(Profile(**record["profile"]))
                try:
                    self._query(record, provider)
                except McsfxError as exc:
                    self._remember_error(record, exc)
                    raise
                finally:
                    provider.close()
            if record.get("last_error") and record["status"] == "local_failed":
                old = record["last_error"]
                raise McsfxError(
                    old["code"], old["message"], retryable=old["retryable"], details=old["details"]
                )
            return envelope(record)

    def fetch(self, job_id: str, *, wait: bool = False, timeout: float | None = None) -> dict:
        with self.store.lock(job_id):
            record = self._load(job_id)
            self._assert_submitted(record)
            output = Path(record["output_dir"])
            generated = output / "generated.mp4"
            provider = None
            try:
                if record["status"] == "completed":
                    result = Path(record["artifacts"]["result"])
                    if not result.is_file() or digest(result) != record["result_sha256"]:
                        raise McsfxError(
                            "ARTIFACT_CHANGED",
                            "Completed result is missing or changed; it will not be overwritten.",
                        )
                    self._auto_cleanup(record)
                    return envelope(record)
                if not generated.exists():
                    profile = Profile(**record["profile"])
                    provider = self.provider_factory(profile)
                    deadline = self.clock() + (
                        timeout if timeout is not None else self.settings.wait_timeout
                    )
                    while True:
                        video_url = self._query(record, provider)
                        if record["remote_status"] == "succeeded":
                            if not video_url:
                                raise McsfxError(
                                    "PROTOCOL_ERROR", "Successful task did not include a video URL."
                                )
                            if profile.download_mode == "same_origin_content":
                                self.progress(
                                    "Downloading this job's trusted content endpoint; redirects are unauthenticated."
                                )
                                self.downloader(
                                    video_url,
                                    generated,
                                    profile=profile,
                                    remote_id=record["remote_id"],
                                )
                            else:
                                self.progress(
                                    "Downloading generated video without API authorization headers."
                                )
                                self.downloader(video_url, generated)
                            break
                        if not wait:
                            return envelope(record)
                        remaining = deadline - self.clock()
                        if remaining <= 0:
                            raise McsfxError(
                                "WAIT_TIMEOUT",
                                "Local wait timed out. The remote job is still active; use fetch --wait.",
                                retryable=True,
                                exit_code=3,
                            )
                        self.sleep(min(self.settings.poll_interval, remaining))
                elif record["remote_status"] != "succeeded":
                    raise McsfxError(
                        "ARTIFACT_CONFLICT",
                        "Unexpected local generated file; no file was overwritten.",
                    )
                record["artifacts"]["generated"] = str(generated)
                if (
                    record.get("generated_sha256")
                    and digest(generated) != record["generated_sha256"]
                ):
                    raise McsfxError("ARTIFACT_CHANGED", "Generated video changed after download.")
                generated_info = self.media.verify_generated(generated)
                record.update(downloaded=True, generated_sha256=digest(generated))
                record["metrics"]["generated_duration_seconds"] = generated_info.duration
                record["metrics"]["generated_video_duration_delta_seconds"] = (
                    generated_info.duration - record["input"]["duration"]
                )
                self._save(record)
                result = generated
                if record["mode"] == "preserve":
                    original = Path(record["source_copy"])
                    if digest(original) != record["input"]["sha256"]:
                        raise McsfxError(
                            "ARTIFACT_CHANGED",
                            "The saved input snapshot changed; refusing to remux.",
                        )
                    result = self._finalize_preserved(record, original, generated)
                record["artifacts"]["result"] = str(result)
                record.update(status="completed", last_error=None, result_sha256=digest(result))
                record.pop("pending_result", None)
                self._save(record)
                self._auto_cleanup(record)
                return envelope(record)
            except McsfxError as exc:
                if record["remote_status"] == "succeeded":
                    record["status"] = "local_failed"
                self._remember_error(record, exc)
                self._auto_cleanup(record)
                raise
            finally:
                if provider is not None:
                    provider.close()

    def _finalize_preserved(self, record: dict, original: Path, generated: Path) -> Path:
        output = Path(record["output_dir"])
        result = output / "result.mp4"
        pending = record.get("pending_result")
        if pending is None:
            if result.exists():
                raise McsfxError(
                    "ARTIFACT_CONFLICT",
                    "Unexpected result already exists; no file was overwritten.",
                )
            # Record the checksum before publishing result.mp4 so a crash between
            # publication and the completed-state write can be recovered offline.
            temporary = output / (".result-" + uuid.uuid4().hex + ".mp4")
            metrics = self.media.preserve(original, generated, temporary)
            metrics.pop("path", None)
            for warning in metrics.pop("warnings", []):
                if warning not in record["warnings"]:
                    record["warnings"].append(warning)
            pending = {"path": str(temporary), "sha256": digest(temporary), "metrics": metrics}
            record["pending_result"] = pending
            self._save(record)
        temporary = Path(pending["path"])
        if result.exists():
            if digest(result) != pending["sha256"]:
                raise McsfxError(
                    "ARTIFACT_CHANGED", "Uncommitted result does not match its saved checksum."
                )
        else:
            if not temporary.is_file() or digest(temporary) != pending["sha256"]:
                raise McsfxError("ARTIFACT_CHANGED", "Staged result is missing or changed.")
            os.link(temporary, result)
        record["metrics"].update(pending["metrics"])
        temporary.unlink(missing_ok=True)
        return result

    def _cleanup(self, record: dict) -> None:
        upload = record.get("upload")
        if not upload or upload["phase"] == "deleted":
            return
        if record["remote_status"] not in TERMINAL:
            raise McsfxError(
                "CLEANUP_UNSAFE",
                "Remote task is active or unknown; temporary input must be retained.",
            )
        config = TosConfig(**record["tos"])
        if upload["key"] != object_key(config, record["job_id"]):
            raise McsfxError("CLEANUP_UNSAFE", "Recorded object does not belong to this job.")
        self.storage_factory(config).delete(upload["key"], upload.get("version_id"))
        upload["phase"] = "deleted"
        record["warnings"] = [w for w in record["warnings"] if w != "cleanup_pending"]
        self._save(record)

    def _auto_cleanup(self, record: dict) -> None:
        if record["remote_status"] not in TERMINAL:
            return
        try:
            self._cleanup(record)
        except McsfxError:
            if "cleanup_pending" not in record["warnings"]:
                record["warnings"].append("cleanup_pending")
            self._save(record)

    def cleanup(self, job_id: str) -> dict:
        with self.store.lock(job_id):
            record = self._load(job_id)
            self._cleanup(record)
            return envelope(record)
