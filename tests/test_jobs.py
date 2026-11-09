from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from mcsfx.config import Settings, TosConfig
from mcsfx.errors import McsfxError
from mcsfx.jobs import JobRunner, JobStore


@dataclass
class Info:
    duration: float = 5.0

    def as_dict(self):
        return {"duration": self.duration, "width": 1280, "height": 720}


class Media:
    preserve_error = None

    def validate_input(self, path, preset, transport):
        assert Path(path).is_file()
        return Info()

    def prepare(self, source, dest):
        dest.write_bytes(source.read_bytes())
        return Info()

    def verify_generated(self, path):
        assert path.read_bytes() == b"generated"
        return Info()

    def preserve(self, original, generated, dest):
        if self.preserve_error:
            raise self.preserve_error
        dest.write_bytes(original.read_bytes() + generated.read_bytes())
        return {"duration_delta_seconds": 0, "audio_adjustment_seconds": 0}


class Provider:
    def __init__(self):
        self.submits = 0
        self.queries = 0
        self.submit_error = None
        self.query_error = None
        self.remote_state = "succeeded"

    def submit(self, reference, prompt):
        self.submits += 1
        if self.submit_error:
            raise self.submit_error
        assert reference.startswith(("data:video/mp4;base64,", "https://"))
        return "remote-123"

    def query(self, remote_id):
        self.queries += 1
        assert remote_id == "remote-123"
        if self.query_error:
            raise self.query_error
        return SimpleNamespace(
            status=self.remote_state,
            video_url="https://video.example/output?token=SECRET",
            error=None,
        )

    def close(self):
        pass


class Storage:
    def __init__(self):
        self.deleted = []
        self.fail_delete = False

    def upload(self, path, key):
        return {"version_id": "version-1"}

    def signed_url(self, key, version_id):
        return "https://input.example/video?secret=SIGNED"

    def delete(self, key, version_id):
        if self.fail_delete:
            raise McsfxError("STORAGE_ERROR", "Delete failed.")
        self.deleted.append((key, version_id))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-xai")
    monkeypatch.setenv("ARK_API_KEY", "test-ark")
    monkeypatch.setenv("TOS_ACCESS_KEY_ID", "test-ak")
    monkeypatch.setenv("TOS_SECRET_ACCESS_KEY", "test-sk")
    settings = Settings(
        state_dir=tmp_path / "state",
        poll_interval=0.1,
        tos=TosConfig(
            endpoint="https://tos-cn-beijing.volces.com", region="cn-beijing", bucket="test-bucket"
        ),
    )
    source = tmp_path / "clip with spaces \u6280\u80fd.mp4"
    source.write_bytes(b"original")
    provider, media, storage = Provider(), Media(), Storage()
    downloads = []

    def download(url, path):
        downloads.append(url)
        path.write_bytes(b"generated")

    runner = JobRunner(
        settings,
        media=media,
        provider_factory=lambda p: provider,
        storage_factory=lambda c: storage,
        downloader=download,
        sleep=lambda _: None,
    )
    return SimpleNamespace(
        runner=runner,
        provider=provider,
        source=source,
        out=tmp_path / "output",
        media=media,
        storage=storage,
        downloads=downloads,
    )


def submit(s, profile="grok", mode="preserve"):
    return s.runner.generate(s.source, profile_name=profile, mode=mode, prompt="impact", out=s.out)[
        "job_id"
    ]


class KeyframeMedia(Media):
    duration = 5.1
    extraction_error = None

    def validate_input(self, path, preset, transport):
        assert Path(path).is_file()
        return Info(self.duration)

    def extract_keyframe(self, source, dest):
        if self.extraction_error:
            raise self.extraction_error
        dest.write_bytes(b"\x89PNG\r\n\x1a\nunit-test-placeholder")
        return Info(0)


class KeyframeProvider(Provider):
    def submit(self, reference, prompt, *, duration=None):
        self.submits += 1
        self.request = {"reference": reference, "prompt": prompt, "duration": duration}
        if self.submit_error:
            raise self.submit_error
        assert reference.startswith("data:image/png;base64,")
        return "remote-123"


@pytest.fixture
def keyframe_setup(setup):
    setup.media, setup.provider = KeyframeMedia(), KeyframeProvider()
    setup.runner.media = setup.media
    setup.runner.provider_factory = lambda p: setup.provider
    return setup


def test_keyframe_unknown_submission_recovery_never_repeats(keyframe_setup):
    s = keyframe_setup
    s.provider.submit_error = McsfxError("SUBMISSION_UNKNOWN", "Unknown.")
    with pytest.raises(McsfxError):
        submit(s, profile="grok-sfx", mode="regenerate")
    job = s.runner.current["job_id"]
    for recover in (s.runner.status, s.runner.fetch):
        with pytest.raises(McsfxError) as caught:
            recover(job)
        assert caught.value.code == "SUBMISSION_UNKNOWN"
    assert s.provider.submits == 1
    assert s.runner.store.load(job)["generation"]["duration"] == 6


def test_keyframe_crashed_submission_not_replayed(keyframe_setup):
    s = keyframe_setup
    job = submit(s, profile="grok-sfx")
    record = s.runner.store.load(job)
    record.update(status="submitting", remote_id=None, remote_status="unknown")
    s.runner.store.save(record)
    with pytest.raises(McsfxError) as caught:
        s.runner.fetch(job)
    assert caught.value.code == "SUBMISSION_UNKNOWN"
    assert s.provider.submits == 1


@pytest.mark.parametrize("mode", ["regenerate", "preserve"])
def test_keyframe_preparation_and_restored_records(keyframe_setup, mode):
    s = keyframe_setup
    job = submit(s, profile="grok-sfx", mode=mode)
    assert s.provider.request["duration"] == 6
    record = s.runner.store.load(job)
    assert record["profile"]["download_mode"] == "public"
    assert record["keyframe"]["sha256"]
    assert Path(record["prepared"]).read_bytes() == b"original"
    assert "keyframe_guided_motion_unverified" in record["warnings"]
    assert ("regenerate_recommended" in record["warnings"]) == (mode == "preserve")
    result = s.runner.fetch(job)
    assert s.runner.fetch(job) == result
    assert s.provider.submits == 1
    assert len(s.downloads) == 1
    manifest = Path(result["artifacts"]["manifest"]).read_text(encoding="utf-8")
    assert "data:image" not in manifest and "test-xai" not in manifest


def test_keyframe_overlong_rejected_before_artifacts_or_submission(keyframe_setup):
    s = keyframe_setup
    s.media.duration = 15.01
    with pytest.raises(McsfxError) as caught:
        submit(s, profile="grok-sfx")
    assert caught.value.code == "MEDIA_CONSTRAINT"
    assert not s.out.exists()
    assert s.provider.submits == 0


def test_keyframe_extraction_failure_never_submits(keyframe_setup):
    s = keyframe_setup
    s.media.extraction_error = McsfxError("MEDIA_DECODE_FAILED", "Failed to decode.")
    with pytest.raises(McsfxError) as caught:
        submit(s, profile="grok-sfx")
    assert caught.value.code == "MEDIA_DECODE_FAILED"
    assert s.runner.current["remote_status"] == "not_submitted"
    assert s.provider.submits == 0


def test_keyframe_preserve_still_enforces_timeline(keyframe_setup):
    s = keyframe_setup
    s.media.preserve_error = McsfxError("TIMELINE_MISMATCH", "Strict duration mismatch.")
    job = submit(s, profile="grok-sfx", mode="preserve")
    for _ in range(2):
        with pytest.raises(McsfxError) as caught:
            s.runner.fetch(job)
        assert caught.value.code == "TIMELINE_MISMATCH"
    assert s.provider.submits == 1
    assert len(s.downloads) == 1
    assert Path(s.runner.current["artifacts"]["generated"]).read_bytes() == b"generated"


def test_keyframe_dry_run_without_credentials_or_durable_writes(keyframe_setup, monkeypatch):
    s = keyframe_setup
    monkeypatch.delenv("XAI_API_KEY")
    result = s.runner.generate(
        s.source,
        profile_name="grok-sfx",
        mode="regenerate",
        prompt="still reference",
        out=s.out,
        dry_run=True,
    )
    assert result["request"]["duration"] == 6
    assert result["keyframe"]["duration"] == 0
    assert not s.out.exists()
    assert not s.runner.settings.state_dir.exists()
    assert s.provider.submits == 0


def test_content_download_failure_recovers_using_saved_profile(keyframe_setup):
    s = keyframe_setup
    profile = replace(
        s.runner.settings.profile("grok-sfx"),
        base_url="https://relay.example/v1",
        download_mode="same_origin_content",
    )
    s.runner.settings = replace(s.runner.settings, profiles={"grok-sfx": profile})
    job = submit(s, profile="grok-sfx", mode="regenerate")
    attempts = []

    def download(url, path, **kwargs):
        attempts.append(kwargs)
        assert kwargs == {"profile": profile, "remote_id": "remote-123"}
        if len(attempts) == 1:
            raise McsfxError("DOWNLOAD_FAILED", "Network failure.", retryable=True)
        path.write_bytes(b"generated")

    s.runner.downloader = download
    with pytest.raises(McsfxError):
        s.runner.fetch(job)
    s.runner.settings = Settings(state_dir=s.runner.settings.state_dir)
    assert s.runner.fetch(job)["status"] == "completed"
    assert len(attempts) == 2
    assert s.provider.submits == 1


def test_dry_run_is_offline_and_does_not_write(setup):
    s = setup
    result = s.runner.generate(
        s.source, profile_name="grok", mode="preserve", prompt="impact", out=s.out, dry_run=True
    )
    assert result["status"] == "validated"
    assert s.provider.submits == 0
    assert not s.out.exists()
    assert not s.runner.settings.state_dir.exists()


def test_submit_fetch_and_repeat_do_not_resubmit(setup):
    s = setup
    job = submit(s)
    assert s.runner.current["status"] == "queued"
    result = s.runner.fetch(job, wait=True)
    assert result["status"] == "completed"
    assert Path(result["artifacts"]["result"]).read_bytes() == b"originalgenerated"
    assert s.runner.fetch(job) == result
    assert s.provider.submits == 1
    assert len(s.downloads) == 1
    assert s.source.read_bytes() == b"original"
    text = Path(result["artifacts"]["manifest"]).read_text(encoding="utf-8")
    assert "token=SECRET" not in text and "data:video" not in text
    assert "test-ak" not in text and "test-sk" not in text


def test_regenerate_does_not_remux(setup):
    s = setup
    s.media.preserve_error = AssertionError("must not remux")
    result = s.runner.fetch(submit(s, mode="regenerate"))
    assert result["artifacts"]["result"] == result["artifacts"]["generated"]


def test_unknown_submission_never_retried_or_cleaned(setup):
    s = setup
    s.provider.submit_error = McsfxError("SUBMISSION_UNKNOWN", "Outcome unknown.")
    with pytest.raises(McsfxError, match="Outcome unknown"):
        submit(s, profile="seedance")
    record = s.runner.current
    assert record["status"] == "submission_unknown"
    with pytest.raises(McsfxError) as exc:
        s.runner.fetch(record["job_id"], wait=True)
    assert exc.value.code == "SUBMISSION_UNKNOWN"
    with pytest.raises(McsfxError) as exc:
        s.runner.cleanup(record["job_id"])
    assert exc.value.code == "CLEANUP_UNSAFE"
    assert s.provider.submits == 1
    assert s.storage.deleted == []


def test_keyboard_interrupt_in_submit_is_unknown(setup):
    s = setup
    s.provider.submit_error = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        submit(s)
    assert s.runner.current["status"] == "submission_unknown"


def test_crash_in_submitting_state_is_never_replayed(setup):
    s = setup
    job = submit(s)
    record = s.runner.store.load(job)
    record.update(status="submitting", remote_id=None, remote_status="unknown")
    s.runner.store.save(record)
    with pytest.raises(McsfxError) as exc:
        s.runner.status(job)
    assert exc.value.code == "SUBMISSION_UNKNOWN"
    assert s.provider.submits == 1


def test_definite_rejection_can_cleanup_own_object(setup):
    s = setup
    s.provider.submit_error = McsfxError("AUTH_ERROR", "Authentication failed.")
    with pytest.raises(McsfxError):
        submit(s, profile="seedance")
    assert s.runner.current["remote_status"] == "not_submitted"
    assert len(s.storage.deleted) == 1


def test_wait_timeout_does_not_cancel_or_cleanup(setup):
    s = setup
    s.provider.remote_state = "running"
    times = iter([0, 10])
    s.runner.clock = lambda: next(times)
    job = submit(s, profile="seedance")
    with pytest.raises(McsfxError) as exc:
        s.runner.fetch(job, wait=True, timeout=1)
    assert exc.value.code == "WAIT_TIMEOUT"
    assert s.runner.current["remote_status"] == "running"
    assert s.storage.deleted == []
    assert s.provider.submits == 1


def test_fetch_without_wait_returns_running(setup):
    s = setup
    s.provider.remote_state = "running"
    result = s.runner.fetch(submit(s))
    assert result["status"] == "running" and result["ok"]
    assert not s.downloads


def test_failed_download_recovers_without_submit(setup):
    s = setup
    download = s.runner.downloader
    job = submit(s)

    def fail(url, target):
        raise McsfxError("DOWNLOAD_FAILED", "Download interrupted.", retryable=True)

    s.runner.downloader = fail
    with pytest.raises(McsfxError):
        s.runner.fetch(job)
    s.runner.downloader = download
    assert s.runner.fetch(job)["status"] == "completed"
    assert s.provider.submits == 1


def test_existing_download_is_reused_after_crash(setup):
    s = setup
    job = submit(s)
    s.runner.status(job)
    path = Path(s.runner.current["output_dir"]) / "generated.mp4"
    path.write_bytes(b"generated")
    assert s.runner.fetch(job)["status"] == "completed"
    assert s.downloads == []


def test_timeline_failure_retains_generated_video(setup):
    s = setup
    job = submit(s)
    s.media.preserve_error = McsfxError("TIMELINE_MISMATCH", "Duration mismatch.")
    with pytest.raises(McsfxError) as exc:
        s.runner.fetch(job)
    assert exc.value.code == "TIMELINE_MISMATCH"
    assert Path(s.runner.current["artifacts"]["generated"]).exists()
    assert s.runner.current["status"] == "local_failed"
    with pytest.raises(McsfxError) as exc:
        s.runner.status(job)
    assert exc.value.code == "TIMELINE_MISMATCH"
    assert s.provider.submits == 1


def test_cleanup_failure_does_not_discard_success(setup):
    s = setup
    s.storage.fail_delete = True
    job = submit(s, profile="seedance")
    result = s.runner.fetch(job)
    assert result["ok"] and "cleanup_pending" in result["warnings"]
    s.storage.fail_delete = False
    assert "cleanup_pending" not in s.runner.cleanup(job)["warnings"]
    assert s.storage.deleted[0][1] == "version-1"


def test_cleanup_rejects_foreign_object(setup):
    s = setup
    job = submit(s, profile="seedance")
    record = s.runner.store.load(job)
    record["remote_status"] = "succeeded"
    record["upload"]["key"] = "someone-else/video.mp4"
    s.runner.store.save(record)
    with pytest.raises(McsfxError) as exc:
        s.runner.cleanup(job)
    assert exc.value.code == "CLEANUP_UNSAFE"
    assert s.storage.deleted == []


def test_completed_artifact_tampering_is_reported(setup):
    s = setup
    job = submit(s)
    result = s.runner.fetch(job)
    Path(result["artifacts"]["result"]).write_bytes(b"changed")
    with pytest.raises(McsfxError) as exc:
        s.runner.fetch(job)
    assert exc.value.code == "ARTIFACT_CHANGED"
    assert s.provider.submits == 1


def test_job_lock_is_exclusive_and_released(setup):
    s = setup
    job = submit(s)
    with s.runner.store.lock(job):
        with pytest.raises(McsfxError) as exc, JobStore(s.runner.settings.state_dir).lock(job):
            pytest.fail("second lock acquired")
        assert exc.value.code == "JOB_BUSY"
    with s.runner.store.lock(job):
        pass


@pytest.mark.parametrize("job_id", ["../other", "C:/outside", "", "abc"])
def test_job_id_cannot_escape_registry(tmp_path, job_id):
    with pytest.raises(McsfxError):
        JobStore(tmp_path).directory(job_id)


def test_crash_after_result_publication_recovers_without_remux(setup):
    s = setup
    job = submit(s)
    save = s.runner._save

    def interrupt_completed(record):
        if record["status"] == "completed":
            raise KeyboardInterrupt
        save(record)

    s.runner._save = interrupt_completed
    with pytest.raises(KeyboardInterrupt):
        s.runner.fetch(job)
    assert (Path(s.runner.current["output_dir"]) / "result.mp4").is_file()
    s.runner._save = save
    s.media.preserve_error = AssertionError("must reuse committed checksum")
    assert s.runner.fetch(job)["status"] == "completed"
    assert s.provider.submits == 1
