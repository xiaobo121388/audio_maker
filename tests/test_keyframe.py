import json
import traceback
from dataclasses import replace

import httpx
import pytest

from mcsfx import providers
from mcsfx.config import BUILTINS, Profile, load_settings, validate_profile
from mcsfx.errors import McsfxError
from mcsfx.media import keyframe_duration
from mcsfx.providers import Provider, RemoteResult, download_video

SECRET = "mock-keyframe-secret"
IMAGE = "data:image/png;base64,iVBORw0KGgo="
CONTENT_PROFILE = replace(
    BUILTINS["grok-sfx"],
    base_url="https://relay.example/prefix/v1/",
    download_mode="same_origin_content",
)
CONTENT_PATH = "/prefix/v1/videos/remote-123/content"
CONTENT_URL = "https://relay.example" + CONTENT_PATH


@pytest.fixture(autouse=True)
def mock_key(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", SECRET)


def test_profile_schema_and_old_snapshot_defaults(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        '[profiles.grok-sfx]\nbase_url="https://relay.example/v1"\n'
        'download_mode="same_origin_content"\n',
        encoding="utf-8",
    )
    profile = load_settings(str(config)).profile("grok-sfx")
    assert profile.preset == "grok-keyframe"
    assert profile.download_mode == "same_origin_content"
    assert profile.transport == "data_uri"
    assert profile.model == "grok-imagine-video"
    old = BUILTINS["grok"].snapshot()
    old.pop("download_mode")
    assert Profile(**old).download_mode == "public"
    assert BUILTINS["grok"].preset == "grok-edit"


@pytest.mark.parametrize(
    "name,changes",
    [
        ("seedance", {"download_mode": "same_origin_content"}),
        ("grok-sfx", {"transport": "tos"}),
        ("grok-sfx", {"download_mode": "any_url"}),
        ("grok-sfx", {"protocol": "ark"}),
    ],
)
def test_keyframe_invalid_profiles(name, changes):
    with pytest.raises(McsfxError):
        validate_profile(replace(BUILTINS[name], **changes))


@pytest.mark.parametrize(
    "seconds,expected", [(0.1, 1), (1, 1), (2, 2), (5.01, 6), (14.9, 15), (15, 15)]
)
def test_keyframe_duration_ceiling(seconds, expected):
    assert keyframe_duration(seconds) == expected


@pytest.mark.parametrize("seconds", [0, -1, 15.00001, 16, float("nan"), float("inf")])
def test_keyframe_duration_never_truncates(seconds):
    with pytest.raises(McsfxError) as caught:
        keyframe_duration(seconds)
    assert caught.value.code == "MEDIA_CONSTRAINT"


@pytest.mark.parametrize("duration", [1, 2, 6, 15])
def test_keyframe_generation_fields(duration):
    calls = []

    def handle(request):
        calls.append(request)
        assert request.method == "POST"
        assert str(request.url) == "https://api.x.ai/v1/videos/generations"
        assert json.loads(request.content) == {
            "model": "grok-imagine-video",
            "prompt": "action and sound",
            "image": {"url": IMAGE},
            "duration": duration,
            "resolution": "480p",
            "generate_audio": True,
        }
        return httpx.Response(200, json={"request_id": "remote-123"})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        assert (
            Provider(BUILTINS["grok-sfx"], client=client).submit(
                IMAGE, "action and sound", duration=duration
            )
            == "remote-123"
        )
    assert len(calls) == 1


@pytest.mark.parametrize("duration", [None, 0, 16, -1, True, 1.5, "5", float("nan")])
def test_keyframe_invalid_duration_never_submits(duration):
    with (
        httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail("Submitted"))) as client,
        pytest.raises(McsfxError) as caught,
    ):
        Provider(BUILTINS["grok-sfx"], client=client).submit(IMAGE, "sound", duration=duration)
    assert caught.value.code == "INPUT_ERROR"


@pytest.mark.parametrize("url", [CONTENT_PATH, CONTENT_URL])
@pytest.mark.parametrize("explicit_status", [True, False])
def test_content_result_exact_saved_endpoint(url, explicit_status):
    data = {"video": {"url": url}}
    if explicit_status:
        data["status"] = "done"
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=data))
    ) as client:
        result = Provider(CONTENT_PROFILE, client=client).query("remote-123")
    assert result == RemoteResult("succeeded", video_url=CONTENT_URL)


@pytest.mark.parametrize(
    "url",
    [
        "/v1/videos/remote-123/content",
        "videos/remote-123/content",
        "//relay.example" + CONTENT_PATH,
        "/prefix/v1/videos/other-id/content",
        CONTENT_PATH + "/",
        CONTENT_PATH + "?token=anything",
        CONTENT_PATH + "#fragment",
        "/prefix/v1/videos/other/../remote-123/content",
        "/prefix/v1/videos/remote%2D123/content",
        "/prefix/v1/videos/remote-123/%63ontent",
        "https://evil.example" + CONTENT_PATH,
        "https://relay.example.evil.example" + CONTENT_PATH,
        "https://relay.example:444" + CONTENT_PATH,
        "https://relay.example:443" + CONTENT_PATH,
        "https://relay.example@evil.example" + CONTENT_PATH,
        "http://relay.example" + CONTENT_PATH,
        CONTENT_URL + "?x=1",
        CONTENT_URL + "#x",
        CONTENT_URL + " ",
        CONTENT_PATH + "\\other",
        None,
        {"url": CONTENT_URL},
    ],
)
def test_content_rejects_untrusted_endpoint_before_download(url, tmp_path):
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda _: pytest.fail("Leaked request"))
        ) as client,
        pytest.raises(McsfxError) as caught,
    ):
        download_video(
            url,
            tmp_path / "v.mp4",
            profile=CONTENT_PROFILE,
            remote_id="remote-123",
            client=client,
        )
    assert caught.value.code == "PROTOCOL_ERROR"
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json={"status": "done", "video": {"url": url}})
            )
        ) as client,
        pytest.raises(McsfxError) as caught,
    ):
        Provider(CONTENT_PROFILE, client=client).query("remote-123")
    assert caught.value.code == "PROTOCOL_ERROR"
    assert not (tmp_path / "v.mp4").exists()


@pytest.mark.parametrize("redirect", ["https://cdn.example/result.mp4", "/other", CONTENT_PATH])
def test_content_bearer_only_on_initial_request(tmp_path, redirect):
    calls = []

    def handle(request):
        calls.append(request)
        assert "Cookie" not in request.headers
        assert "X-Private" not in request.headers
        assert "injected" not in request.url.params
        assert request.headers["Accept-Encoding"] == "identity"
        if len(calls) == 1:
            assert str(request.url) == CONTENT_URL
            assert request.headers["Authorization"] == f"Bearer {SECRET}"
            return httpx.Response(302, headers={"Location": redirect, "Set-Cookie": "private=1"})
        assert "Authorization" not in request.headers
        if len(calls) == 2:
            return httpx.Response(307, headers={"Location": CONTENT_URL})
        return httpx.Response(200, content=b"video")

    with httpx.Client(
        transport=httpx.MockTransport(handle),
        auth=("private", "password"),
        headers={"Authorization": "untrusted-default", "X-Private": "secret"},
        cookies={"secret": "cookie"},
        params={"injected": "secret"},
        follow_redirects=True,
    ) as client:
        destination = tmp_path / "v.mp4"
        download_video(
            CONTENT_PATH,
            destination,
            profile=CONTENT_PROFILE,
            remote_id="remote-123",
            client=client,
        )
        assert not client.is_closed
    assert destination.read_bytes() == b"video"
    assert len(calls) == 3


def assert_redacted(error):
    text = json.dumps(error.as_dict()) + "".join(traceback.format_exception(error))
    assert SECRET not in text
    assert "upstream-private" not in text
    assert error.__cause__ is None


@pytest.mark.parametrize(
    "reference",
    ["https://example.test/a.png", "data:video/mp4;base64,AAAA", "data:image/png;base64,"],
)
def test_keyframe_rejects_non_png_reference(reference):
    with (
        httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail("Submitted"))) as client,
        pytest.raises(McsfxError),
    ):
        Provider(BUILTINS["grok-sfx"], client=client).submit(reference, "sound", duration=5)


@pytest.mark.parametrize("failure", ["timeout", "malformed", "secret_id", 502, 307])
def test_keyframe_ambiguous_submission_is_once_and_redacted(failure):
    calls = []

    def handle(request):
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout(SECRET, request=request)
        if failure == "malformed":
            return httpx.Response(200, text=SECRET)
        if failure == "secret_id":
            return httpx.Response(200, json={"request_id": SECRET})
        return httpx.Response(failure, text=SECRET)

    with (
        httpx.Client(transport=httpx.MockTransport(handle)) as client,
        pytest.raises(McsfxError) as caught,
    ):
        Provider(BUILTINS["grok-sfx"], client=client, sleep=lambda _: pytest.fail("Retry")).submit(
            IMAGE, "sound", duration=5
        )
    assert len(calls) == 1
    assert caught.value.code == "SUBMISSION_UNKNOWN"
    assert_redacted(caught.value)


@pytest.mark.parametrize(
    "failure",
    [
        "size",
        "stream-size",
        "empty",
        "retry",
        "transport",
        "stream",
        "exists",
        "race",
        "http",
        "loop",
    ],
)
def test_content_download_retains_safety_limits(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(providers, "MAX_DOWNLOAD_BYTES", 5)
    destination = tmp_path / "v.mp4"
    calls = []
    sleeps = []
    if failure == "exists":
        destination.write_bytes(b"keep")

    class BrokenStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b"part"
            raise httpx.ReadError(SECRET)

    def handle(request):
        calls.append(request)
        if failure == "loop":
            assert ("Authorization" in request.headers) == (len(calls) == 1)
            return httpx.Response(302, headers={"Location": CONTENT_PATH})
        assert request.headers["Authorization"] == f"Bearer {SECRET}"
        if failure in {"retry", "transport", "stream"} and len(calls) < 3:
            if failure == "transport":
                raise httpx.ConnectError(SECRET, request=request)
            if failure == "stream":
                return httpx.Response(200, stream=BrokenStream())
            return httpx.Response(503, text="upstream-private " + SECRET)
        if failure == "size":
            return httpx.Response(200, headers={"Content-Length": "6"}, content=b"123456")
        if failure == "stream-size":
            return httpx.Response(200, stream=httpx.ByteStream(b"123456"))
        if failure == "empty":
            return httpx.Response(200, content=b"")
        if failure == "race":
            destination.write_bytes(b"keep")
        if failure == "http":
            return httpx.Response(403, text="upstream-private " + SECRET)
        return httpx.Response(200, content=b"video")

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        if failure in {"retry", "transport", "stream"}:
            download_video(
                CONTENT_PATH,
                destination,
                client=client,
                sleep=sleeps.append,
                profile=CONTENT_PROFILE,
                remote_id="remote-123",
            )
            assert destination.read_bytes() == b"video"
            assert len(calls) == 3 and len(sleeps) == 2
        else:
            with pytest.raises(McsfxError) as caught:
                download_video(
                    CONTENT_PATH,
                    destination,
                    client=client,
                    sleep=sleeps.append,
                    profile=CONTENT_PROFILE,
                    remote_id="remote-123",
                )
            assert_redacted(caught.value)
            if failure in {"exists", "race"}:
                assert caught.value.code == "OUTPUT_EXISTS"
                assert destination.read_bytes() == b"keep"
            else:
                assert not destination.exists()
            assert len(calls) == {"exists": 0, "empty": 3, "loop": providers.MAX_REDIRECTS + 1}.get(
                failure, 1
            )
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize(
    "location", ["http://evil.example/a", "//user:password@evil.example/a", "file:///tmp/a"]
)
def test_content_unsafe_redirect_not_requested(tmp_path, location):
    calls = []

    def handle(request):
        calls.append(request)
        assert str(request.url) == CONTENT_URL
        return httpx.Response(302, headers={"Location": location})

    with (
        httpx.Client(transport=httpx.MockTransport(handle)) as client,
        pytest.raises(McsfxError),
    ):
        download_video(
            CONTENT_PATH,
            tmp_path / "v.mp4",
            client=client,
            profile=CONTENT_PROFILE,
            remote_id="remote-123",
        )
    assert len(calls) == 1


@pytest.mark.parametrize(
    "base_url", ["https://relay.example/prefix/../v1", "https://relay.example/v%31"]
)
def test_content_normalized_base_cannot_expand_authorized_path(tmp_path, base_url):
    profile = replace(CONTENT_PROFILE, base_url=base_url)
    with (
        httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail("Request"))) as client,
        pytest.raises(McsfxError),
    ):
        download_video(
            base_url + "/videos/remote-123/content",
            tmp_path / "v.mp4",
            client=client,
            profile=profile,
            remote_id="remote-123",
        )


def test_public_profile_does_not_accept_relative_content_or_enable_auth(tmp_path):
    with httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail("Request"))) as client:
        with pytest.raises(McsfxError):
            download_video(CONTENT_PATH, tmp_path / "v.mp4", client=client)
        with pytest.raises(McsfxError):
            download_video(
                CONTENT_URL,
                tmp_path / "v.mp4",
                client=client,
                profile=BUILTINS["grok-sfx"],
                remote_id="remote-123",
            )
