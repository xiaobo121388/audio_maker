import json
import traceback
from dataclasses import replace

import httpx
import pytest

from mcsfx import providers
from mcsfx.config import BUILTINS
from mcsfx.errors import McsfxError
from mcsfx.providers import Provider, RemoteResult, download_video

SECRET = "test-secret-not-a-real-key"
VIDEO_URL = "https://media.example/video.mp4?signature=private-url-token"
UPSTREAM = f"upstream-private-text {SECRET} {VIDEO_URL}"


@pytest.fixture(autouse=True)
def api_keys(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", SECRET)
    monkeypatch.setenv("ARK_API_KEY", SECRET)


def assert_redacted(error):
    rendered = json.dumps(error.as_dict()) + "".join(traceback.format_exception(error))
    for sensitive in (SECRET, VIDEO_URL, "upstream-private-text", "private-url-token"):
        assert sensitive not in rendered
    assert error.__cause__ is None


def test_submission_receipt_retains_safe_http_status_and_request_id():
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    502, text=UPSTREAM, headers={"x-request-id": "request-safe-123"}
                )
            )
        ) as client,
        pytest.raises(McsfxError) as caught,
    ):
        Provider(BUILTINS["grok"], client=client).submit(VIDEO_URL, "prompt")
    assert caught.value.details == {"http_status": 502, "provider_request_id": "request-safe-123"}
    assert_redacted(caught.value)


@pytest.mark.parametrize("request_id", [SECRET, VIDEO_URL, UPSTREAM])
def test_submission_receipt_does_not_echo_sensitive_request_ids(request_id):
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(502, text=UPSTREAM, headers={"x-request-id": request_id})
            )
        ) as client,
        pytest.raises(McsfxError) as caught,
    ):
        Provider(BUILTINS["grok"], client=client).submit(VIDEO_URL, "prompt")
    assert caught.value.details == {"http_status": 502}
    assert_redacted(caught.value)


@pytest.mark.parametrize("name", ["grok", "seedance"])
@pytest.mark.parametrize("relay", [False, True])
def test_exact_submit_protocol_and_custom_relay(name, relay):
    profile = BUILTINS[name]
    if relay:
        profile = replace(
            profile,
            base_url="https://relay.example/prefix/api/",
            model="custom-model",
            api_key_env="RELAY_API_KEY",
        )
    reference = "data:video/mp4;base64,AAAA" if name == "grok" else VIDEO_URL
    requests = []

    def handle(request):
        requests.append(request)
        assert request.method == "POST"
        assert request.headers["Authorization"] == f"Bearer {SECRET}"
        assert request.headers["Content-Type"] == "application/json"
        assert request.extensions["timeout"]["read"] == 60.0
        body = json.loads(request.content)
        if name == "grok":
            assert str(request.url) == profile.base_url.rstrip("/") + "/videos/edits"
            assert body == {
                "model": profile.model,
                "prompt": "add synchronized sound",
                "video": {"url": reference},
            }
            return httpx.Response(200, json={"request_id": "request-123"})
        assert str(request.url) == profile.base_url.rstrip("/") + "/contents/generations/tasks"
        assert body == {
            "model": profile.model,
            "content": [
                {"type": "text", "text": "add synchronized sound"},
                {"type": "video_url", "video_url": {"url": reference}, "role": "reference_video"},
            ],
            "generate_audio": True,
            "omni_reference_task_type": "edit",
            "ratio": "adaptive",
            "duration": -1,
            "resolution": "720p",
        }
        return httpx.Response(201, json={"id": "request-123"})

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("RELAY_API_KEY", SECRET)
        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            provider = Provider(profile, client=client, sleep=lambda _: pytest.fail("POST retried"))
            assert provider.submit(reference, "add synchronized sound") == "request-123"
    assert len(requests) == 1


@pytest.mark.parametrize("name", ["grok", "seedance"])
@pytest.mark.parametrize(
    "failure",
    [
        "timeout",
        "connect",
        "read",
        "decode",
        408,
        500,
        502,
        503,
        307,
        "invalid_json",
        "array",
        "empty",
        "missing_id",
        "wrong_id",
        "unsafe_id",
        "error_with_id",
    ],
)
def test_submission_ambiguity_never_retries(name, failure):
    calls = []
    sleeps = []

    def handle(request):
        calls.append(request)
        exceptions = {
            "timeout": httpx.ReadTimeout,
            "connect": httpx.ConnectError,
            "read": httpx.ReadError,
            "decode": httpx.DecodingError,
        }
        if failure in exceptions:
            raise exceptions[failure](UPSTREAM, request=request)
        if isinstance(failure, int):
            return httpx.Response(failure, text=UPSTREAM, headers={"Location": VIDEO_URL})
        id_field = "request_id" if name == "grok" else "id"
        bodies = {"invalid_json": b"not json", "empty": b""}
        if failure in bodies:
            return httpx.Response(200, content=bodies[failure])
        payloads = {
            "array": [],
            "missing_id": {"error": UPSTREAM},
            "wrong_id": {id_field: 123},
            "unsafe_id": {id_field: VIDEO_URL},
            "error_with_id": {id_field: "task-1", "error": UPSTREAM},
        }
        return httpx.Response(200, json=payloads[failure])

    with httpx.Client(transport=httpx.MockTransport(handle), follow_redirects=True) as client:
        provider = Provider(BUILTINS[name], client=client, sleep=sleeps.append)
        with pytest.raises(McsfxError) as caught:
            provider.submit(VIDEO_URL, "prompt")
    assert caught.value.code == "SUBMISSION_UNKNOWN"
    assert not caught.value.retryable
    assert_redacted(caught.value)
    assert len(calls) == 1
    assert sleeps == []


@pytest.mark.parametrize("method", ["submit", "query"])
@pytest.mark.parametrize(
    "status,code",
    [
        (400, "PROVIDER_ERROR"),
        (401, "AUTH_ERROR"),
        (403, "AUTH_ERROR"),
        (404, "PROVIDER_ERROR"),
        (422, "PROVIDER_ERROR"),
    ],
)
def test_definitive_http_errors_are_redacted_without_retries(method, status, code):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(status, text=UPSTREAM)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        provider = Provider(BUILTINS["grok"], client=client)
        with pytest.raises(McsfxError) as caught:
            if method == "submit":
                provider.submit(VIDEO_URL, "prompt")
            else:
                provider.query("request-1")
    assert caught.value.code == code
    assert not caught.value.retryable
    assert_redacted(caught.value)
    assert len(calls) == 1


def test_submission_rate_limit_is_not_retried():
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(429))) as client:
        provider = Provider(
            BUILTINS["grok"], client=client, sleep=lambda _: pytest.fail("POST retried")
        )
        with pytest.raises(McsfxError, match="rate limit") as caught:
            provider.submit(VIDEO_URL, "prompt")
    assert caught.value.code == "RATE_LIMITED"
    assert not caught.value.retryable


@pytest.mark.parametrize("name", ["grok", "seedance"])
@pytest.mark.parametrize(
    "state", ["queued", "running", "succeeded", "failed", "cancelled", "expired"]
)
def test_normalized_states_and_safe_failure_messages(name, state):
    profile = replace(BUILTINS[name], base_url="https://relay.example/nested/v1/")
    payload = {"status": state}
    if state == "succeeded":
        payload.update(
            {"video": {"url": VIDEO_URL}}
            if name == "grok"
            else {"content": {"video_url": VIDEO_URL}}
        )
    if state in ("failed", "cancelled", "expired"):
        payload["error"] = {"code": SECRET, "message": UPSTREAM}

    def handle(request):
        assert request.method == "GET"
        route = "/videos" if name == "grok" else "/contents/generations/tasks"
        assert str(request.url) == profile.base_url.rstrip("/") + route + "/task-1"
        assert request.headers["Authorization"] == f"Bearer {SECRET}"
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        result = Provider(profile, client=client).query("task-1")
    assert result.status == state
    assert result.video_url == (VIDEO_URL if state == "succeeded" else None)
    if state in ("failed", "cancelled", "expired"):
        assert result.error
        assert SECRET not in result.error
        assert "https://" not in result.error
    else:
        assert result.error is None


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"status": "pending"}, RemoteResult("running")),
        ({"status": "done", "video": {"url": VIDEO_URL}}, RemoteResult("succeeded", VIDEO_URL)),
        (
            {
                "video": {"url": VIDEO_URL, "respect_moderation": True},
                "model": "grok-imagine-video",
            },
            RemoteResult("succeeded", VIDEO_URL),
        ),
        ({"status": "running", "video": {"url": VIDEO_URL}}, RemoteResult("running")),
    ],
)
def test_xai_documented_and_statusless_completed_results(payload, expected):
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    ) as c:
        assert Provider(BUILTINS["grok"], client=c).query("request-1") == expected


def test_xai_empty_202_is_running_not_completed():
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(202))) as client:
        assert Provider(BUILTINS["grok"], client=client).query("task-1") == RemoteResult("running")


@pytest.mark.parametrize("status", [None, "done"])
def test_xai_moderation_rejection_is_failed(status):
    payload = {"video": {"url": VIDEO_URL, "respect_moderation": False}}
    if status:
        payload["status"] = status
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    ) as c:
        result = Provider(BUILTINS["grok"], client=c).query("task-1")
    assert result.status == "failed"
    assert result.video_url is None
    assert "moderation" in result.error


@pytest.mark.parametrize(
    "name,payload,http_status",
    [
        ("grok", {}, 200),
        ("grok", [], 200),
        ("grok", None, 200),
        ("grok", {"status": "new-state", "video": {"url": VIDEO_URL}}, 200),
        ("grok", {"status": None, "video": {"url": VIDEO_URL}}, 200),
        ("grok", {"status": []}, 200),
        ("grok", {"status": "done"}, 200),
        ("grok", {"status": "done", "video": {"url": "http://media.example/video"}}, 200),
        (
            "grok",
            {"status": "done", "video": {"url": "https://user:password@media.example/v"}},
            200,
        ),
        ("grok", {"status": "done", "video": {"url": VIDEO_URL}, "error": UPSTREAM}, 200),
        ("grok", {"video": {"url": VIDEO_URL}, "error": UPSTREAM}, 200),
        ("grok", {"video": {"url": VIDEO_URL}}, 202),
        ("grok", {"status": "done", "video": {"url": VIDEO_URL}}, 202),
        (
            "grok",
            {"status": "done", "video": {"url": VIDEO_URL, "respect_moderation": "false"}},
            200,
        ),
        ("seedance", {"content": {"video_url": VIDEO_URL}}, 200),
        ("seedance", {"status": "done", "content": {"video_url": VIDEO_URL}}, 200),
        ("seedance", {"status": "succeeded"}, 200),
        ("seedance", {"status": "succeeded", "content": []}, 200),
        ("seedance", {"status": "succeeded", "content": {"video_url": 1}}, 200),
        ("seedance", {"status": "succeeded", "video_url": VIDEO_URL}, 200),
        ("seedance", {"status": "running", "error": UPSTREAM}, 200),
    ],
)
def test_unknown_or_malformed_results_are_never_assumed_complete(name, payload, http_status):
    def handle(_):
        return httpx.Response(http_status, content=json.dumps(payload).encode())

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        provider = Provider(
            BUILTINS[name], client=client, sleep=lambda _: pytest.fail("Invalid schema retried")
        )
        with pytest.raises(McsfxError) as caught:
            provider.query("task-1")
    assert caught.value.code == "PROTOCOL_ERROR"
    assert_redacted(caught.value)


@pytest.mark.parametrize("name", ["grok", "seedance"])
def test_query_invalid_json_is_redacted(name):
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, text=UPSTREAM))
        ) as c,
        pytest.raises(McsfxError) as caught,
    ):
        Provider(BUILTINS[name], client=c).query("task-1")
    assert caught.value.code == "PROTOCOL_ERROR"
    assert_redacted(caught.value)


@pytest.mark.parametrize("failure", ["transport", 429, 500, 502, 503])
@pytest.mark.parametrize("recover", [False, True])
def test_query_bounded_retries(failure, recover):
    calls = []
    sleeps = []

    def handle(request):
        calls.append(request)
        if recover and len(calls) == 3:
            return httpx.Response(200, json={"status": "pending"})
        if failure == "transport":
            raise httpx.ReadTimeout(UPSTREAM, request=request)
        return httpx.Response(failure, text=UPSTREAM, headers={"Retry-After": "99999"})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        provider = Provider(BUILTINS["grok"], client=client, sleep=sleeps.append)
        if recover:
            assert provider.query("task-1").status == "running"
        else:
            with pytest.raises(McsfxError) as caught:
                provider.query("task-1")
            assert caught.value.retryable
            assert_redacted(caught.value)
    assert len(calls) == 3
    assert len(sleeps) == 2
    assert all(0 < delay <= 30 for delay in sleeps)


def test_provider_does_not_follow_redirects_or_merge_injected_credentials():
    requests = []

    def handle(request):
        requests.append(request)
        assert request.headers["Authorization"] == f"Bearer {SECRET}"
        assert "Cookie" not in request.headers
        assert "X-Api-Key" not in request.headers
        assert not request.url.query
        return httpx.Response(302, headers={"Location": VIDEO_URL}, text=UPSTREAM)

    with (
        httpx.Client(
            transport=httpx.MockTransport(handle),
            follow_redirects=True,
            auth=("client-user", "client-password"),
            cookies={"auth": SECRET},
            headers={"X-Api-Key": SECRET},
            params={"api_key": SECRET},
        ) as client,
        pytest.raises(McsfxError) as caught,
    ):
        Provider(BUILTINS["grok"], client=client).query("task-1")
    assert_redacted(caught.value)
    assert len(requests) == 1


@pytest.mark.parametrize("method", ["query", "submit"])
@pytest.mark.parametrize("key", [None, "", " ", "secret\r\nInjected: true", "non-ascii-\u00e9"])
def test_missing_or_invalid_credentials_fail_before_network(method, key, monkeypatch):
    if key is None:
        monkeypatch.delenv("XAI_API_KEY")
    else:
        monkeypatch.setenv("XAI_API_KEY", key)
    with httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail("Unexpected HTTP"))) as c:
        provider = Provider(BUILTINS["grok"], client=c)
        with pytest.raises(McsfxError) as caught:
            if method == "query":
                provider.query("task-1")
            else:
                provider.submit(VIDEO_URL, "prompt")
    assert caught.value.code == "AUTH_ERROR"
    assert_redacted(caught.value)


@pytest.mark.parametrize(
    "remote_id",
    [
        "",
        "..",
        "../private",
        "id/path",
        "id?api_key=secret",
        "https://other.example",
        "id#fragment",
        "%2e%2e",
        1,
    ],
)
def test_remote_id_cannot_change_request_path(remote_id):
    with (
        httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail("Unexpected HTTP"))) as c,
        pytest.raises(McsfxError) as caught,
    ):
        Provider(BUILTINS["grok"], client=c).query(remote_id)
    assert caught.value.code == "INPUT_ERROR"


def test_provider_client_ownership(monkeypatch):
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200))) as borrowed:
        Provider(BUILTINS["grok"], client=borrowed).close()
        assert not borrowed.is_closed
    owned = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
    monkeypatch.setattr(providers, "_new_client", lambda: owned)
    provider = Provider(BUILTINS["grok"])
    provider.close()
    provider.close()
    assert owned.is_closed


class VideoStream(httpx.SyncByteStream):
    def __init__(self, chunks, *, failure=None, inspect=None):
        self.chunks = chunks
        self.failure = failure
        self.inspect = inspect
        self.closed = False

    def __iter__(self):
        for chunk in self.chunks:
            if self.inspect:
                self.inspect()
            yield chunk
        if self.failure:
            raise self.failure

    def close(self):
        self.closed = True


def test_download_streams_to_unique_part_and_installs_atomically(tmp_path):
    destination = tmp_path / "video.mp4"
    stale = tmp_path / "unrelated.part"
    stale.write_bytes(b"keep")
    observed = []

    def inspect():
        assert not destination.exists()
        parts = list(tmp_path.glob(".mcsfx-*.part"))
        assert len(parts) == 1
        observed.append(parts[0])

    stream = VideoStream([b"video-", b"payload"], inspect=inspect)
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                stream=stream,
                headers={"Content-Length": "13"},
            )
        )
    ) as client:
        download_video(VIDEO_URL, destination, client=client)
        assert not client.is_closed
    assert destination.read_bytes() == b"video-payload"
    assert stream.closed
    assert len(observed) == 2
    assert not list(tmp_path.glob(".mcsfx-*.part"))
    assert stale.read_bytes() == b"keep"


def test_download_redirects_are_unauthenticated_even_with_client_defaults(tmp_path):
    requests = []

    def handle(request):
        requests.append(request)
        for header in ("Authorization", "Proxy-Authorization", "X-Api-Key", "Cookie", "Referer"):
            assert header not in request.headers
        assert "api_key" not in request.url.params
        assert request.headers["Accept-Encoding"] == "identity"
        if len(requests) == 1:
            return httpx.Response(
                302,
                headers={
                    "Location": "/redirect?signature=new-token",
                    "Set-Cookie": "secret=value; Path=/",
                },
            )
        if len(requests) == 2:
            return httpx.Response(307, headers={"Location": "//cdn.example/final.mp4"})
        return httpx.Response(200, content=b"video")

    with httpx.Client(
        transport=httpx.MockTransport(handle),
        follow_redirects=True,
        auth=("username", SECRET),
        cookies={"session": SECRET},
        headers={
            "Authorization": f"Bearer {SECRET}",
            "X-Api-Key": SECRET,
            "Proxy-Authorization": SECRET,
            "Referer": VIDEO_URL,
        },
        params={"api_key": SECRET},
    ) as client:
        download_video(VIDEO_URL, tmp_path / "video.mp4", client=client)
    assert len(requests) == 3
    assert str(requests[1].url) == "https://media.example/redirect?signature=new-token"
    assert str(requests[2].url) == "https://cdn.example/final.mp4"
    assert (tmp_path / "video.mp4").read_bytes() == b"video"


@pytest.mark.parametrize(
    "url",
    [
        "http://media.example/v",
        "file:///private",
        "data:video/mp4,abc",
        "https://user:password@media.example/v",
        "https://[invalid/v",
        "https://media.example:99999/v",
        "https://media.example/v\n",
        "https://media.example/v#secret",
        "https://media.example\\@private/v",
    ],
)
@pytest.mark.parametrize("redirect", [False, True])
def test_download_revalidates_every_url(url, redirect, tmp_path):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(302, headers={"Location": url})

    with (
        httpx.Client(transport=httpx.MockTransport(handle), follow_redirects=True) as client,
        pytest.raises(McsfxError) as caught,
    ):
        download_video(
            VIDEO_URL if redirect else url,
            tmp_path / "v.mp4",
            client=client,
            sleep=lambda _: None,
        )
    assert caught.value.code == "DOWNLOAD_FAILED"
    # HTTPX rejects control characters while building next_request, before our
    # redirect validator runs, as a transport error eligible for bounded retries.
    expected = providers.MAX_ATTEMPTS if redirect and "\n" in url else int(redirect)
    assert len(calls) == expected
    assert not list(tmp_path.iterdir())
    assert_redacted(caught.value)


@pytest.mark.parametrize("location", [None, "/loop"])
def test_download_missing_redirect_or_loop_is_bounded(location, tmp_path):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(302, headers={"Location": location} if location else {})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client, pytest.raises(McsfxError):
        download_video(VIDEO_URL, tmp_path / "v.mp4", client=client)
    assert len(calls) == (providers.MAX_REDIRECTS + 1 if location else 1)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("failure", ["transport", "stream", 429, 500, 503, "empty", "truncated"])
@pytest.mark.parametrize("recover", [False, True])
def test_download_retries_cleanly_and_redacts_errors(failure, recover, tmp_path):
    calls = []
    sleeps = []
    streams = []

    def handle(request):
        assert not list(tmp_path.glob(".mcsfx-*.part"))
        calls.append(request)
        if recover and len(calls) == 3:
            return httpx.Response(200, content=b"whole-video")
        if failure == "transport":
            raise httpx.ConnectError(UPSTREAM, request=request)
        if failure == "stream":
            stream = VideoStream([b"x" * 65536], failure=httpx.ReadError(UPSTREAM, request=request))
            streams.append(stream)
            return httpx.Response(200, stream=stream)
        if failure == "empty":
            return httpx.Response(200, content=b"")
        if failure == "truncated":
            return httpx.Response(200, content=b"short", headers={"Content-Length": "100"})
        return httpx.Response(failure, text=UPSTREAM, headers={"Retry-After": "99999"})

    destination = tmp_path / "video.mp4"
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        if recover:
            download_video(VIDEO_URL, destination, client=client, sleep=sleeps.append)
            assert destination.read_bytes() == b"whole-video"
        else:
            with pytest.raises(McsfxError) as caught:
                download_video(VIDEO_URL, destination, client=client, sleep=sleeps.append)
            assert caught.value.code == "DOWNLOAD_FAILED"
            assert caught.value.retryable
            assert_redacted(caught.value)
            assert not destination.exists()
    assert len(calls) == 3
    assert len(sleeps) == 2
    assert all(0 < delay <= 30 for delay in sleeps)
    assert all(stream.closed for stream in streams)
    assert not list(tmp_path.glob(".mcsfx-*.part"))


@pytest.mark.parametrize(
    "headers,payload",
    [
        ({"Content-Length": "11"}, b"video"),
        ({"Content-Length": "-1"}, b"video"),
        ({"Content-Length": "nonsense"}, b"video"),
        ({"Content-Length": "3"}, b"video"),
        ({"Content-Encoding": "custom"}, b"video"),
        ({}, b"x" * 11),
    ],
)
def test_download_rejects_invalid_sizes_and_encodings(headers, payload, tmp_path, monkeypatch):
    monkeypatch.setattr(providers, "MAX_DOWNLOAD_BYTES", 10)
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    content=payload,
                    headers=headers,
                )
            )
        ) as client,
        pytest.raises(McsfxError) as caught,
    ):
        download_video(
            VIDEO_URL,
            tmp_path / "v.mp4",
            client=client,
            sleep=lambda _: pytest.fail("Invalid download retried"),
        )
    assert caught.value.code == "DOWNLOAD_FAILED"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("status", [204, 206, 400, 401, 403, 404])
def test_download_rejected_responses_are_not_saved_or_retried(status, tmp_path):
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    status,
                    text=UPSTREAM,
                )
            )
        ) as client,
        pytest.raises(McsfxError) as caught,
    ):
        download_video(
            VIDEO_URL,
            tmp_path / "v.mp4",
            client=client,
            sleep=lambda _: pytest.fail("Definitive failure retried"),
        )
    assert not caught.value.retryable
    assert_redacted(caught.value)
    assert not list(tmp_path.iterdir())


def test_download_never_overwrites_an_existing_file(tmp_path):
    destination = tmp_path / "v.mp4"
    destination.write_bytes(b"existing")
    with (
        httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail("Unexpected HTTP"))) as c,
        pytest.raises(McsfxError) as caught,
    ):
        download_video(VIDEO_URL, destination, client=c)
    assert caught.value.code == "OUTPUT_EXISTS"
    assert destination.read_bytes() == b"existing"


def test_download_atomic_install_handles_competing_writer(tmp_path, monkeypatch):
    destination = tmp_path / "v.mp4"
    original_link = providers.os.link

    def race(source, target):
        destination.write_bytes(b"other-writer")
        original_link(source, target)

    monkeypatch.setattr(providers.os, "link", race)
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"ours"))
        ) as c,
        pytest.raises(McsfxError) as caught,
    ):
        download_video(VIDEO_URL, destination, client=c)
    assert caught.value.code == "OUTPUT_EXISTS"
    assert destination.read_bytes() == b"other-writer"
    assert not list(tmp_path.glob(".mcsfx-*.part"))


def test_download_filesystem_failure_is_safe_and_cleans_part(tmp_path, monkeypatch):
    def denied(*_):
        raise PermissionError(UPSTREAM)

    monkeypatch.setattr(providers.os, "link", denied)
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"video"))
        ) as c,
        pytest.raises(McsfxError) as caught,
    ):
        download_video(VIDEO_URL, tmp_path / "v.mp4", client=c)
    assert caught.value.code == "FILESYSTEM_ERROR"
    assert_redacted(caught.value)
    assert not list(tmp_path.iterdir())


def test_default_download_client_is_separate_and_closed(tmp_path, monkeypatch):
    clients = []
    original_client = httpx.Client

    def create_client(**kwargs):
        assert kwargs["trust_env"] is False
        assert kwargs["follow_redirects"] is False
        client = original_client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"video")),
            **kwargs,
        )
        clients.append(client)
        return client

    monkeypatch.setattr(providers.httpx, "Client", create_client)
    provider = Provider(BUILTINS["grok"])
    download_video(VIDEO_URL, tmp_path / "v.mp4")
    assert len(clients) == 2
    assert not clients[0].is_closed
    assert clients[1].is_closed
    provider.close()
    assert clients[0].is_closed


@pytest.mark.parametrize("method", ["submit", "query"])
def test_provider_malformed_redirect_remains_safe(method):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(302, headers={"Location": "data:video/mp4,abc"}, text=UPSTREAM)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        provider = Provider(
            BUILTINS["grok"],
            client=client,
            sleep=lambda _: pytest.fail("Malformed redirect retried"),
        )
        with pytest.raises(McsfxError) as caught:
            if method == "submit":
                provider.submit(VIDEO_URL, "prompt")
            else:
                provider.query("task-1")
    expected = "SUBMISSION_UNKNOWN" if method == "submit" else "PROTOCOL_ERROR"
    assert caught.value.code == expected
    assert_redacted(caught.value)
    assert len(calls) == 1


@pytest.mark.parametrize("retry_after", ["-1", "NaN", "Infinity", "not-a-delay", ""])
def test_bad_retry_after_uses_bounded_backoff(retry_after):
    sleeps = []
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(429, headers={"Retry-After": retry_after})
            )
        ) as client,
        pytest.raises(McsfxError) as caught,
    ):
        Provider(BUILTINS["seedance"], client=client, sleep=sleeps.append).query("task-1")
    assert caught.value.code == "RATE_LIMITED"
    assert sleeps == [0.5, 1.0]


@pytest.mark.parametrize("size", [0, 10, 11])
def test_streamed_download_size_limit_without_content_length(size, tmp_path, monkeypatch):
    monkeypatch.setattr(providers, "MAX_DOWNLOAD_BYTES", 10)
    streams = []

    def handle(_):
        stream = VideoStream([b"v" * size])
        streams.append(stream)
        return httpx.Response(200, stream=stream)

    destination = tmp_path / "v.mp4"
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        if size == 10:
            download_video(VIDEO_URL, destination, client=client, sleep=lambda _: None)
            assert destination.read_bytes() == b"v" * 10
        else:
            with pytest.raises(McsfxError) as caught:
                download_video(VIDEO_URL, destination, client=client, sleep=lambda _: None)
            assert caught.value.code == "DOWNLOAD_FAILED"
            assert not destination.exists()
    assert all(stream.closed for stream in streams)
    assert len(streams) == (3 if size == 0 else 1)
    assert not list(tmp_path.glob(".mcsfx-*.part"))


def test_download_missing_parent_is_safe(tmp_path):
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"video"))
        ) as client,
        pytest.raises(McsfxError) as caught,
    ):
        download_video(VIDEO_URL, tmp_path / "missing-parent" / "v.mp4", client=client)
    assert caught.value.code == "FILESYSTEM_ERROR"
    assert_redacted(caught.value)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("status", [200, 403])
def test_owned_download_client_closes_on_success_and_failure(status, tmp_path, monkeypatch):
    owned = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(status, content=b"video"))
    )
    monkeypatch.setattr(providers, "_new_client", lambda: owned)
    if status == 200:
        download_video(VIDEO_URL, tmp_path / "v.mp4")
    else:
        with pytest.raises(McsfxError):
            download_video(VIDEO_URL, tmp_path / "v.mp4")
    assert owned.is_closed
