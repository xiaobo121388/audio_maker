"""Provider protocols and bounded downloads with explicit, exact-path authentication."""

import math
import os
import re
import tempfile
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx

from mcsfx.config import Profile, validate_profile
from mcsfx.errors import McsfxError

MAX_ATTEMPTS = 3
MAX_REDIRECTS = 5
MAX_DOWNLOAD_BYTES = 1024**3
_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
_REMOTE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")
_STATES = {
    state: state
    for state in (
        "queued",
        "running",
        "succeeded",
        "failed",
        "cancelled",
        "expired",
    )
}
_TERMINAL_ERRORS = {
    "failed": "Remote video generation failed.",
    "cancelled": "Remote video generation was cancelled.",
    "expired": "Remote video generation expired.",
}


@dataclass(frozen=True)
class RemoteResult:
    status: str
    video_url: str | None = None
    error: str | None = None


def _new_client() -> httpx.Client:
    # Ignore ambient proxy/CA overrides: inherited routes may be broken or untrusted.
    # Explicit proxy/trust configuration belongs on an injected client.
    return httpx.Client(timeout=_TIMEOUT, follow_redirects=False, trust_env=False)


def _send(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    headers: dict,
    body: dict | None = None,
    stream: bool = False,
) -> httpx.Response:
    # Raw requests bypass client header/cookie/parameter merging, including on redirects.
    request = httpx.Request(
        method, url, headers=headers, json=body, extensions={"timeout": _TIMEOUT.as_dict()}
    )
    return client.send(request, auth=None, follow_redirects=False, stream=stream)


def _retry_delay(attempt: int, response: httpx.Response | None = None) -> float:
    delay = 0.5 * (2**attempt)
    if response is not None:
        try:
            seconds = float(response.headers.get("Retry-After", ""))
            if math.isfinite(seconds) and seconds >= 0:
                delay = max(delay, seconds)
        except ValueError:
            pass
    return min(delay, 30.0)


def _transient(status: int) -> bool:
    return status == 429 or 500 <= status <= 599


def _https_url(value: object) -> bool:
    if not isinstance(value, str) or not value or any(ord(c) <= 32 or ord(c) == 127 for c in value):
        return False
    try:
        parts = urlsplit(value)
        return bool(
            parts.scheme == "https"
            and parts.hostname
            and not parts.username
            and not parts.password
            and "@" not in parts.netloc
            and not parts.fragment
            and "\\" not in value
            and parts.port != 0
            and httpx.URL(value).host
        )
    except (ValueError, httpx.InvalidURL):
        return False


def _api_headers(profile: Profile) -> dict[str, str]:
    key = os.environ.get(profile.api_key_env, "")
    if not key or any(not 33 <= ord(c) <= 126 for c in key):
        raise McsfxError("AUTH_ERROR", "The configured API key is missing or invalid.")
    return {"Authorization": f"Bearer {key}", "Accept": "application/json"}


def _content_url(profile: Profile, remote_id: str, value: object) -> str:
    """Resolve only the saved job's literal content endpoint, never a supplied URL join."""
    validate_profile(profile)
    if profile.protocol != "xai" or profile.download_mode != "same_origin_content":
        raise McsfxError("CONFIG_ERROR", "Authenticated content downloads require explicit opt-in.")
    if not isinstance(remote_id, str) or not _REMOTE_ID.fullmatch(remote_id):
        raise McsfxError("INPUT_ERROR", "The remote task identifier is invalid.")
    expected = profile.base_url.rstrip("/") + "/videos/" + remote_id + "/content"
    if not _https_url(expected) or "%" in expected or str(httpx.URL(expected)) != expected:
        raise McsfxError("CONFIG_ERROR", "Content downloads require a canonical HTTPS base_url.")
    if value not in (expected, urlsplit(expected).path):
        raise McsfxError("PROTOCOL_ERROR", "Video URL does not match this job's content endpoint.")
    return expected


def _json_object(response: httpx.Response, code: str) -> dict:
    try:
        data = response.json()
    except (ValueError, UnicodeError, RecursionError):
        raise McsfxError(code, "Provider returned an invalid response.") from None
    if not isinstance(data, dict):
        raise McsfxError(code, "Provider returned an invalid response.")
    return data


def _receipt(response: httpx.Response, secret: str) -> dict:
    details = {"http_status": response.status_code}
    for header in ("x-request-id", "x-correlation-id", "request-id"):
        value = response.headers.get(header, "")
        if value and secret not in value and re.fullmatch(r"[A-Za-z0-9._-]{1,128}", value):
            details["provider_request_id"] = value
            break
    return details


def _submission_unknown(details: dict | None = None) -> McsfxError:
    return McsfxError(
        "SUBMISSION_UNKNOWN",
        "Submission outcome is unknown; do not automatically submit again. "
        "Check the provider account before starting another paid request.",
        details=details,
    )


def _http_error(status: int, *, query: bool) -> McsfxError:
    if status in (401, 403):
        return McsfxError(
            "AUTH_ERROR",
            "Provider authentication or permission was rejected.",
            details={"http_status": status},
        )
    if status == 429:
        return McsfxError(
            "RATE_LIMITED",
            "Provider rate limit reached.",
            retryable=query,
            details={"http_status": status},
        )
    return McsfxError(
        "PROVIDER_ERROR",
        "Provider rejected or could not serve the request.",
        retryable=query and _transient(status),
        details={"http_status": status},
    )


class Provider:
    """One profile; caller-supplied clients remain owned by the caller.

    Submission is never retried here. Injected transports must likewise have retries
    disabled. Only query/download failures are eligible for bounded retries.
    """

    def __init__(
        self,
        profile: Profile,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        validate_profile(profile)
        self.profile = profile
        self._client = client if client is not None else _new_client()
        self._owns_client = client is None
        self._sleep = sleep

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _headers(self) -> dict[str, str]:
        return _api_headers(self.profile)

    def _endpoint(self) -> str:
        route = "/videos" if self.profile.protocol == "xai" else "/contents/generations/tasks"
        return self.profile.base_url.rstrip("/") + route

    def submit(self, video_reference: str, prompt: str, *, duration: int | None = None) -> str:
        if (
            not isinstance(video_reference, str)
            or not video_reference.strip()
            or not isinstance(prompt, str)
        ):
            raise McsfxError("INPUT_ERROR", "A video reference and text prompt are required.")
        keyframe = self.profile.preset == "grok-keyframe"
        if keyframe:
            prefix = "data:image/png;base64,"
            if (
                type(duration) is not int
                or not 1 <= duration <= 15
                or not video_reference.startswith(prefix)
                or len(video_reference) == len(prefix)
            ):
                raise McsfxError(
                    "INPUT_ERROR",
                    "grok-keyframe requires a PNG data URI and 1..15 integer seconds.",
                )
        elif duration is not None:
            raise McsfxError(
                "INPUT_ERROR", "Explicit duration is only supported for grok-keyframe."
            )
        if self.profile.protocol == "xai":
            url = self._endpoint() + ("/generations" if keyframe else "/edits")
            body = {"model": self.profile.model, "prompt": prompt}
            if keyframe:
                body.update(
                    image={"url": video_reference},
                    duration=duration,
                    resolution="480p",
                    generate_audio=True,
                )
            else:
                body["video"] = {"url": video_reference}
            id_field = "request_id"
        else:
            url = self._endpoint()
            body = {
                "model": self.profile.model,
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "video_url",
                        "video_url": {"url": video_reference},
                        "role": "reference_video",
                    },
                ],
                "generate_audio": True,
                "omni_reference_task_type": "edit",
                "ratio": "adaptive",
                "duration": -1,
                "resolution": "720p",
            }
            id_field = "id"
        headers = self._headers()
        try:
            response = _send(self._client, "POST", url, headers=headers, body=body)
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            raise _submission_unknown({"transport_error": type(exc).__name__}) from None
        with closing(response):
            receipt = _receipt(response, headers["Authorization"][7:])
            status = response.status_code
            if status == 408 or status >= 500 or not 200 <= status < 500 or 300 <= status < 400:
                raise _submission_unknown(receipt)
            if status >= 400:
                raise _http_error(status, query=False)
            try:
                data = _json_object(response, "SUBMISSION_UNKNOWN")
                remote_id = data.get(id_field)
                valid = (
                    isinstance(remote_id, str)
                    and _REMOTE_ID.fullmatch(remote_id)
                    and headers["Authorization"][7:] not in remote_id
                    and data.get("error") is None
                )
            except McsfxError:
                raise _submission_unknown({**receipt, "reason": "invalid_response"}) from None
            if not valid:
                raise _submission_unknown({**receipt, "reason": "invalid_task_id_or_error"})
            return remote_id

    def query(self, remote_id: str) -> RemoteResult:
        if not isinstance(remote_id, str) or not _REMOTE_ID.fullmatch(remote_id):
            raise McsfxError("INPUT_ERROR", "The remote task identifier is invalid.")
        headers = self._headers()
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = _send(
                    self._client, "GET", self._endpoint() + "/" + remote_id, headers=headers
                )
            except httpx.InvalidURL:
                raise McsfxError(
                    "PROTOCOL_ERROR", "Provider returned an invalid redirect."
                ) from None
            except httpx.HTTPError:
                if attempt == MAX_ATTEMPTS - 1:
                    raise McsfxError(
                        "NETWORK_ERROR", "Provider query could not be completed.", retryable=True
                    ) from None
                self._sleep(_retry_delay(attempt))
                continue
            with closing(response):
                if _transient(response.status_code) and attempt < MAX_ATTEMPTS - 1:
                    delay = _retry_delay(attempt, response)
                else:
                    if not 200 <= response.status_code < 300:
                        raise _http_error(response.status_code, query=True)
                    # The xAI OpenAPI explicitly allows an empty 202 for a pending job.
                    if (
                        self.profile.protocol == "xai"
                        and response.status_code == 202
                        and not response.content.strip()
                    ):
                        return RemoteResult("running")
                    return self._result(
                        _json_object(response, "PROTOCOL_ERROR"), response.status_code, remote_id
                    )
            self._sleep(delay)
        raise AssertionError("Unreachable retry state")

    def _result(self, data: dict, http_status: int, remote_id: str) -> RemoteResult:
        states = dict(_STATES)
        if self.profile.protocol == "xai":
            states.update(pending="running", done="succeeded")
            output = data.get("video")
            url = output.get("url") if isinstance(output, dict) else None
            # Compatibility: completed xAI responses can contain video without status.
            status = data.get("status")
            if "status" not in data and http_status == 200:
                if self.profile.download_mode == "same_origin_content":
                    _content_url(self.profile, remote_id, url)
                    status = "done"
                elif _https_url(url):
                    status = "done"
        else:
            output = data.get("content")
            url = output.get("video_url") if isinstance(output, dict) else None
            status = data.get("status")
        if not isinstance(status, str) or status not in states:
            raise McsfxError("PROTOCOL_ERROR", "Provider returned an unrecognized task state.")
        normalized = states[status]
        if normalized in _TERMINAL_ERRORS:
            return RemoteResult(normalized, error=_TERMINAL_ERRORS[normalized])
        if data.get("error") is not None:
            raise McsfxError("PROTOCOL_ERROR", "Provider returned conflicting task results.")
        if normalized != "succeeded":
            return RemoteResult(normalized)
        if http_status != 200:
            raise McsfxError("PROTOCOL_ERROR", "Provider returned conflicting task results.")
        if self.profile.protocol == "xai" and isinstance(output, dict):
            moderation = output.get("respect_moderation", True)
            if moderation is False:
                return RemoteResult("failed", error="Remote video was rejected by moderation.")
            if moderation is not True:
                raise McsfxError("PROTOCOL_ERROR", "Provider returned an invalid video result.")
        if self.profile.download_mode == "same_origin_content":
            url = _content_url(self.profile, remote_id, url)
        elif not _https_url(url):
            raise McsfxError("PROTOCOL_ERROR", "Provider did not return a valid video URL.")
        return RemoteResult("succeeded", video_url=url)


def _download_response(
    client: httpx.Client, url: str, initial_headers: dict | None = None
) -> httpx.Response:
    for redirects in range(MAX_REDIRECTS + 1):
        if not _https_url(url):
            raise McsfxError("DOWNLOAD_FAILED", "Video downloads require a valid HTTPS URL.")
        try:
            # Even a redirect back to the original endpoint must be credential-free.
            headers = dict(initial_headers or {}) if redirects == 0 else {}
            headers["Accept-Encoding"] = "identity"
            response = _send(client, "GET", url, headers=headers, stream=True)
        except httpx.InvalidURL:
            raise McsfxError("DOWNLOAD_FAILED", "Video download redirect was invalid.") from None
        if response.status_code not in (301, 302, 303, 307, 308):
            return response
        with closing(response):
            location = response.headers.get("Location")
            if (
                not location
                or redirects == MAX_REDIRECTS
                or any(ord(c) <= 32 or ord(c) == 127 for c in location)
                or "\\" in location
            ):
                raise McsfxError(
                    "DOWNLOAD_FAILED", "Video download redirect was invalid or excessive."
                )
            try:
                url = urljoin(url, location)
            except ValueError:
                raise McsfxError(
                    "DOWNLOAD_FAILED", "Video download redirect was invalid."
                ) from None
    raise AssertionError("Unreachable redirect state")


def _write_download(response: httpx.Response, destination: Path) -> None:
    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
        raise McsfxError("DOWNLOAD_FAILED", "Unexpected video download encoding.")
    length = response.headers.get("Content-Length")
    if length is not None and (
        not re.fullmatch(r"[0-9]{1,20}", length) or int(length) > MAX_DOWNLOAD_BYTES
    ):
        raise McsfxError("DOWNLOAD_FAILED", "Invalid or excessive video download size.")
    expected = int(length) if length is not None else None
    part = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=".mcsfx-", suffix=".part", delete=False
        ) as output:
            part = Path(output.name)
            total = 0
            for chunk in response.iter_bytes(chunk_size=64 * 1024):
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES or (expected is not None and total > expected):
                    raise McsfxError("DOWNLOAD_FAILED", "Video download exceeded its size limit.")
                output.write(chunk)
            if total == 0 or (expected is not None and total != expected):
                raise McsfxError(
                    "DOWNLOAD_FAILED", "Video download was empty or incomplete.", retryable=True
                )
            output.flush()
            os.fsync(output.fileno())
        # Creating a hard link publishes the complete file atomically without replacement,
        # including if a competing writer creates the destination after our initial check.
        os.link(part, destination)
    finally:
        if part is not None:
            part.unlink(missing_ok=True)


def download_video(
    url: str,
    destination: Path,
    *,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
    profile: Profile | None = None,
    remote_id: str | None = None,
) -> None:
    """Stream at most 1 GiB, then atomically install into an existing parent directory.

    A supplied client is borrowed, never closed, and its default authentication,
    headers, cookies, and query parameters are not used. Each retry starts afresh.
    Filesystems without hard-link support fail safely instead of overwriting.
    Passing a profile requires same_origin_content opt-in and its saved remote ID;
    only the exact initial content endpoint receives API authorization.
    """
    initial_headers = None
    if profile is not None:
        url = _content_url(profile, remote_id, url)
        initial_headers = {"Authorization": _api_headers(profile)["Authorization"]}
    elif remote_id is not None:
        raise McsfxError("CONFIG_ERROR", "Content downloads require an opted-in profile.")
    if not _https_url(url):
        raise McsfxError("DOWNLOAD_FAILED", "Video downloads require a valid HTTPS URL.")
    owned = client is None
    download_client = client if client is not None else _new_client()
    try:
        for attempt in range(MAX_ATTEMPTS):
            if destination.exists() or destination.is_symlink():
                raise McsfxError("OUTPUT_EXISTS", "The video destination already exists.")
            response = None
            try:
                response = _download_response(download_client, url, initial_headers)
                with closing(response):
                    if response.status_code != 200:
                        raise McsfxError(
                            "DOWNLOAD_FAILED",
                            "Video download was rejected.",
                            retryable=_transient(response.status_code),
                        )
                    _write_download(response, destination)
                return
            except httpx.HTTPError:
                failure = McsfxError(
                    "DOWNLOAD_FAILED", "Video download could not be completed.", retryable=True
                )
            except McsfxError as exc:
                failure = exc
            if not failure.retryable or attempt == MAX_ATTEMPTS - 1:
                raise failure from None
            sleep(_retry_delay(attempt, response))
    except FileExistsError:
        raise McsfxError("OUTPUT_EXISTS", "The video destination already exists.") from None
    except OSError:
        raise McsfxError("FILESYSTEM_ERROR", "Could not install the downloaded video.") from None
    finally:
        if owned:
            download_client.close()
