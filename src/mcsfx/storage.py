"""Narrow TOS access for job-owned input videos.

Persist object_key() BEFORE upload, then persist the returned version_id and pass
it to signed_url() and delete(). An upload exception does not prove that the
object is absent. This adapter never performs automatic cleanup: the job layer
must retain uncertain uploads until non-submission or a terminal remote state
is established.

The requested URL TTL defaults to 345600 seconds, not a guaranteed lifetime.
STS credentials can expire earlier and invalidate the URL; callers must ensure
their credentials remain valid for the provider's entire input-fetch window.

SDK contract verified against tos 2.9.3 and the official presigning examples:
https://docs.volcengine.com/docs/torch-object-storage/generalpresigned-python-sdk?lang=en
"""

import logging
import os
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

import tos

from mcsfx.config import TosConfig
from mcsfx.errors import McsfxError

_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DNS_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?")
_UUID_TEXT = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
_MAX_PREFIX = 1024 - len("/00000000-0000-0000-0000-000000000000/input.mp4")


def _endpoint(value: str) -> str:
    """Accept an HTTPS origin or bare DNS endpoint, never a URL with payloads."""
    valid = False
    if (
        isinstance(value, str)
        and value.isascii()
        and not any(c.isspace() or ord(c) < 32 or c in "\\@?#" for c in value)
    ):
        try:
            url = urlsplit(value if "://" in value else "https://" + value)
            host = url.hostname or ""
            valid = (
                url.scheme == "https"
                and bool(host)
                and len(host) <= 253
                and all(_DNS_LABEL.fullmatch(label) for label in host.split("."))
                and url.path in ("", "/")
                and not url.username
                and not url.password
                and (url.port is None or 1 <= url.port <= 65535)
                and not url.netloc.endswith(":")
            )
        except ValueError:
            pass
    if not valid:
        raise McsfxError("CONFIG_ERROR", "TOS endpoint must be an HTTPS origin or bare hostname.")
    return "https://" + url.netloc


def _validate_prefix(prefix: str) -> None:
    if (
        not isinstance(prefix, str)
        or not 1 <= len(prefix) <= _MAX_PREFIX
        or not re.fullmatch(r"[A-Za-z0-9._/-]+", prefix)
        or any(part in ("", ".", "..") for part in prefix.split("/"))
    ):
        raise McsfxError(
            "CONFIG_ERROR",
            "TOS prefix must be a nonempty relative ASCII path without dot segments.",
        )


def _credential(env_name: str, *, optional: bool = False) -> str | None:
    value = os.environ.get(env_name)
    if optional and value in (None, ""):
        return None
    if not value or not all(33 <= ord(c) <= 126 for c in value):
        raise McsfxError(
            "CONFIG_ERROR", "TOS credential environment variables are missing or invalid."
        )
    return value


def validate_tos(config: TosConfig, *, require_credentials: bool = True) -> None:
    """Validate public fields and, unless disabled for dry runs, environment credentials."""
    if not isinstance(config, TosConfig):
        raise McsfxError("CONFIG_ERROR", "Expected a TOS configuration.")
    _endpoint(config.endpoint)
    if not isinstance(config.region, str) or not re.fullmatch(
        r"[a-z0-9]+(?:-[a-z0-9]+)*", config.region
    ):
        raise McsfxError("CONFIG_ERROR", "TOS region must be a nonempty region identifier.")
    if not isinstance(config.bucket, str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]", config.bucket
    ):
        raise McsfxError("CONFIG_ERROR", "TOS bucket must be a valid 3-63 character bucket name.")
    _validate_prefix(config.prefix)
    for field in ("access_key_env", "secret_key_env", "security_token_env"):
        name = getattr(config, field)
        if not isinstance(name, str) or not _ENV_NAME.fullmatch(name):
            raise McsfxError(
                "CONFIG_ERROR", "TOS credential fields must name environment variables."
            )
    if type(config.expires_seconds) is not int or not 1 <= config.expires_seconds <= 604800:
        raise McsfxError(
            "CONFIG_ERROR", "TOS URL expiry must be an integer from 1 to 604800 seconds."
        )
    if require_credentials:
        _credential(config.access_key_env)
        _credential(config.secret_key_env)
        _credential(config.security_token_env, optional=True)


def object_key(config: TosConfig, job_id: str) -> str:
    """Build the deterministic key without needing credentials or making SDK calls."""
    _validate_prefix(config.prefix)
    if not isinstance(job_id, str) or not _UUID_TEXT.fullmatch(job_id):
        raise McsfxError("INVALID_JOB_ID", "Expected a canonical UUID job ID.")
    return f"{config.prefix}/{UUID(job_id)}/input.mp4"


def _validate_key(config: TosConfig, key: str) -> None:
    if isinstance(key, str) and key.startswith(config.prefix + "/"):
        suffix = key[len(config.prefix) + 1 :]
        job_id, separator, filename = suffix.partition("/")
        if separator and filename == "input.mp4" and _UUID_TEXT.fullmatch(job_id):
            return
    raise McsfxError("INVALID_OBJECT_KEY", "TOS key must identify a job input within its prefix.")


def _validate_version(version_id: str | None) -> None:
    # The literal string "null" is an actual TOS version identifier, not None.
    if version_id is not None and (
        not isinstance(version_id, str)
        or not version_id
        or not all(33 <= ord(c) <= 126 for c in version_id)
    ):
        raise McsfxError("INVALID_VERSION_ID", "TOS version ID must be a nonempty opaque string.")


class _QuietSdkThread(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self.thread_id = threading.get_ident()

    def filter(self, record: logging.LogRecord) -> bool:
        return threading.get_ident() != self.thread_id


@contextmanager
def _quiet_sdk():
    # SDK debug output includes canonical queries, STS tokens and signatures.
    # Suppress only this call's thread; leave application logging configuration intact.
    loggers = (
        tos.log.get_logger(),
        logging.getLogger("urllib3.connectionpool"),
        logging.getLogger("urllib3.util.retry"),
    )
    guard = _QuietSdkThread()
    for logger in loggers:
        logger.addFilter(guard)
    try:
        yield
    finally:
        for logger in loggers:
            logger.removeFilter(guard)


class TosStorage:
    """A single-object adapter. Injected clients bypass only credential lookup."""

    def __init__(self, config: TosConfig, *, client=None):
        validate_tos(config, require_credentials=client is None)
        self.config = config
        if client is not None:
            self._client = client
            return
        ak = _credential(config.access_key_env)
        sk = _credential(config.secret_key_env)
        token = _credential(config.security_token_env, optional=True)
        try:
            with _quiet_sdk():
                self._client = tos.TosClientV2(
                    ak=ak,
                    sk=sk,
                    security_token=token,
                    endpoint=_endpoint(config.endpoint),
                    region=config.region,
                    connection_time=10,
                    socket_timeout=60,
                    max_retry_count=2,
                )
        except Exception:  # noqa: BLE001 - SDK/transport errors can contain credentials.
            raise McsfxError("TOS_CLIENT_ERROR", "Cannot initialize TOS storage.") from None

    def upload(self, path: Path, key: str) -> dict:
        """Upload one MP4; failure can leave an object and must not trigger cleanup here."""
        _validate_key(self.config, key)
        try:
            valid_file = isinstance(path, Path) and path.is_file()
        except OSError:
            valid_file = False
        if not valid_file:
            raise McsfxError("INPUT_ERROR", "TOS upload requires an existing regular file.")
        try:
            with _quiet_sdk():
                output = self._client.put_object_from_file(
                    bucket=self.config.bucket,
                    key=key,
                    file_path=str(path),
                    content_type="video/mp4",
                )
                version_id = output.version_id
                _validate_version(version_id)
        except Exception:  # noqa: BLE001 - SDK/transport errors can contain credentials.
            raise McsfxError(
                "TOS_UPLOAD_FAILED",
                "TOS upload failed; retain the recorded key for recovery.",
                details={"upload_outcome": "unknown"},
            ) from None
        return {"version_id": version_id}

    def signed_url(self, key: str, version_id: str | None = None) -> str:
        """Presign HTTPS GET. STS expiration may shorten the configured URL lifetime."""
        _validate_key(self.config, key)
        _validate_version(version_id)
        try:
            with _quiet_sdk():
                output = self._client.pre_signed_url(
                    http_method=tos.HttpMethodType.Http_Method_Get,
                    bucket=self.config.bucket,
                    key=key,
                    expires=self.config.expires_seconds,
                    query={"versionId": version_id} if version_id is not None else None,
                )
                signed = output.signed_url
                url = urlsplit(signed) if isinstance(signed, str) else None
                if (
                    url is None
                    or url.scheme != "https"
                    or not url.hostname
                    or url.username is not None
                    or url.password is not None
                    or url.fragment
                    or not signed.isascii()
                    or any(ord(c) <= 32 or ord(c) == 127 for c in signed)
                    or "\\" in signed
                    or (url.port is not None and not 1 <= url.port <= 65535)
                ):
                    raise ValueError("Invalid presigned URL")
        except Exception:  # noqa: BLE001 - SDK/transport errors can contain signed URLs.
            raise McsfxError(
                "TOS_SIGN_FAILED", "Cannot generate an HTTPS TOS download URL."
            ) from None
        return signed

    def delete(self, key: str, version_id: str | None = None) -> None:
        """Delete exactly the caller-approved object/version, never list or recurse.

        Without a version ID, TOS may create a delete marker in a versioned bucket;
        that is not evidence that older versions have been physically removed.
        """
        _validate_key(self.config, key)
        _validate_version(version_id)
        try:
            with _quiet_sdk():
                self._client.delete_object(
                    bucket=self.config.bucket,
                    key=key,
                    version_id=version_id,
                )
        except Exception:  # noqa: BLE001 - SDK/transport errors can contain credentials.
            raise McsfxError(
                "TOS_DELETE_FAILED", "Cannot delete the recorded TOS object."
            ) from None
