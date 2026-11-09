import inspect
import logging
import traceback
from dataclasses import replace
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
import requests
import tos

from mcsfx.config import TosConfig
from mcsfx.errors import McsfxError
from mcsfx.storage import TosStorage, object_key, validate_tos

JOB_ID = "12345678-1234-4abc-8def-123456789abc"
SIGNED_URL = "https://test-bucket.tos-cn-beijing.volces.com/input?X-Tos-Signature=secret"
SECRETS = ("access-key-sensitive", "secret-key-sensitive", "session-token-sensitive", SIGNED_URL)


class FakeClient:
    def __init__(self, *, version_id=None, signed_url=SIGNED_URL, failure=None):
        self.version_id = version_id
        self.signed_url = signed_url
        self.failure = failure
        self.calls = []

    def _call(self, method, kwargs):
        inspect.signature(getattr(tos.TosClientV2, method)).bind(self, **kwargs)
        self.calls.append((method, kwargs))
        if self.failure:
            raise self.failure

    def put_object_from_file(self, **kwargs):
        self._call("put_object_from_file", kwargs)
        return SimpleNamespace(version_id=self.version_id)

    def pre_signed_url(self, **kwargs):
        self._call("pre_signed_url", kwargs)
        return SimpleNamespace(signed_url=self.signed_url)

    def delete_object(self, **kwargs):
        self._call("delete_object", kwargs)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Storage tests must not send HTTP requests")

    monkeypatch.setattr(requests.Session, "request", fail)


@pytest.fixture
def config(monkeypatch):
    result = TosConfig(
        endpoint="https://tos-cn-beijing.volces.com",
        region="cn-beijing",
        bucket="test-bucket",
        prefix="mcsfx/tests",
        access_key_env="TEST_STORAGE_AK",
        secret_key_env="TEST_STORAGE_SK",
        security_token_env="TEST_STORAGE_STS",
    )
    for name, value in zip(
        (result.access_key_env, result.secret_key_env, result.security_token_env), SECRETS
    ):
        monkeypatch.setenv(name, value)
    return result


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "input.mp4"
    path.write_bytes(b"fake video, never uploaded")
    return path


def test_validate_config_and_default_expiry(config):
    assert config.expires_seconds == 345600
    assert validate_tos(config) is None


@pytest.mark.parametrize("expires", [1, 345600, 604800])
def test_valid_expiry_bounds(config, expires):
    validate_tos(replace(config, expires_seconds=expires))


@pytest.mark.parametrize(
    "field,value",
    [
        ("endpoint", ""),
        ("endpoint", None),
        ("endpoint", 123),
        ("endpoint", "http://tos.example.com"),
        ("endpoint", "https://user:password@tos.example.com"),
        ("endpoint", "https://tos.example.com/path"),
        ("endpoint", "https://tos.example.com?secret=value"),
        ("endpoint", "https://tos.example.com#fragment"),
        ("endpoint", "https://tos.example.com?"),
        ("endpoint", "https://tos.example.com#"),
        ("endpoint", "https://tos.example.com:0"),
        ("endpoint", "https://tos.example.com:65536"),
        ("endpoint", "https://tos.example.com:"),
        ("endpoint", "https://tos.example.com:\\secret"),
        ("endpoint", "https://tos.example.com\n"),
        ("endpoint", "https://bad_host.example.com"),
        ("endpoint", "https://" + "a" * 64 + ".com"),
        ("endpoint", "https://[invalid"),
        ("region", ""),
        ("region", None),
        ("region", "cn-beijing\n"),
        ("region", "cn-beijing/secret"),
        ("region", "CN-BEIJING"),
        ("bucket", ""),
        ("bucket", None),
        ("bucket", "ab"),
        ("bucket", "a" * 64),
        ("bucket", "BadBucket"),
        ("bucket", "bad.bucket"),
        ("bucket", "-bucket"),
        ("bucket", "bucket-"),
        ("access_key_env", "literal-secret!"),
        ("secret_key_env", ""),
        ("security_token_env", None),
        ("security_token_env", "TOKEN\n"),
        ("expires_seconds", 0),
        ("expires_seconds", 604801),
        ("expires_seconds", True),
        ("expires_seconds", 345600.0),
        ("expires_seconds", "345600"),
        ("expires_seconds", None),
    ],
)
def test_invalid_public_config(config, field, value):
    with pytest.raises(McsfxError) as error:
        validate_tos(replace(config, **{field: value}), require_credentials=False)
    assert error.value.code == "CONFIG_ERROR"


@pytest.mark.parametrize(
    "prefix",
    [
        "",
        "/",
        "/mcsfx",
        "mcsfx/",
        "mcsfx//jobs",
        ".",
        "..",
        "mcsfx/../other",
        "mcsfx/./jobs",
        "mcsfx\\jobs",
        "mcsfx/%2e%2e",
        "mcsfx?secret",
        "mcsfx#part",
        "mcsfx/*",
        "mcsfx\n",
        "mcsfx/with space",
        "\u97f3\u6548",
        None,
        10,
        "a" * 978,
    ],
)
def test_unsafe_prefix_is_rejected_by_config_and_key(config, prefix):
    bad = replace(config, prefix=prefix)
    for operation in (lambda: validate_tos(bad), lambda: object_key(bad, JOB_ID)):
        with pytest.raises(McsfxError) as error:
            operation()
        assert error.value.code == "CONFIG_ERROR"


@pytest.mark.parametrize("prefix", ["mcsfx", "team/sound_effects.v1", "a" * 977])
def test_key_is_deterministic_and_bounded(config, prefix):
    config = replace(config, prefix=prefix)
    expected = f"{prefix}/{JOB_ID}/input.mp4"
    assert object_key(config, JOB_ID) == expected
    assert object_key(config, JOB_ID) == expected
    assert len(expected.encode("utf-8")) <= 1024


@pytest.mark.parametrize(
    "job_id",
    [
        "",
        None,
        1,
        "../job",
        "not-a-uuid",
        JOB_ID.upper(),
        JOB_ID.replace("-", ""),
        "{" + JOB_ID + "}",
        "urn:uuid:" + JOB_ID,
        JOB_ID + "/other",
        JOB_ID + "\n",
    ],
)
def test_key_requires_canonical_uuid(config, job_id):
    with pytest.raises(McsfxError) as error:
        object_key(config, job_id)
    assert error.value.code == "INVALID_JOB_ID"


def test_key_only_requires_prefix():
    assert object_key(TosConfig(), JOB_ID) == f"mcsfx/{JOB_ID}/input.mp4"


def test_dry_run_and_injected_client_do_not_read_credentials(config, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Credential lookup is forbidden here")

    monkeypatch.setattr("mcsfx.storage._credential", fail)
    validate_tos(config, require_credentials=False)
    TosStorage(config, client=FakeClient())


def test_injected_client_still_requires_valid_public_configuration(config):
    with pytest.raises(McsfxError) as error:
        TosStorage(replace(config, bucket=""), client=FakeClient())
    assert error.value.code == "CONFIG_ERROR"


@pytest.mark.parametrize("field", ["access_key_env", "secret_key_env"])
@pytest.mark.parametrize("value", [None, "", " ", " padded", "has space", "line\nbreak"])
def test_credentials_are_required_from_configured_env(config, monkeypatch, field, value):
    monkeypatch.setenv("TOS_ACCESS_KEY_ID", "must-not-fall-back")
    monkeypatch.setenv("TOS_SECRET_ACCESS_KEY", "must-not-fall-back")
    name = getattr(config, field)
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)
    with pytest.raises(McsfxError) as error:
        validate_tos(config)
    assert error.value.code == "CONFIG_ERROR"


@pytest.mark.parametrize("token", [None, "", "short-lived-sts-token"])
@pytest.mark.parametrize(
    "endpoint",
    [
        "https://tos-cn-beijing.volces.com",
        "tos-cn-beijing.volces.com",
        "https://tos-cn-beijing.volces.com/",
    ],
)
def test_client_uses_exact_env_and_verified_sdk_options(config, monkeypatch, token, endpoint):
    config = replace(config, endpoint=endpoint)
    if token is None:
        monkeypatch.delenv(config.security_token_env)
    else:
        monkeypatch.setenv(config.security_token_env, token)
    signature = inspect.signature(tos.TosClientV2)
    calls = []
    fake = FakeClient()

    def create(**kwargs):
        signature.bind(**kwargs)
        calls.append(kwargs)
        return fake

    monkeypatch.setattr(tos, "TosClientV2", create)
    storage = TosStorage(config)
    assert storage._client is fake
    assert calls == [
        {
            "ak": SECRETS[0],
            "sk": SECRETS[1],
            "security_token": token or None,
            "endpoint": "https://tos-cn-beijing.volces.com",
            "region": "cn-beijing",
            "connection_time": 10,
            "socket_timeout": 60,
            "max_retry_count": 2,
        }
    ]


def test_invalid_optional_token_is_not_silently_dropped(config, monkeypatch):
    monkeypatch.setenv(config.security_token_env, " \t")
    with pytest.raises(McsfxError):
        validate_tos(config)


@pytest.mark.parametrize("version_id", [None, "version+id/with=encoding", "null"])
def test_upload_sign_delete_preserve_exact_object_version(config, video, version_id):
    client = FakeClient(version_id=version_id)
    storage = TosStorage(config, client=client)
    key = object_key(config, JOB_ID)
    assert storage.upload(video, key) == {"version_id": version_id}
    assert storage.signed_url(key, version_id) == SIGNED_URL
    assert storage.delete(key, version_id) is None
    assert client.calls == [
        (
            "put_object_from_file",
            {
                "bucket": config.bucket,
                "key": key,
                "file_path": str(video),
                "content_type": "video/mp4",
            },
        ),
        (
            "pre_signed_url",
            {
                "http_method": tos.HttpMethodType.Http_Method_Get,
                "bucket": config.bucket,
                "key": key,
                "expires": 345600,
                "query": {"versionId": version_id} if version_id is not None else None,
            },
        ),
        ("delete_object", {"bucket": config.bucket, "key": key, "version_id": version_id}),
    ]


def test_sign_uses_configured_expiry_and_default_version(config):
    config = replace(config, expires_seconds=120)
    client = FakeClient()
    storage = TosStorage(config, client=client)
    storage.signed_url(object_key(config, JOB_ID))
    assert client.calls[0][1]["expires"] == 120
    assert client.calls[0][1]["query"] is None


@pytest.mark.parametrize(
    "key",
    [
        "",
        None,
        "mcsfx/tests",
        f"mcsfx/tests/{JOB_ID}",
        f"other/{JOB_ID}/input.mp4",
        f"mcsfx/tests-other/{JOB_ID}/input.mp4",
        f"mcsfx/tests/../{JOB_ID}/input.mp4",
        f"mcsfx/tests/{JOB_ID}/other.mp4",
        f"mcsfx/tests/{JOB_ID}/input.mp4/child",
        f"mcsfx/tests/{JOB_ID}/input.mp4?secret",
        f"mcsfx/tests/{JOB_ID.upper()}/input.mp4",
        "mcsfx/tests/*/input.mp4",
    ],
)
def test_all_operations_reject_keys_outside_exact_job_scope(config, video, key):
    client = FakeClient()
    storage = TosStorage(config, client=client)
    for operation in (
        lambda: storage.upload(video, key),
        lambda: storage.signed_url(key),
        lambda: storage.delete(key),
    ):
        with pytest.raises(McsfxError) as error:
            operation()
        assert error.value.code == "INVALID_OBJECT_KEY"
    assert client.calls == []


@pytest.mark.parametrize("version_id", ["", " ", "\nsecret", 0, False, [], {}])
def test_invalid_versions_never_fall_back_to_unversioned_calls(config, version_id):
    client = FakeClient()
    storage = TosStorage(config, client=client)
    key = object_key(config, JOB_ID)
    for operation in (storage.signed_url, storage.delete):
        with pytest.raises(McsfxError) as error:
            operation(key, version_id)
        assert error.value.code == "INVALID_VERSION_ID"
    assert client.calls == []


def test_missing_file_or_directory_never_uploads(config, tmp_path):
    client = FakeClient()
    storage = TosStorage(config, client=client)
    for path in (tmp_path, tmp_path / "missing.mp4"):
        with pytest.raises(McsfxError) as error:
            storage.upload(path, object_key(config, JOB_ID))
        assert error.value.code == "INPUT_ERROR"
    assert client.calls == []


@pytest.mark.parametrize(
    "signed",
    [
        "http://example.com/input?secret=value",
        "//example.com/input",
        "",
        None,
        12,
        "https://user:password@example.com/input",
        "https://example.com:65536/input",
        "https://example.com/input#fragment",
        "https://example.com/input\n",
        "https://example.com\\other/input",
    ],
)
def test_bad_signed_urls_are_not_returned_or_exposed(config, signed):
    storage = TosStorage(config, client=FakeClient(signed_url=signed))
    with pytest.raises(McsfxError) as error:
        storage.signed_url(object_key(config, JOB_ID))
    assert error.value.code == "TOS_SIGN_FAILED"
    assert "example.com" not in str(error.value.as_dict())


def test_invalid_upload_response_remains_unknown_and_never_deletes(config, video):
    client = FakeClient(version_id=123)
    storage = TosStorage(config, client=client)
    with pytest.raises(McsfxError) as error:
        storage.upload(video, object_key(config, JOB_ID))
    assert error.value.code == "TOS_UPLOAD_FAILED"
    assert error.value.details == {"upload_outcome": "unknown"}
    assert [name for name, kwargs in client.calls] == ["put_object_from_file"]


def test_uncertain_upload_is_not_retried_or_deleted_by_adapter(config, video):
    client = FakeClient(failure=TimeoutError("connection lost after upload"))
    storage = TosStorage(config, client=client)
    with pytest.raises(McsfxError) as error:
        storage.upload(video, object_key(config, JOB_ID))
    assert error.value.code == "TOS_UPLOAD_FAILED"
    assert error.value.details == {"upload_outcome": "unknown"}
    assert not error.value.retryable
    assert [name for name, kwargs in client.calls] == ["put_object_from_file"]


@pytest.mark.parametrize(
    "method,code",
    [
        ("upload", "TOS_UPLOAD_FAILED"),
        ("signed_url", "TOS_SIGN_FAILED"),
        ("delete", "TOS_DELETE_FAILED"),
        ("constructor", "TOS_CLIENT_ERROR"),
    ],
)
def test_sdk_failures_and_diagnostics_are_redacted(
    config, video, monkeypatch, caplog, method, code
):
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(logging, "logThreads", False)
    loggers = [
        tos.log.get_logger(),
        logging.getLogger("urllib3.connectionpool"),
        logging.getLogger("urllib3.util.retry"),
    ]
    original_filters = [logger.filters[:] for logger in loggers]
    secret_message = " ".join(SECRETS)
    client = FakeClient()

    def fail(*args, **kwargs):
        for logger in loggers:
            logger.error("Upstream payload: %s", secret_message)
        raise RuntimeError(secret_message)

    if method == "constructor":
        monkeypatch.setattr(tos, "TosClientV2", fail)
        operation = lambda: TosStorage(config)
    else:
        monkeypatch.setattr(client, "_call", fail)
        storage = TosStorage(config, client=client)
        key = object_key(config, JOB_ID)
        operation = (
            (lambda: storage.upload(video, key))
            if method == "upload"
            else (lambda: getattr(storage, method)(key))
        )
    with pytest.raises(McsfxError) as error:
        operation()
    assert error.value.code == code
    assert not error.value.retryable
    displayed = repr(error.value.as_dict()) + "".join(traceback.format_exception(error.value))
    for secret in SECRETS:
        assert secret not in displayed
        assert secret not in caplog.text
    assert [logger.filters for logger in loggers] == original_filters
    loggers[0].warning("normal logging restored")
    assert "normal logging restored" in caplog.text
    assert client.calls == []


def test_config_errors_do_not_echo_supplied_values(config):
    for field in ("endpoint", "region", "bucket", "prefix", "access_key_env"):
        with pytest.raises(McsfxError) as error:
            validate_tos(replace(config, **{field: SIGNED_URL}))
        assert SIGNED_URL not in str(error.value.as_dict())


def test_actual_sdk_presigns_offline_without_leaking_credentials(config, caplog):
    caplog.set_level(logging.DEBUG)
    storage = TosStorage(config)
    key = object_key(config, JOB_ID)
    try:
        signed = storage.signed_url(key, "version+id/with=encoding")
        url = urlsplit(signed)
        query = parse_qs(url.query)
        assert url.scheme == "https"
        assert url.hostname == "test-bucket.tos-cn-beijing.volces.com"
        assert url.path == "/" + key
        assert query["X-Tos-Expires"] == ["345600"]
        assert query["versionId"] == ["version+id/with=encoding"]
        assert query["X-Tos-Security-Token"] == [SECRETS[2]]
        assert "X-Tos-Signature" in query
        for secret in (*SECRETS, signed, query["X-Tos-Signature"][0]):
            assert secret not in caplog.text
    finally:
        storage._client.close()
