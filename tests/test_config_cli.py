import json
from dataclasses import replace

import pytest

from mcsfx.cli import main
from mcsfx.config import BUILTINS, load_settings, validate_profile
from mcsfx.errors import McsfxError


def test_defaults_without_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MCSFX_CONFIG", raising=False)
    settings = load_settings()
    assert settings.profiles["grok"].transport == "data_uri"
    assert settings.profiles["seedance"].transport == "tos"


def test_custom_relay_profile_and_bom(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        """[profiles.relay]
protocol = "xai"
preset = "grok-edit"
base_url = "https://relay.example/v1"
model = "custom-alias"
api_key_env = "RELAY_KEY"
transport = "tos"
""",
        encoding="utf-8-sig",
    )
    profile = load_settings(str(config)).profile("relay")
    assert profile.model == "custom-alias"
    assert profile.base_url == "https://relay.example/v1"


@pytest.mark.parametrize(
    "fields",
    [
        {"base_url": "http://relay.example/v1"},
        {"base_url": "https://key:secret@relay.example/v1"},
        {"base_url": "https://relay.example/v1?key=secret"},
        {"preset": "seedance-2.5-edit"},
        {"api_key_env": "literal key with spaces"},
        {"transport": "magic_upload"},
    ],
)
def test_profile_validation(fields):
    with pytest.raises(McsfxError):
        validate_profile(replace(BUILTINS["grok"], **fields))


def test_literal_key_is_not_accepted_in_config(tmp_path):
    config = tmp_path / "bad.toml"
    config.write_text('[profiles.grok]\napi_key="do-not-leak"', encoding="utf-8")
    with pytest.raises(McsfxError) as exc:
        load_settings(str(config))
    assert "do-not-leak" not in str(exc.value)


@pytest.mark.parametrize(
    "args",
    [
        ["unknown", "--json"],
        ["generate", "--json"],
        ["fetch", "abc", "--timeout", "nan", "--json"],
        ["fetch", "abc", "--timeout", "-1", "--json"],
    ],
)
def test_argument_failures_are_single_json(args, capsys):
    assert main(args) == 2
    result = json.loads(capsys.readouterr().out)
    assert not result["ok"]
    assert result["error"]["code"] == "ARGUMENT_ERROR"


def test_missing_config_redacted_json(tmp_path, capsys):
    code = main(["doctor", "--config", str(tmp_path / "missing.toml"), "--json"])
    assert code != 0
    result = json.loads(capsys.readouterr().out)
    assert result["error"]["code"] == "CONFIG_ERROR"


def test_missing_job_is_json(tmp_path, capsys):
    code = main(
        ["status", "12345678-1234-1234-1234-123456789012", "--state-dir", str(tmp_path), "--json"]
    )
    assert code != 0
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "JOB_NOT_FOUND"
