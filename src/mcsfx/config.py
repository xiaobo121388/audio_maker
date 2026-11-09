"""Strict, non-secret configuration and built-in capability profiles."""

import os
import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from mcsfx.errors import McsfxError


@dataclass(frozen=True)
class Profile:
    name: str
    protocol: str
    preset: str
    base_url: str
    model: str
    api_key_env: str
    transport: str
    download_mode: str = "public"

    def snapshot(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TosConfig:
    endpoint: str = ""
    region: str = ""
    bucket: str = ""
    prefix: str = "mcsfx"
    access_key_env: str = "TOS_ACCESS_KEY_ID"
    secret_key_env: str = "TOS_SECRET_ACCESS_KEY"
    security_token_env: str = "TOS_SECURITY_TOKEN"
    expires_seconds: int = 345600

    def snapshot(self) -> dict:
        return asdict(self)


def default_state_dir() -> Path:
    root = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "state"))
    return root / "mcsfx" / "jobs"


BUILTINS = {
    "grok": Profile(
        "grok",
        "xai",
        "grok-edit",
        "https://api.x.ai/v1",
        "grok-imagine-video",
        "XAI_API_KEY",
        "data_uri",
    ),
    "seedance": Profile(
        "seedance",
        "ark",
        "seedance-2.5-edit",
        "https://ark.cn-beijing.volces.com/api/v3",
        "doubao-seedance-2-5-260628",
        "ARK_API_KEY",
        "tos",
    ),
    "grok-sfx": Profile(
        "grok-sfx",
        "xai",
        "grok-keyframe",
        "https://api.x.ai/v1",
        "grok-imagine-video",
        "XAI_API_KEY",
        "data_uri",
    ),
}


@dataclass(frozen=True)
class Settings:
    profiles: dict[str, Profile] = field(default_factory=lambda: dict(BUILTINS))
    tos: TosConfig = field(default_factory=TosConfig)
    state_dir: Path = field(default_factory=default_state_dir)
    ffmpeg: str | None = None
    ffprobe: str | None = None
    poll_interval: float = 5.0
    wait_timeout: float = 600.0
    config_path: Path | None = None

    def profile(self, name: str) -> Profile:
        if name not in self.profiles:
            raise McsfxError("CONFIG_ERROR", f"Unknown profile: {name}")
        return self.profiles[name]


def validate_url(value: str, label: str) -> None:
    try:
        url = urlsplit(value)
        valid = (
            url.scheme == "https"
            and url.hostname
            and not url.username
            and not url.password
            and not url.query
            and not url.fragment
        )
        _ = url.port
    except ValueError:
        valid = False
    if not valid:
        raise McsfxError(
            "CONFIG_ERROR", f"{label} must be an HTTPS URL without credentials or query."
        )


def validate_profile(profile: Profile) -> None:
    presets = {"xai": {"grok-edit", "grok-keyframe"}, "ark": {"seedance-2.5-edit"}}
    if profile.protocol not in presets or profile.preset not in presets[profile.protocol]:
        raise McsfxError("CONFIG_ERROR", "Profile protocol and capability preset do not match.")
    if profile.transport not in {"data_uri", "tos"} or (
        profile.protocol == "ark" and profile.transport != "tos"
    ):
        raise McsfxError("CONFIG_ERROR", "Ark requires TOS transport; xAI allows data_uri or tos.")
    if profile.preset == "grok-keyframe" and profile.transport != "data_uri":
        raise McsfxError("CONFIG_ERROR", "grok-keyframe requires PNG data_uri transport.")
    if profile.download_mode not in {"public", "same_origin_content"} or (
        profile.download_mode == "same_origin_content" and profile.protocol != "xai"
    ):
        raise McsfxError("CONFIG_ERROR", "same_origin_content downloads require xAI protocol.")
    validate_url(profile.base_url, "base_url")
    if not profile.model.strip() or not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", profile.api_key_env
    ):
        raise McsfxError(
            "CONFIG_ERROR", "A model and valid API key environment variable are required."
        )


def load_settings(config_path: str | None = None, state_dir: str | None = None) -> Settings:
    explicit = config_path or os.environ.get("MCSFX_CONFIG")
    path = Path(explicit).expanduser() if explicit else Path.cwd() / "mcsfx.toml"
    data = {}
    if explicit or path.exists():
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
            raise McsfxError("CONFIG_ERROR", "Cannot read valid UTF-8 TOML configuration.") from exc
    else:
        path = None
    try:
        if set(data) - {"profiles", "tos", "tools", "runtime"}:
            raise ValueError("Unknown configuration section")
        profiles = dict(BUILTINS)
        for name, raw in data.get("profiles", {}).items():
            base = BUILTINS.get(name)
            values = base.snapshot() if base else {"name": name}
            if "name" in raw:
                raise ValueError("Profile name belongs in the table heading")
            values.update(raw)
            profile = Profile(**values)
            if not all(isinstance(v, str) for v in profile.snapshot().values()):
                raise ValueError("Profile fields must be strings")
            validate_profile(profile)
            profiles[name] = profile
        tos = TosConfig(**data.get("tos", {}))
        tools = data.get("tools", {})
        runtime = data.get("runtime", {})
        if set(tools) - {"ffmpeg", "ffprobe"} or set(runtime) - {
            "state_dir",
            "poll_interval",
            "wait_timeout",
        }:
            raise ValueError("Unknown tools/runtime option")
        poll = float(runtime.get("poll_interval", 5))
        timeout = float(runtime.get("wait_timeout", 600))
        if not 0.1 <= poll <= 300 or not 0 < timeout <= 259200:
            raise ValueError("Invalid polling/wait limit")
        state = state_dir or os.environ.get("MCSFX_STATE_DIR") or runtime.get("state_dir")
        return Settings(
            profiles=profiles,
            tos=tos,
            state_dir=Path(state).expanduser().resolve() if state else default_state_dir(),
            ffmpeg=os.environ.get("MCSFX_FFMPEG") or tools.get("ffmpeg"),
            ffprobe=os.environ.get("MCSFX_FFPROBE") or tools.get("ffprobe"),
            poll_interval=poll,
            wait_timeout=timeout,
            config_path=path.resolve() if path else None,
        )
    except (TypeError, ValueError, AttributeError) as exc:
        raise McsfxError("CONFIG_ERROR", "Invalid or unknown TOML configuration field.") from exc
