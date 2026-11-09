"""Small synchronous CLI with machine-readable, resumable job commands."""

import argparse
import json
import os
import sys
from pathlib import Path

from mcsfx import __version__
from mcsfx.config import load_settings
from mcsfx.errors import McsfxError
from mcsfx.jobs import JobRunner, envelope
from mcsfx.media import MediaTools
from mcsfx.storage import validate_tos


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise McsfxError(
            "ARGUMENT_ERROR", "Invalid command arguments; run mcsfx --help.", exit_code=2
        )


def positive_seconds(value: str) -> float:
    try:
        seconds = float(value)
        if not 0 < seconds <= 259200:
            raise ValueError
        return seconds
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected a positive timeout up to 259200 seconds."
        ) from exc


def build_parser() -> Parser:
    parser = Parser(prog="mcsfx", description="Generate sound-effect videos from short game clips.")
    parser.add_argument("--version", action="version", version=f"mcsfx {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--json", action="store_true", help="Emit one versioned JSON object on stdout."
    )
    common.add_argument("--config", help="TOML file (or MCSFX_CONFIG; defaults to ./mcsfx.toml).")
    common.add_argument("--state-dir", help="Job registry (or MCSFX_STATE_DIR).")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser(
        "doctor",
        parents=[common],
        help="Check local tools and configuration without network calls.",
    )
    doctor.add_argument("--profile", help="Check a single profile; otherwise report all profiles.")
    generate = commands.add_parser(
        "generate",
        parents=[common],
        help="Submit one generation request, never automatically resubmitted.",
    )
    generate.add_argument("video", type=Path)
    generate.add_argument("--profile", default="grok", help="Profile name, including grok-sfx.")
    generate.add_argument(
        "--mode",
        choices=["preserve", "regenerate"],
        default="preserve",
        help="preserve keeps strict source timing; regenerate is recommended for grok-sfx.",
    )
    generate.add_argument("--prompt", default="")
    generate.add_argument("--out", type=Path, default=Path("outputs"))
    generate.add_argument("--dry-run", action="store_true")
    generate.add_argument("--wait", action="store_true")
    generate.add_argument(
        "--timeout",
        type=positive_seconds,
        help="Local wait deadline in seconds; does not cancel the remote task.",
    )
    for name in ("status", "fetch", "cleanup"):
        command = commands.add_parser(name, parents=[common])
        command.add_argument("job_id")
        if name == "fetch":
            command.add_argument("--wait", action="store_true")
            command.add_argument("--timeout", type=positive_seconds)
    return parser


def doctor(settings, profile_name: str | None) -> dict:
    result = envelope({"status": "checked"})
    try:
        result["tools"] = {"ok": True, **MediaTools(settings.ffmpeg, settings.ffprobe).check()}
    except McsfxError as exc:
        result["tools"] = {"ok": False, "error": exc.as_dict()}
    profiles = (
        [settings.profile(profile_name)] if profile_name else list(settings.profiles.values())
    )
    result["profiles"] = {}
    for profile in profiles:
        ready = bool(os.environ.get(profile.api_key_env, "").strip())
        check = {
            "ready": ready,
            "protocol": profile.protocol,
            "preset": profile.preset,
            "model": profile.model,
            "transport": profile.transport,
            "download_mode": profile.download_mode,
            "api_key_env": profile.api_key_env,
            "key_present": ready,
        }
        if profile.transport == "tos":
            try:
                validate_tos(settings.tos)
                check["tos_ready"] = True
            except McsfxError as exc:
                check.update(tos_ready=False, ready=False, error=exc.as_dict())
        result["profiles"][profile.name] = check
    result["ok"] = result["tools"]["ok"] and any(p["ready"] for p in result["profiles"].values())
    result["warnings"] = ["local_checks_only_no_provider_or_bucket_connectivity_test"]
    if not result["ok"]:
        result["error"] = McsfxError(
            "NOT_READY", "Required local tools or credentials/configuration are missing."
        ).as_dict()
    return result


def emit(result: dict, machine: bool) -> None:
    if machine:
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return
    print(f"mcsfx: {result['status']}")
    if result.get("job_id"):
        print(f"job_id: {result['job_id']}")
    for key, value in result.get("artifacts", {}).items():
        print(f"{key}: {value}")
    for warning in result.get("warnings", []):
        print(f"warning: {warning}")
    if "tools" in result:
        print(
            json.dumps(
                {"tools": result["tools"], "profiles": result["profiles"]},
                ensure_ascii=False,
                indent=2,
            )
        )
    if "request" in result:
        print(json.dumps(result["request"], ensure_ascii=False, indent=2))
    if result.get("error"):
        print(f"{result['error']['code']}: {result['error']['message']}")


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    machine = "--json" in arguments
    runner = None
    # PowerShell redirects must still receive UTF-8 rather than the active ANSI code page.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        args = build_parser().parse_args(arguments)
        settings = load_settings(args.config, args.state_dir)
        if args.command == "doctor":
            result = doctor(settings, args.profile)
        else:
            runner = JobRunner(settings, progress=lambda text: print(text, file=sys.stderr))
            if args.command == "generate":
                result = runner.generate(
                    args.video,
                    profile_name=args.profile,
                    mode=args.mode,
                    prompt=args.prompt,
                    out=args.out,
                    dry_run=args.dry_run,
                )
                if args.wait and not args.dry_run:
                    result = runner.fetch(result["job_id"], wait=True, timeout=args.timeout)
            elif args.command == "fetch":
                result = runner.fetch(args.job_id, wait=args.wait, timeout=args.timeout)
            elif args.command == "status":
                result = runner.status(args.job_id)
            else:
                result = runner.cleanup(args.job_id)
        emit(result, machine)
        return 0 if result["ok"] else 1
    except KeyboardInterrupt:
        error = McsfxError(
            "INTERRUPTED",
            "Local operation interrupted. Any submitted remote task may still be running; recover using its job ID.",
            retryable=True,
            exit_code=130,
        )
    except McsfxError as exc:
        error = exc
    except OSError:
        error = McsfxError(
            "LOCAL_IO_ERROR",
            "Local file/tool operation failed. Check permissions, free disk space and the job manifest.",
            retryable=True,
        )
    except Exception:  # noqa: BLE001 - Keep upstream exception payloads out of agent output.
        error = McsfxError(
            "INTERNAL_ERROR",
            "Unexpected local failure. Preserve the job files; no automatic resubmission was performed.",
        )
    emit(envelope(runner.current if runner else None, error=error), machine)
    return error.exit_code
