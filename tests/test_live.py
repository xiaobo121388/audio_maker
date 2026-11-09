"""Explicit opt-in paid smoke tests; never invoked by default test runs."""

import os
from pathlib import Path

import pytest

from mcsfx.config import load_settings
from mcsfx.jobs import JobRunner

PROFILES = [
    name.strip()
    for name in os.environ.get("MCSFX_LIVE_PROFILES", "grok,seedance").split(",")
    if name.strip()
]


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("MCSFX_LIVE_TESTS") != "1",
    reason="Paid model tests are disabled; set MCSFX_LIVE_TESTS=1 explicitly.",
)
@pytest.mark.parametrize("profile", PROFILES)
def test_paid_short_clip(profile):
    source = os.environ.get("MCSFX_LIVE_VIDEO")
    if not source:
        pytest.skip(
            "MCSFX_LIVE_VIDEO must point to the same 4-8 second game clip for all providers."
        )
    output = Path(os.environ.get("MCSFX_LIVE_OUT", "outputs/live")).resolve()
    settings = load_settings(os.environ.get("MCSFX_LIVE_CONFIG"), str(output / ".state"))
    if not os.environ.get(settings.profile(profile).api_key_env):
        pytest.skip("The selected profile's credential is not configured.")
    runner = JobRunner(settings)
    submission = runner.generate(
        Path(source),
        profile_name=profile,
        mode="preserve",
        prompt="Match the visible action impacts precisely.",
        out=output,
    )
    result = runner.fetch(submission["job_id"], wait=True, timeout=1800)
    assert result["status"] == "completed"
    assert Path(result["artifacts"]["generated"]).is_file()
    assert Path(result["artifacts"]["result"]).is_file()
    assert "alignment_unverified" in result["warnings"]
