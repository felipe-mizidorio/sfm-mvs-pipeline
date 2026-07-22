"""The unscaled-output opt-in must be wired into both entry-point scripts.

The policy logic is unit-tested in test_scale_policy.py. These tests guard the
wiring: that the flag exists, that it reaches `args.allow_unscaled`, and above
all that it defaults to OFF. A default flip here would silently restore the
failure path the policy exists to close, without breaking any other test.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = ["run_pipeline.py", "resume_from_mvs.py"]


def _load_script(name: str):
    path = _REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"_script_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse(module, argv: list[str]):
    original = sys.argv
    sys.argv = ["prog", *argv]
    try:
        return module._parse_args()
    finally:
        sys.argv = original


@pytest.mark.parametrize("script", _SCRIPTS)
def test_unscaled_output_is_off_by_default(script: str) -> None:
    """The operator must opt in to unscaled output, never out of it."""
    module = _load_script(script)

    args = _parse(module, ["--output-dir", "out", "--image-dir", "imgs"])

    assert args.allow_unscaled is False


@pytest.mark.parametrize("script", _SCRIPTS)
def test_allow_unscaled_flag_is_accepted(script: str) -> None:
    module = _load_script(script)

    args = _parse(
        module, ["--output-dir", "out", "--image-dir", "imgs", "--allow-unscaled"]
    )

    assert args.allow_unscaled is True
