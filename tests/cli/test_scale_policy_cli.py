"""The unscaled-output opt-in must be wired into both entry-point scripts.

The policy logic is unit-tested in test_scale_policy.py. These tests guard the
wiring: that the flag exists, that it reaches `args.allow_unscaled`, and above
all that it defaults to OFF. A default flip here would silently restore the
failure path the policy exists to close, without breaking any other test.
"""

import importlib
import sys

import pytest

_SCRIPTS = [
    "sfm_mvs_pipeline.cli.run",
    "sfm_mvs_pipeline.cli.resume_mvs",
    "sfm_mvs_pipeline.cli.resume_dense",
]


def _load_script(name: str):
    return importlib.import_module(name)


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
