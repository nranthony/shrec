"""Gate for `scripts/refs.py`: the repo has no CI, so pytest runs it.

A path that a doc or comment names must exist, and the check's own tests must
pass. Both scripts are stdlib-only and live outside `testpaths`.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("script", ["refs.py", "refs_test.py"])
def test_reference_check_is_clean(script):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
