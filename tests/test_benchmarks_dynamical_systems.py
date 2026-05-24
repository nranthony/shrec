"""Regression tests for benchmarks/dynamical_systems.py.

Tied to docs/history/2026-05-refactor-assessment.md §3A bug A5.
"""
import os
import sys

import numpy as np
import pytest

# benchmarks/ is not packaged; add to sys.path the same way the benchmark
# notebooks do.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from benchmarks.dynamical_systems import DrivenLorenz, DrivenLogistic  # noqa: E402


# --- §3A.A5 — DrivenLorenz.self.ar undefined --------------------------------

class TestDrivenLorenzCoupling:
    """Pre-fix: `DrivenLorenz.rhs_response_ensemble` references `self.ar`,
    which was commented out in __init__. Every call to `rhs()` raised
    AttributeError, making DrivenLorenz unusable as a benchmark fixture.
    """

    def test_ar_attribute_is_defined(self):
        sys_ = DrivenLorenz(random_state=0, driver="rossler")
        # The whole class was broken until `self.ar` was uncommented.
        assert hasattr(sys_, "ar"), (
            "DrivenLorenz.ar must be defined; rhs_response_ensemble "
            "depends on it."
        )
        assert np.isfinite(sys_.ar)

    def test_rhs_runs_without_attribute_error(self):
        sys_ = DrivenLorenz(random_state=0, driver="rossler")
        n = sys_.n_drive + 3 * sys_.n_sys
        X0 = np.random.default_rng(0).standard_normal(n)
        out = sys_.rhs(0.0, X0)  # pre-fix: AttributeError("'DrivenLorenz' object has no attribute 'ar'")
        assert len(out) == n
        assert np.all(np.isfinite(out))


class TestDrivenLogisticSmoke:
    """DrivenLogistic was not flagged in §3A; included here to give the bug-
    fix PR a green smoke baseline alongside DrivenLorenz."""

    def test_rhs_runs(self):
        sys_ = DrivenLogistic(n_response=4, random_state=0)
        X0 = np.array([0.3] + [0.4] * 4)
        out = sys_.rhs(0.0, X0)
        assert len(out) == 5
        assert np.all(np.isfinite(out))
