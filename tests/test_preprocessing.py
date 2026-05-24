"""Regression tests for utils/preprocessing.py helpers used by the
RecurrenceModel base class.

Tied to docs/history/2026-05-refactor-assessment.md §3A bug A9.
"""
import numpy as np
import pytest

from shrec.utils.preprocessing import allclose_len, nan_fill


# --- §3A.A9 — allclose_len silent broadcast bug -----------------------------

class TestAllcloseLen:
    """Pre-fix: `np.allclose` broadcasts mismatched-but-broadcastable shapes
    and returned True for, e.g., `(3,)` vs `(2, 3)`. The docstring promised
    "False if different shapes" but the try/except only caught hard
    ValueError on non-broadcastable shapes (e.g. `(3,)` vs `(4,)`).

    Consequence: in ClassicalRecurrenceClustering.fit, two index lists of
    different length could be silently treated as equal, mis-assigning a
    timepoint to an earlier cluster's label.
    """

    def test_equal_arrays_return_true(self):
        assert allclose_len(np.array([1, 2, 3]), np.array([1, 2, 3])) is True

    def test_close_but_not_equal_within_tol(self):
        a = np.array([1.0, 2.0, 3.0])
        b = a + 1e-9
        assert allclose_len(a, b) is True

    def test_different_lengths_return_false(self):
        # Already correct pre-fix (np.allclose raises ValueError on
        # (3,) vs (4,) which the try/except catches).
        assert allclose_len(np.array([1, 2, 3]), np.array([1, 2, 3, 4])) is False

    def test_broadcastable_mismatched_shapes_return_false(self):
        # This is the regression: pre-fix `np.allclose([1,2,3], [[1,2,3],
        # [1,2,3]])` returns True via broadcasting. Post-fix, the explicit
        # shape check returns False.
        a = np.array([1, 2, 3])
        b = np.array([[1, 2, 3], [1, 2, 3]])
        assert allclose_len(a, b) is False, (
            "allclose_len must distinguish shape (3,) from (2, 3); "
            "the broadcasting bug returned True here."
        )

    def test_scalar_vs_array_returns_false(self):
        # Same root cause: scalar broadcasts to any array shape.
        assert allclose_len(np.float64(1.0), np.array([1.0, 1.0, 1.0])) is False


# --- nan_fill — utility used adjacent to A1 ---------------------------------

class TestNanFill:
    """Not a §3A bug per se, but exercised by the same code path as the A1
    fix; kept here to pin its current contract."""

    def test_passes_through_when_no_nan(self):
        x = np.arange(10.0).reshape(5, 2)
        np.testing.assert_array_equal(nan_fill(x), x)

    def test_back_fills_along_first_axis(self):
        x = np.array([[1.0, 2.0], [np.nan, np.nan], [3.0, 4.0]])
        out = nan_fill(x)
        # nan_fill copies the previous row forward.
        np.testing.assert_array_equal(out[1], out[0])
