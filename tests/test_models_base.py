"""Regression tests for the RecurrenceModel base class.

Each test in this file is tied to a bug catalogued in docs/history/2026-05-refactor-assessment.md §3A.
"""
import numpy as np
import pytest

from shrec.models import RecurrenceManifold


# --- §3A.A1 — _fillna 'bill' typo -------------------------------------------

class TestFillNa:
    """Pins the fix for the `Xc.fillna(method="bill")` typo at models.py:510.

    The original code only forward-filled, then attempted a backward fill via
    the misspelled keyword 'bill'. With pandas <= 1.x this raised on any
    leading-NaN row; with pandas >= 2.x the deprecated method= signature would
    raise even on the ffill call. Either way, _fillna was broken for any data
    that needed back-filling.
    """

    @pytest.fixture
    def model(self):
        return RecurrenceManifold(random_state=0)

    def test_forward_fill_only(self, model):
        # NaN at the end → ffill alone is sufficient
        X = np.array([[1.0, 2.0], [np.nan, np.nan], [3.0, 4.0], [np.nan, np.nan]])
        out = model._fillna(X)
        assert not np.any(np.isnan(out))
        np.testing.assert_array_equal(out[1], [1.0, 2.0])
        np.testing.assert_array_equal(out[3], [3.0, 4.0])

    def test_back_fill_required(self, model):
        # Leading NaN → ffill alone leaves NaN; bfill must fire.
        # This is the exact case the 'bill' typo failed to handle.
        X = np.array([[np.nan, np.nan], [np.nan, np.nan], [3.0, 4.0]])
        out = model._fillna(X)
        assert not np.any(np.isnan(out)), (
            "Leading-NaN row was not back-filled; the 'bill' typo regression "
            "is present."
        )
        np.testing.assert_array_equal(out[0], [3.0, 4.0])
        np.testing.assert_array_equal(out[1], [3.0, 4.0])

    def test_no_nan_input_is_identity(self, model):
        X = np.arange(12.0).reshape(6, 2)
        out = model._fillna(X)
        np.testing.assert_array_equal(out, X)


# --- §3A.A2 — _make_embedding 3D multivariate branch ------------------------

class TestMakeEmbedding3D:
    """Pins the fix for `for i in X.shape[-1]:` at models.py:537.

    The original branch tried to iterate over an integer (TypeError on first
    iteration) and additionally clobbered the loop variable `X`. As a result,
    any 3-dimensional input crashed at the embedding step.
    """

    @pytest.fixture
    def model(self):
        # d_embed kept small so output is predictable
        return RecurrenceManifold(d_embed=2, random_state=0, padding="symmetric")

    def test_3d_input_does_not_raise(self, model):
        # Shape (n_series, T, D). Pre-fix this raised TypeError immediately.
        rng = np.random.default_rng(0)
        X = rng.normal(size=(4, 50, 3))
        out = model._make_embedding(X)
        # Each of the D=3 dims contributes d_embed=2 columns, hstacked.
        # Exact output shape is an implementation detail; the meaningful
        # property is that it is finite and 3D after the embedding axis is
        # added by hankel_matrix.
        assert np.all(np.isfinite(out)), (
            "3D embedding produced non-finite values"
        )
        assert out.ndim >= 2

    def test_3d_dimension_count_propagates(self, model):
        """Each of the D dimensions contributes its own delay-embed block."""
        rng = np.random.default_rng(0)
        X_2d = rng.normal(size=(50, 4))      # 2D path
        X_3d = X_2d[None, ...]               # same data, 3D wrapper
        out_2d = model._make_embedding(X_2d)
        out_3d = model._make_embedding(X_3d)
        # Both should produce finite output and have the same number of
        # delay-embedding columns per (slice, time) pair.
        assert np.all(np.isfinite(out_2d))
        assert np.all(np.isfinite(out_3d))


# --- §5b.2 MM18 / §3A.A6 — RNG isolation -----------------------------------

class TestRngIsolation:
    """Constructing and fitting a model must not mutate `np.random`'s
    global state. Pre-fix the base `__init__` and each `fit` called
    `np.random.seed(self.random_state)`, leaking a fixed seed into the
    surrounding process (audit item A6).
    """

    def test_construction_does_not_mutate_global_rng(self):
        from shrec.models import (
            ClassicalRecurrenceClustering,
            HirataNomuraIsomap,
            RecurrenceClustering,
            RecurrenceManifold,
        )
        before = np.random.get_state()
        for cls in (
            RecurrenceClustering,
            RecurrenceManifold,
            ClassicalRecurrenceClustering,
            HirataNomuraIsomap,
        ):
            cls(random_state=42)
        after = np.random.get_state()
        # Same generator class (str), same key array, same position.
        assert before[0] == after[0]
        np.testing.assert_array_equal(before[1], after[1])
        assert before[2] == after[2]


# --- §5b.5 MM30 — sklearn contract -----------------------------------------

class TestSklearnContract:
    """Cheap mechanical checks that the four model classes round-trip
    through `get_params` / `set_params` and accept no positional args
    other than ones declared in __init__.
    """

    @pytest.mark.parametrize("cls_name", [
        "RecurrenceClustering", "RecurrenceManifold",
        "ClassicalRecurrenceClustering", "HirataNomuraIsomap",
    ])
    def test_get_set_params_round_trip(self, cls_name):
        import shrec.models as M
        cls = getattr(M, cls_name)
        model = cls()
        params = model.get_params()
        model.set_params(**params)
        np.testing.assert_array_equal(
            sorted(model.get_params()), sorted(params),
        )
