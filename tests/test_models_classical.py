"""Regression tests for ClassicalRecurrenceClustering.

Tied to docs/history/2026-05-refactor-assessment.md §3A bug A3.
"""
import numpy as np
import pytest

from shrec.models import ClassicalRecurrenceClustering


# --- §3A.A3 — use_sparse kwarg passed to data_to_connectivity ----------------

class TestUseSparseKwargRegression:
    """`ClassicalRecurrenceClustering.fit` previously forwarded
    `use_sparse=False` to `data_to_connectivity`, which does not accept
    that kwarg → TypeError on first call. After the fix, `fit` must run
    end-to-end on a small synthetic input.
    """

    def test_fit_runs_on_simple_periodic_input(self, tiled_sine):
        # Pre-fix: TypeError("data_to_connectivity() got an unexpected
        # keyword argument 'use_sparse'") at line 726.
        data = tiled_sine(T=120, n_series=3)
        model = ClassicalRecurrenceClustering(
            tolerance=0.05, random_state=0
        )
        model.fit(data)
        # The class assigns self.labels_ to a length-T array of integers.
        assert hasattr(model, "labels_")
        assert len(model.labels_) == data.shape[0]
        assert model.labels_.dtype.kind in ("i", "u")  # integer labels

    def test_fit_predict_runs(self, tiled_sine):
        """`fit_predict` is the public sklearn-style entry point — must work
        too, since it just calls `fit` underneath.
        """
        data = tiled_sine(T=120, n_series=3)
        model = ClassicalRecurrenceClustering(
            tolerance=0.05, random_state=0
        )
        labels = model.fit_predict(data)
        assert len(labels) == data.shape[0]
