"""Shared pytest fixtures for the shrec test suite."""
import numpy as np
import pytest


@pytest.fixture
def rng():
    """Deterministic numpy Generator. Use this in preference to np.random.seed
    so tests do not leak global RNG state.
    """
    return np.random.default_rng(0)


@pytest.fixture
def tiled_sine():
    """Factory: tiled-sine response stack of shape (T, n_series).

    Matches the synthetic input used by the legacy tests in test_models.py
    so the regression tests are comparable.
    """
    def _build(T=300, n_series=3, end=2.0):
        return np.tile([np.sin(np.linspace(0, end, T))], (n_series, 1)).T
    return _build


@pytest.fixture
def square_wave_responses():
    """Factory: ensemble of distinct logistic-map responses driven by a
    period-2 square wave. Used for the Sauer-limit regression tests later;
    included now so the §3A bug-fix tests can adopt it without churn.
    """
    def _build(T=1000, n_responses=20, coupling=0.5, seed=0):
        rng = np.random.default_rng(seed)
        z = np.where(np.arange(T) % 2 == 0, 0.2, 0.8)  # period-2 driver
        responses = np.empty((T, n_responses))
        r_values = rng.uniform(3.81, 3.97, size=n_responses)
        for k, r in enumerate(r_values):
            x = np.empty(T)
            x[0] = rng.uniform(0.1, 0.9)
            for t in range(T - 1):
                x[t + 1] = r * x[t] * (1 - x[t]) + coupling * z[t]
                x[t + 1] = np.clip(x[t + 1], 0.0, 1.0)
            responses[:, k] = x
        return responses, z
    return _build
