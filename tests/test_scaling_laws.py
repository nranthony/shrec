"""Paper-claim regression tests — quantitative scaling laws (docs/tests-math.md
§5b.4). Slow; run under `-m slow` / nightly CI.

MM27 — accuracy scaling with total data `NT/τ` (paper Appendix E.2):

    Acc(NT/τ) = Acc_max · (1 − exp(−β·√(NT/τ)))

i.e. reconstruction accuracy rises and saturates as more responses are added.
The paper states this for the discrete-driver ARI; we test it on the
continuous-driver path (RecurrenceManifold + Spearman |ρ|), which both (a)
sidesteps the period-4 Leiden-resolution collapse that caps the discrete ARI
(MM20, xfail) and (b) gives a smooth accuracy in [0, 1] that the saturating
form can be fit against. The scaling only becomes visible in a mildly
data-limited regime, so we add light observation noise (σ=0.05) — with none,
accuracy saturates at N=2 and there is nothing to fit. See
docs/math-learning-notes.md (Round 5).
"""
import numpy as np
import pytest
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

from shrec.models import RecurrenceManifold

T = 500
TAU = 1
N_SWEEP = np.array([2, 4, 8, 16, 32])
N_SEEDS = 8
OBS_NOISE = 0.05


def _logistic_ensemble(driver, n_responses, seed, obs_noise=OBS_NOISE):
    """`n_responses` chaotic logistic maps forced by `driver`, each observed
    through light additive noise. Distinct seed per (N, seed) cell."""
    rng = np.random.default_rng(1000 * seed + n_responses)
    n = len(driver)
    r_values = rng.uniform(3.7, 3.9, size=n_responses)
    X = np.empty((n, n_responses))
    for k, r in enumerate(r_values):
        x = np.empty(n)
        x[0] = rng.uniform(0.1, 0.9)
        for t in range(n - 1):
            x[t + 1] = np.clip(r * x[t] * (1 - x[t]) + 0.4 * driver[t], 0.0, 1.0)
        X[:, k] = x + obs_noise * rng.standard_normal(n)
    return X


@pytest.mark.slow
class TestAccuracyScalingLaw:
    """MM27 — fit the saturating-exponential accuracy law and assert its
    qualitative content: accuracy grows with total data (β > 0), saturates at
    a usable level (Acc_max), improves markedly from the smallest to the
    largest ensemble, and the saturating form fits better than a flat line.
    """

    def test_accuracy_saturates_with_ensemble_size(self):
        driver = 0.5 + 0.4 * np.sin(2 * np.pi * np.arange(T) / 100.0)

        mean_acc = np.empty(len(N_SWEEP))
        for i, n_resp in enumerate(N_SWEEP):
            accs = []
            for seed in range(N_SEEDS):
                X = _logistic_ensemble(driver, int(n_resp), seed)
                v = RecurrenceManifold(random_state=1).fit(X).labels_
                accs.append(abs(spearmanr(v, driver).correlation))
            mean_acc[i] = np.mean(accs)

        # Fit Acc(x) = Acc_max (1 − exp(−β x)) with x = √(NT/τ).
        x = np.sqrt(N_SWEEP * T / TAU)

        def law(x, acc_max, beta):
            return acc_max * (1.0 - np.exp(-beta * x))

        (acc_max, beta), _ = curve_fit(
            law, x, mean_acc, p0=[0.8, 1e-3],
            bounds=([0.0, 0.0], [1.0, 1.0]), maxfev=10000,
        )

        # Goodness of fit vs a flat-mean baseline.
        resid = mean_acc - law(x, acc_max, beta)
        r2 = 1.0 - np.sum(resid ** 2) / np.sum((mean_acc - mean_acc.mean()) ** 2)

        assert beta > 0.0, "accuracy did not increase with total data NT/τ"
        assert 0.6 <= acc_max <= 1.0, f"saturation accuracy {acc_max:.2f} off"
        assert mean_acc[-1] - mean_acc[0] > 0.25, (
            f"no clear accuracy gain from N={N_SWEEP[0]} to N={N_SWEEP[-1]}: "
            f"{mean_acc[0]:.2f} → {mean_acc[-1]:.2f}"
        )
        assert r2 > 0.5, (
            f"saturating-exponential form did not fit (R²={r2:.2f}); "
            f"means={np.round(mean_acc, 3)}"
        )
        # Near-monotone in the means (allow small sampling dips).
        assert np.all(np.diff(mean_acc) > -0.05)
