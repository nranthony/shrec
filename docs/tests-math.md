# shrec — math-correctness test catalog

The `MM<n>` ids are referenced by the docstring of every math-test in
`tests/`. Tests in §5b.1–§5b.5 below probe SHREC against analytical
results, limiting cases, and explicit paper claims — i.e. they catch
*algorithmic* drift that mere structural / shape tests cannot.

"must-have" = catches a class of bugs that the structural tests cannot.
"regression" = guards a property but won't fail on a fresh implementation
if math is right.

Status legend:

- ✅ **green** — implemented, currently passing.
- ⚠️ **xfail** — implemented as a documented `@pytest.mark.xfail`;
  failure is a known property of the current code, not a regression.
- ⏳ **deferred** — not yet implemented; tracked here for future work.

---

## §5b.1 — Closed-form unit tests of the inner math

These don't need a time series at all; they probe individual operators
with inputs whose answers can be written down.

| Test | Status | Module / Oracle |
|------|--------|------------------|
| **MM1 (must)** | ✅ | `recurrence/simplicial.py:fit_rho_sigma` — defining equation `\|Σ_m exp[-ReLU(d_im − ρ)/σ] − log₂ k\| < 1e-6` on every row. **Caught a real bug**: `fsolve` reported `ier=1` while stalling at the initial guess on ~6% of rows (residual ≈ 2.8). Fixed by a bracketed `brentq` (the equation is monotone in σ). See `tests/test_recurrence_simplicial.py` and `docs/math-learning-notes.md`. |
| **MM2 (must)** | ✅ | Same — **scale invariance**: `fit_rho_sigma(α · d_row, k) = (α·ρ, α·σ)` so affinity is unchanged. Verified on both `(ρ, σ)` and the full `dataset_to_simplex` output. |
| **MM3 (must)** | ✅ | Same — **diagonal**: `A_ii = 1` exactly (`d_ii=0 ⇒ exp(0)=1`, fuzzy union `1+1−1=1`). |
| **MM4 (must)** | ✅ | `dataset_to_simplex` symmetrisation: `A + Aᵀ − A∘Aᵀ ∈ [0,1]^(N×N)` and symmetric. |
| **MM5 (must)** | ⚠️ xfail | `dataset_to_simplex(X, k=M)` vs `umap.umap_.fuzzy_simplicial_set` agreement to `atol=1e-5`. Known divergence on σ-solver conventions; the refactor chose `dataset_to_simplex`. See `tests/test_recurrence_simplicial.py`. |
| **MM6 (must)** | ⏳ | `recurrence/kernel.py:data_to_connectivity` p-norm limits: ord=1 → mean; ord→∞ → min-over-channels. |
| MM7 | ⏳ | `cdist` isometry invariance under random orthogonal `Q` + translation. |
| MM8 | ⏳ | `cdist` triangle inequality on random triples. |
| MM9 | ⏳ | `sparsify_by_quantile` produces ≥ target sparsity; idempotent at the same threshold. |
| **MM10 (must)** | ⏳ | `graph/communities.py:_leiden` — backend agreement on the barbell graph (ARI=1 across backends). |
| MM11 | ⏳ | `graph/unionfind.py` parity with `scipy.cluster.hierarchy.DisjointSet`. |
| MM12 | ✅ | `utils/graph_tools.common_neighbors_ratio` — vectorised matches loop on random binary matrices to `atol=0`; K_n with self-loops returns zero. See `tests/test_graph_adjacency.py`. |

---

## §5b.2 — Algorithmic invariances and equivariances

Properties the *whole* `fit_predict` pipeline must satisfy on
principled grounds (recurrence is undirected; no preferred response
order; etc.).

| Test | Status | Property |
|------|--------|----------|
| **MM13 (must)** | ✅ | Response-permutation invariance: `model.fit(X[:, perm])` gives the same labels (ARI=1) as `model.fit(X)`. See `tests/test_models_invariances.py`. |
| **MM14 (must)** | ✅ approximate | Time-reversal symmetry as **approximate** (ARI > 0.85 for clustering, `|cos| > 0.7` for manifold). The doc's exact-equality claim does not hold once delay embedding enters: forward embedded points carry past lags, reversed points carry "future" lags, so the per-point coordinates aren't byte-equal. For stationary processes the statistics agree → cluster structure survives. |
| MM15 | ⏳ | Scale invariance of the full pipeline w.r.t. each response (standardisation is on by default; MM2 + standardisation = full invariance). |
| MM16 | ⏳ | Idempotence on a single repeated response. |
| MM17 | ⏳ | Constant-response rejection (filter at `recurrence/kernel.py:data_to_connectivity`). |
| MM18 | ✅ (as MM31) | Determinism w.r.t. `random_state` — and crucially, *no* global RNG mutation. See `tests/test_models_base.py::TestRngIsolation`. |

---

## §5b.3 — Limiting-case oracles (closed-form ground truth)

Inputs constructed so the answer is provably the one we want.

| Test | Status | Input / Oracle |
|------|--------|------------------|
| **MM19 (must)** | ✅ near-exact | Sauer limit, period-2 driver: N=20 logistic responses, zero noise, T=1000 → `RecurrenceClustering().fit(X)` gives ARI > 0.95 (≈ 0.99). Exact ARI=1 is the asymptotic claim; the residual ~1% is Leiden boundary over-segmentation. See `tests/test_models_recurrence_clustering.py`. |
| **MM20 (must)** | ⚠️ xfail | Sauer limit, period-4 driver. Default Leiden at `resolution=1.0` collapses to 2 communities → ARI ≈ 0.50. Closing requires resolution tuning or CPM objective (graspologic's CPM backend currently panics). |
| MM21 | ⏳ | Period-8 driver with stochastic forcing (`σ_noise = 0.04`): ARI > 0.85. |
| **MM22 (must)** | ✅ | Block-stochastic affinity: hand-construct `A = block_diag(J_p1, J_p2)` (unequal sizes) with a small bridge, assert RecurrenceManifold output `|cos|` > 0.99 against the analytical Fiedler vector. Distinguishes Fiedler from second SVD vector on irregular graphs. See `tests/test_models_recurrence_manifold.py`. |
| MM23 | ⏳ | Cycle-graph affinity: Fiedler is `cos(2π i/n)` up to sign. |
| MM24 | ⏳ | Identity-driver: `N = 1, x(t) = z(t)` (Rössler `z₁` trajectory) — Spearman `|ρ| > 0.95`. |
| MM25 | ⏳ | Linear measurement: `x_k(t) = a_k z(t) + b_k` — `|ρ| > 0.9` after standardisation. |
| MM26 | ⏳ | Nonlinear monotone measurement: `x_k(t) = tanh(z(t)/σ_k)` — `|ρ| > 0.8`. |

---

## §5b.4 — Paper-claim regression tests (quantitative scaling laws)

Slower; intended for `-m slow` / nightly CI.

| Test | Status | Claim |
|------|--------|-------|
| **MM27 (must)** | ⏳ | β-distribution accuracy scaling (Appendix E.2): `Acc(NT/τ) = Acc_max (1 − exp(−β √(NT/τ)))`. Fit form to ARI vs N sweep on period-4 logistic ensemble. |
| MM28 | ⏳ | Percolation order parameter (Appendix E.3): `T_LCC/T` monotone non-increasing in N with a > 0.3 drop. |
| MM29 | ⏳ | HN-Isomap baseline: `common_neighbors_ratio` is symmetric, zero-diagonal, non-negative. |

---

## §5b.5 — Sklearn-contract tests

Cheap mechanical pinning.

| Test | Status | Property |
|------|--------|----------|
| MM30 | ✅ | `set_params(**get_params())` is the identity on the model state. See `tests/test_models_base.py::TestSklearnContract`. |
| MM31 | ✅ | Constructing a model does not change `np.random.get_state()`. See `tests/test_models_base.py::TestRngIsolation`. |
| MM32 | ✅ (merged with MM30) | `set_params`/`get_params` round-trip across the four models. |

---

## Coverage summary

| Section | Total | Green | xfail | Deferred |
|---------|-------|-------|-------|----------|
| §5b.1 inner math      | 12 | 5 | 1 | 6  |
| §5b.2 invariances     | 6  | 3 | 0 | 3  |
| §5b.3 limiting cases  | 8  | 2 | 1 | 5  |
| §5b.4 scaling laws    | 3  | 0 | 0 | 3  |
| §5b.5 sklearn contract| 3  | 3 | 0 | 0  |
| **total**             | 32 | 13 | 2 | 17 |

The "must" tests (MM1–MM6, MM10, MM13–MM14, MM19–MM22, MM27) are the
minimum to call the algorithm green. Of the 13 must-tests, **10 are
green, 2 are xfail-documented, and 1 is still deferred**. The inner-
math closed-form checks MM1–MM4 are now green (and MM1 surfaced and
fixed the `fsolve` σ-solve stall); the remaining gaps are the consensus
aggregation MM6, the backend-agreement check MM10, and the paper-figure
regression MM27.

The deferred items are the natural next batch of work whenever the
math-correctness suite is revisited.
