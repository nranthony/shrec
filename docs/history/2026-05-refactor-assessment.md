# 2026-05 refactor assessment (frozen historical record)

> **Status: completed 2026-05-23.** This document is the original
> static audit + modularisation plan, kept as a frozen record of the
> reasoning behind the May 2026 refactor. **Do not edit it as a live
> plan** — for the current architecture map see
> [`docs/architecture.md`](../architecture.md), and for the live math-
> test catalog see [`docs/tests-math.md`](../tests-math.md).
>
> The original §1 (paper summary) and §2 (code → paper mapping) have
> been promoted to `architecture.md`; the math-correctness catalog
> (§5b) has been promoted to `tests-math.md`. What remains below is
> the audit findings (§3), the proposed layout (§4, now implemented),
> the structural test plan (§5), the work plan (§6), and the
> author's caveats (§7) — all of which informed the actual refactor
> and are useful as historical context but should not be read as
> guidance for new work.

---

## 3. Findings — bugs, deviations, and code smells

### A. Outright bugs

| # | Location | Issue |
|---|----------|-------|
| 1 | `models.py:510` | `Xc.fillna(method="bill")` — typo of `"bfill"`. Any call to `_fillna` raises. (Only fires if `fill_nan=True`.) |
| 2 | `models.py:537` | `for i in X.shape[-1]:` iterates over an **integer**, not a range. The 3D multivariate branch of `_make_embedding` is unreachable in practice (`elif len(X.shape) == 3`). |
| 3 | `models.py:726` | `ClassicalRecurrenceClustering.fit` passes `use_sparse=False` to `data_to_connectivity`, which does not accept that kwarg → `TypeError`. The class is unusable as-is. |
| 4 | `models.py:122, 124` | `nx.convert_matrix.from_scipy_sparse_matrix` / `from_numpy_matrix` were removed in NetworkX 3.x (renamed to `_array`). The `cdlib` branch of `_leiden` will break under modern NetworkX. |
| 5 | `benchmarks/dynamical_systems.py:144` | `DrivenLorenz` references `self.ar`, which is commented out at line 64. `rhs` will raise `AttributeError`. |
| 6 | `models.py:495` | `np.random.seed(self.random_state)` runs in `__init__`. Constructing a model **silently mutates global NumPy state** for the rest of the process, which is a footgun when models are constructed in batch (e.g. notebooks, benchmarks). |
| 7 | `utils/graph_tools.py` | `multigraph_to_weighted` is defined **twice** (lines 127 and 246). Second definition shadows the first; harmless but indicates copy-paste. |
| 8 | `models._leiden` | If `method` doesn't match the four supported values, `indices` and `labels` are never assigned, then dereferenced at the return. Needs an explicit `else: raise ValueError`. |
| 9 | `utils/preprocessing.py:allclose_len` | `np.allclose` happily broadcasts mismatched-but-broadcastable shapes (e.g. `(3,)` vs `(3, 4)`) and returns True. The comment promises "False if different shapes". Should check `shape ==` first. |

**Outcome (2026-05):** A1, A2, A3, A4, A5, A7, A8, A9 fixed in the
bug-fix PR with regression tests. A6 fixed in the modularisation PR
(`np.random.default_rng` plumbing; MM18 / MM31 pin it).

### B. Algorithmic deviation from the paper

- **`RecurrenceManifold` uses TruncatedSVD on `A`**, then takes the second right singular vector (line 904–906). The paper specifies the Fiedler eigenvector of the **graph Laplacian** `L = D − A`. For an exactly symmetric, doubly-stochastic `A`, these are related but not identical; for the current code, `A` is symmetric but **not normalised**, so the SVD-based output deviates further from the paper. Commented-out alternatives in the file (`SpectralEmbedding`, `scipy.linalg.eigh`) suggest this was experimented with. **Decide and document one canonical implementation.**

- **`RecurrenceClustering` / `RecurrenceManifold` do not actually use the fuzzy simplicial complex.** They call `data_to_connectivity` (exp-of-distance, fixed scale). The paper's adaptive ρ/σ complex (`dataset_to_simplex`, `data_to_connectivity2`) is in the codebase but unused by the default models. This is a meaningful discrepancy between the paper and what `RecurrenceManifold().fit_predict(X)` actually computes.

**Outcome (2026-05):** Both deviations closed in the canonical-
algorithm PR. `RecurrenceManifold` now uses
`scipy.linalg.eigh(L = D − A)` (MM22 oracle). `RecurrenceClustering`
and `RecurrenceManifold` both call `data_to_connectivity2`
(simplicial); `data_to_connectivity` is retained only by
`ClassicalRecurrenceClustering` as its Sauer baseline kernel. MM19
green at ARI > 0.95; MM20 deferred (Leiden under-segments period-4
at default resolution — xfail with note).

### C. Architectural / module-organisation smells

- **`models/models.py` is 914 lines** containing eight unrelated concerns: union-find data structure, Leiden multi-backend adapter, three different distance→connectivity functions, a connectivity-from-distance helper, a base estimator, three SHREC models, and one classical baseline. The single-file mixing makes targeted unit testing essentially impossible.
- **`utils/__init__.py` hand-maintains a 60-line re-export list** with another 60-line commented-out copy. New utility additions silently fail to be exposed.
- **`_find_distance_matrix` on `RecurrenceModel`** is dead — no subclass calls it, and as written it allocates an `(B, T, T, D)` temporary that blows memory.
- **Hand-rolled `DisjointSet`** duplicates `scipy.cluster.hierarchy.DisjointSet` (≥ scipy 1.6).
- **Two implementations of the same step**: `dataset_to_simplex` reimplements `fuzzy_simplicial_set` from `umap.umap_`. Keep one.
- **`_make_embedding` lives on the base class** but is a pure function of `(X, d_embed, padding, noise, random_state)`. It belongs in `utils/preprocessing.py` and should be tested there.
- **`fit` signatures diverge** across the four models — `ClassicalRecurrenceClustering.fit(X, weighted, use_sparse)`, `RecurrenceClustering.fit(X, objective, method, use_sparse)`, `RecurrenceManifold.fit(X, use_sparse, root_index)`, `HirataNomuraIsomap.fit(X)`. None of these per-call kwargs are part of the sklearn `BaseEstimator` contract — they should be `__init__` parameters so cross-validation and pipelining work.
- **`root_index` parameter** on `RecurrenceManifold.fit` is unused (the body uses it only inside commented-out lines).
- **Sklearn API leak**: `RecurrenceModel` inherits `ClusterMixin` but the manifold model produces continuous values, not cluster labels. `RecurrenceManifold` should inherit `TransformerMixin` (and expose `transform`, not `fit_predict`).

**Outcome (2026-05):** `models.py` split into per-file modules
(`base.py`, `recurrence_clustering.py`, `recurrence_manifold.py`,
`classical.py`, `hirata_nomura.py`). `_make_embedding` extracted to
`embeddings.make_embedding`. The simplicial / Leiden / union-find
pieces moved into `recurrence/` and `graph/` subpackages. `fit`
signatures normalised to `fit(X, y=None)` returning `self`; dead
kwargs (`use_sparse`, `root_index`, `weighted`) dropped. The
ClusterMixin/TransformerMixin issue is **not yet addressed** —
`RecurrenceManifold` still inherits `ClusterMixin`; tracked as
follow-up.

### D. Performance issues

- **All-pairs `cdist` matrix** is allocated densely (`O(T²)`) for every response, then again as a `(T, T)` consensus. Paper claims `O(E)` cost; the implementation is `O(T²·N)`. Pre-pruning to a sparse k-NN graph before averaging would fix this.
- **`common_neighbors_ratio`** is a pure-Python double loop with `np.intersect1d`/`np.union1d` per pair → `O(T² · k)` Python. Vectorisable with a boolean adjacency matmul (`(A @ Aᵀ) / (k_i + k_j − A @ Aᵀ)`).
- **`compress_adjacency`** rebuilds the Pearson matrix on every merge → `O(T⁴)` for `T`-node compression. Won't scale past a few hundred nodes.
- **`data_to_connectivity2`** materialises a dense `(T, T)` `coo_matrix` (defeating the sparse type) before summing — combine sparse incrementally.

**Outcome (2026-05):** `common_neighbors_ratio` vectorised (MM12 pins
parity with the loop). The other three performance issues remain
open — meaningful targets for a follow-up perf PR.

### E. Tests as they stand

`tests/test_models.py` (60 lines) covers two of the four models, with identical synthetic data (`np.sin` tiled three times), and only asserts label-vector length matches input length — not numerical correctness, not behaviour under noise, not the discrete/continuous distinction. Coverage of `utils/` is **zero**. `sys.path.insert` is used instead of relying on the installed package.

**Outcome (2026-05):** Test suite grew from 2 unittest tests to
39 pytest tests + 2 documented xfails. `pytest.importorskip` gates
optional backends. The legacy `tests/test_models.py` is preserved as
a smoke test only — all new tests live in dedicated per-module files.

---

## 4. Proposed module layout

> **Implemented as proposed (with minor simplifications).** See
> `docs/architecture.md` §4 for the actual post-refactor layout.

The principle: each file does one thing, has explicit public exports,
and is unit-testable in isolation. Naming kept conservative to
minimise downstream notebook churn — old import paths can be
preserved via shim re-exports in `models/__init__.py` and
`utils/__init__.py`.

```
src/shrec/
├── __init__.py                  # version, top-level re-exports for public API
│
├── embeddings.py                # time-delay embedding (moved out of base class)
│   ├── embed_ts(X, d, padding=None, noise=0.0, rng=None)
│   ├── hankel_matrix(data, q, p=None)
│   └── add_gaussian_noise(X, sigma, rng)
│
├── preprocessing.py             # scaling, detrending, NaN handling
│   ├── standardize_ts, minmax_ts, spherize_ts, whiten_zca
│   ├── detrend_ts, transform_stationary, unroll_phase
│   ├── nan_fill, fill_forward_back  (← fixes the "bill" typo)
│   └── lift_ts, allclose_shapes  (renamed/fixed allclose_len)
│
├── recurrence/
│   ├── __init__.py
│   ├── distance.py              # pairwise distances, masking
│   │   ├── pairwise_distances(X, metric="euclidean")
│   │   ├── exclude_time_diagonal(D, bandwidth)
│   │   └── batched_distances(X_stack)
│   ├── simplicial.py            # ONE canonical fuzzy-simplicial implementation
│   │   ├── fit_rho_sigma(d_row, k, tol)
│   │   ├── row_affinity(d_row, rho, sigma)
│   │   └── fuzzy_simplicial_complex(X, k, tol, precomputed=False)
│   ├── kernel.py                # the simpler exp(-d/scale) recurrence used today
│   │   ├── exp_recurrence(X, scale, ord=1.0, time_exclude=0)
│   │   └── distance_to_connectivity(D, dscale=None, sparsity=None)
│   ├── consensus.py             # aggregation across responses
│   │   ├── consensus_mean(per_response_A_stack)
│   │   ├── consensus_pnorm(per_response_A_stack, p=1.0)
│   │   └── consensus_min(per_response_d_stack, scale)
│   └── sparsify.py              # binarisation / pruning
│       ├── sparsify_by_quantile(A, sparsity, weighted=False)
│       └── otsu_threshold(a)
│
├── graph/
│   ├── __init__.py
│   ├── adjacency.py             # adjmat, hollow_matrix, common_neighbors_ratio (vectorised)
│   ├── communities.py           # _leiden multi-backend adapter (cleaned, with explicit `raise` on unknown method)
│   ├── unionfind.py             # thin wrapper around scipy.cluster.hierarchy.DisjointSet + solve_union_find
│   ├── connectivity.py          # largest_connected_component, susceptibility_*, compress_adjacency
│   └── associations.py          # graph_from_associations, adjmat_from_associations, graph_threshold
│
├── signal.py                    # discretize_signal, find_psd, find_characteristic_timescale, group_consecutives
├── surrogates.py                # make_surrogate, array_select
├── metrics.py                   # sparsity, evaluate_clustering, outlier_detection_pca
├── transforms.py                # RigidTransform
│
├── models/
│   ├── __init__.py              # public model surface
│   ├── base.py                  # RecurrenceEstimator base (sklearn-clean, no global random seeding)
│   ├── recurrence_manifold.py   # continuous driver — uses Laplacian Fiedler (per paper)
│   ├── recurrence_clustering.py # discrete driver — uses Leiden on consensus graph
│   ├── classical.py             # Sauer (2004) baseline (bug-fixed)
│   └── hirata_nomura.py         # HN-Isomap baseline
```

Notes:

- **Each `models/*.py` file imports building blocks from `recurrence/`, `graph/`, `embeddings.py`** — they become *compositions*, not 200-line monoliths. This is the change that makes property-based testing tractable.
- **One canonical fuzzy-simplicial implementation.** Keep `recurrence.simplicial` (the code's `dataset_to_simplex`); drop `data_to_connectivity2`'s reimplementation, or vice versa. Document the choice.
- **Random state** is plumbed via `np.random.Generator` objects, never `np.random.seed(...)`.
- **`fit` signatures** are nullary (just `X, y=None`) — all behavioural knobs live on `__init__`.
- **Public API** (`src/shrec/__init__.py`) re-exports the four models and the most-used building blocks. Notebooks keep `from shrec.models import RecurrenceManifold` working.

**Outcome (2026-05):** Implemented with two simplifications:

- `recurrence/distance.py` and `recurrence/sparsify.py` were folded
  into `recurrence/simplicial.py` / `utils/metrics.py` — they were
  too small to justify separate modules at this size.
- `utils/` was kept as a single sub-package rather than split out
  to top-level `signal.py`, `surrogates.py`, etc. Same reasoning:
  not enough mass to justify the additional surface.

---

## 5. Per-module test plan

> **Partially implemented.** The "must" tests in §5b drove the
> canonical-algorithm fix and are the live source of truth; the
> §5 structural tests below were implemented where they corresponded
> to a real bug (A1, A2, A3) and skipped where the function was
> already trivially correct. New per-module test files added during
> the refactor are: `tests/test_models_base.py`,
> `tests/test_models_classical.py`,
> `tests/test_models_recurrence_clustering.py`,
> `tests/test_models_recurrence_manifold.py`,
> `tests/test_models_invariances.py`,
> `tests/test_recurrence_simplicial.py`,
> `tests/test_graph_communities.py`,
> `tests/test_graph_adjacency.py`,
> `tests/test_benchmarks_dynamical_systems.py`.

Each test file is one-to-one with a module. I'd switch from `unittest` to **pytest** to get parametrisation, fixtures, and `pytest.mark.parametrize` for the cross-method sweeps. `pytest-cov` for coverage; `hypothesis` is optional but pays off for array/graph utilities.

### `tests/test_embeddings.py`
- `embed_ts` on a 1-D series of length T produces shape `(T, d_embed)` with `padding="symmetric"`, `(T - d_embed, d_embed)` without.
- `embed_ts` on a `(T, D)` series returns `(T, d_embed * D)` rows (verify columns are correct delays).
- `embed_ts` on a `(B, T, D)` stack returns `(B, T, d_embed * D)` — **currently broken** (`for i in X.shape[-1]:` bug), so this catches a regression.
- `hankel_matrix` matches `scipy.linalg.hankel` for a fixed example.
- `add_gaussian_noise` is reproducible given the same RNG.

### `tests/test_preprocessing.py`
- `standardize_ts` makes columnwise mean ≈ 0, std ≈ 1; zero-variance columns are passed through (not NaN'd).
- `minmax_ts` maps to [0, 1] without `clipping`; respects `clipping` quantile.
- `detrend_ts(method="global_linear")` returns zero on a pure line.
- `detrend_ts(method="global_exponential")` returns ~zero on `a * exp(b*t)`.
- `nan_fill` matches `pandas.DataFrame.fillna(method="ffill").fillna(method="bfill")` — **catches the `"bill"` typo bug**.
- `transform_stationary` returns a series that passes both ADF and KPSS at p=0.05 on a known unit-root sample (use a fixed seed).
- `allclose_shapes((3,), (3, 4))` returns False — **catches the broadcasting bug**.
- `spherize_ts` produces zero-mean output with diagonal covariance.

### `tests/test_recurrence_distance.py`
- `pairwise_distances` matches `scipy.spatial.distance.cdist` for euclidean.
- `exclude_time_diagonal` zeroes a `±bandwidth` band and nothing else.

### `tests/test_recurrence_simplicial.py`
- `fit_rho_sigma`: for fixed `d_row = [0, 0.5, 1.0]`, the solved `σ` satisfies `Σ exp[-ReLU(d − ρ)/σ] = log₂ k` within tolerance.
- `fuzzy_simplicial_complex` is symmetric (`A == A.T` after fuzzy union).
- For a periodic signal of period P, the resulting `A_ij` peaks at `|i − j| = P` (a structural property, not a numerical exact value).
- Matches `umap.umap_.fuzzy_simplicial_set` output (within tolerance) — pinpoints divergence between the two implementations the codebase currently carries.

### `tests/test_recurrence_kernel.py`
- `exp_recurrence(X, scale=1)` returns an `N×N` symmetric matrix with unit diagonal (when `time_exclude=0`).
- Larger `ord` (p-norm) → results approach the min over responses (test on a stack of two known matrices).
- `distance_to_connectivity(D, sparsity=0.9)` returns a matrix with the requested sparsity (± one element).

### `tests/test_recurrence_consensus.py`
- `consensus_mean` of identical responses is idempotent.
- `consensus_pnorm` with `p=∞` equals the elementwise max (and `p=1` the mean).
- Skipping constant time series: stacking one constant and one varying response yields the same result as the varying one alone (this is the `np.isclose(X, X[:, :1, :])` filter currently in `data_to_connectivity`).

### `tests/test_recurrence_sparsify.py`
- `sparsify_by_quantile(A, sparsity=0.9)` produces a matrix with ≥ 90% zeros.
- `otsu_threshold` on a bimodal distribution returns a value between the two modes.

### `tests/test_graph_adjacency.py`
- `hollow_matrix` zeros the diagonal and is idempotent.
- `common_neighbors_ratio` on a 3-clique returns the all-ones-off-diagonal matrix (after the `1 − ratio` flip).
- Vectorised implementation matches the current loop-based one on random binary matrices of size ≤ 20.

### `tests/test_graph_communities.py`
- `_leiden` on a barbell graph returns 2 communities for each of the available backends (parametrised). Skip backends that fail to import.
- Unknown method raises `ValueError` (— current code silently returns garbage).
- Random seed produces reproducible labels for the `graspologic` backend.

### `tests/test_graph_unionfind.py`
- `solve_union_find([[0,1], [2,3], [1,2]])` returns one group `{0,1,2,3}`.
- Wraps `scipy.cluster.hierarchy.DisjointSet` correctly (parity with the current hand-rolled one).

### `tests/test_graph_associations.py`
- `adjmat_from_associations` on `[[0,1,1],[1,1,0],[1,0,1]]` returns the all-ones (off-diagonal) matrix (matches docstring example).
- `weighted=True` reflects clique multiplicity.
- Sparse and dense paths return matrices equal under `.toarray()`.

### `tests/test_graph_connectivity.py`
- `largest_connected_component` on a disjoint union of `K_3` and `K_5` returns 5/8.
- `compress_adjacency` to `n_target = N` is a no-op; to `n_target = 1` collapses to a single node.

### `tests/test_signal.py`
- `discretize_signal` on a uniform input returns roughly equal-count bins.
- `find_psd` on `sin(2π·5·t)` peaks at frequency ≈ 5/T.
- `find_characteristic_timescale` recovers the period of a known sinusoid to within one bin.

### `tests/test_surrogates.py`
- `make_surrogate(X, method="random_phase")` preserves the power spectrum (within 1% on a long Gaussian process).
- `random_state` makes the output reproducible.

### `tests/test_metrics.py`
- `sparsity` matches the obvious formula on a sparse CSR.
- `evaluate_clustering(y, y)` returns 1.0 for every metric (perfect agreement).
- `evaluate_clustering` of a random shuffle returns ≈ 0 ARI.

### `tests/test_transforms.py`
- `RigidTransform.fit_transform` aligns two copies of the same point cloud under a random rotation (recovered rotation matches within tol).
- `inverse_transform` undoes `transform`.

### `tests/test_models_base.py`
- Constructing a `RecurrenceEstimator` does **not** call `np.random.seed` globally (assert global state unchanged across construction).
- All public hyperparameters appear in `__init__` (sklearn `get_params` round-trip).

### `tests/test_models_recurrence_manifold.py`
- Lorenz-driven-by-Rössler synthetic (use the existing `benchmarks/data/lorenz_data.npy`, ≤ 500 timepoints) — Spearman correlation with the true driver above a regression threshold (e.g. ρ > 0.7). This is the **only** numerical-accuracy regression test we need.
- On the trivial tiled-sine input, output has the right length (matches today's test).
- Default `RecurrenceManifold()` matches the paper's continuous-driver pipeline: the affinity construction is the fuzzy simplex, and the embedding is the Fiedler vector. (Will fail given today's code — useful to drive the fix.)

### `tests/test_models_recurrence_clustering.py`
- On a square-wave driver with two clearly-separated levels, recovers exactly 2 clusters; ARI vs ground truth > 0.9.
- Leiden backend choice (`method=`) does not change the cluster count for a textbook input.

### `tests/test_models_classical.py`
- Confirms the **`use_sparse=False` TypeError bug** with a smoke test (`pytest.raises(TypeError)`), then verifies the same test passes after the fix.
- Recovers exact equivalence classes on a noise-free 2-symbol periodic signal (the regime where Sauer's algorithm is exact).

### `tests/test_models_hirata_nomura.py`
- Smoke test: runs end-to-end on the tiled-sine input and produces a label vector of the right length.
- 1-component embedding of a periodic signal is monotone with the true phase to within sign.

### `tests/conftest.py` (new)
- Fixtures:
  - `lorenz_data` — loads from `benchmarks/data/`, cached at session scope.
  - `tiled_sine(n_series, T)` — parameter factory replacing today's copy-paste data.
  - `square_wave_driver(T, n_levels)` — for the clustering regression test.
  - `rng` — `numpy.random.default_rng(seed)` fixture.

### Test infrastructure changes
- Drop `sys.path.insert` in `tests/test_models.py` — install with `pip install -e .` and import as a normal package.
- Add `pyproject.toml` test config:
  ```toml
  [tool.pytest.ini_options]
  testpaths = ["tests"]
  addopts = "-ra --strict-markers"
  ```
- Add a `[project.optional-dependencies] dev = ["pytest", "pytest-cov", "hypothesis"]`.
- A handful of tests need optional backends (`graspologic`, `leidenalg`, `cdlib`, `umap-learn`) — gate them with `pytest.importorskip` so the core test suite runs without them.

---

## 6. Suggested order of work

> **All five steps completed by 2026-05-23.** Per-step outcomes are
> in CLAUDE.md "Current state".

1. **Land the bug-fix PR** (no behaviour changes): items A1, A2, A3, A4, A5, A7, A8, A9. Add regression tests for each.
2. **Decide and document the canonical pipeline** for `RecurrenceManifold` (Laplacian Fiedler vs SVD) and `RecurrenceClustering` (fuzzy simplex vs exp-kernel). Section B above flags both — they're behavioural, not bug fixes, so they deserve a separate PR with a regression check on the benchmark datasets.
3. **Modularise utils** first (§4) — it's the lowest-risk change; each function is pure. Land with the per-utility test files (§5).
4. **Modularise models** — extract `_make_embedding` into `embeddings.py`, extract `_leiden` into `graph/communities.py`, then split the four model classes into four files. Old import paths kept alive via shims in `models/__init__.py`.
5. **Fix performance hotspots** (`common_neighbors_ratio`, `compress_adjacency`, sparse path through `data_to_connectivity*`) — separate PR, with `pytest-benchmark` markers if you want regression guards.

The §5b math-correctness layer rearranged step 2 in practice:

1. **Bug-fix PR** (§3A + §5 regression tests). No behavioural change.
2. **Canonical-algorithm PR**: add **MM5, MM19, MM20, MM22** *before* changing any model code. These three tests *will fail* against today's `models.py`. Use the failing tests to drive a single coherent fix that aligns the code with the paper. Lock in the green state.
3. **Modularisation PR** (§4) — now safe, because the math-correctness suite catches anything that drifts.
4. **Invariance / contract PRs** — MM13, MM14, MM18, MM30–32 enforce behaviours that should never have been allowed to drift; add them once and treat any failure as a real regression.
5. **Performance PR** (§3D) — vectorise `common_neighbors_ratio` and `compress_adjacency` only after MM12 / MM10 pin behaviour.

The "must" tests in §5b.1–§5b.3 (MM1–MM6, MM10, MM13–MM14, MM19–MM22, MM27) are the minimum I would want green before any refactor of `models.py`. They cover: (a) the inner math of the simplicial complex (MM1–MM5), (b) the consensus aggregation (MM6), (c) community-detection robustness (MM10), (d) algorithmic invariances that *cannot* be wrong (MM13–MM14), (e) the Sauer limit on which the paper's claim of exactness rests (MM19–MM20), (f) the Fiedler/Laplacian claim for the continuous model (MM22), and (g) the β-scaling that validates the method beyond a single working point (MM27).

---

## 7. Notes on this assessment

I could not actually run the code in this environment — the `.venv` is built for Python 3.11 but the host only has Python 3.12. The audit above is static; the bug list (§3A) is high-confidence (the typos and signature mismatches are mechanical), but the regression-test expectations in §5 should be calibrated against actual runs once the environment is set up.

Sources used for the algorithm:

- Gilpin, *Recurrences Reveal Shared Causal Drivers of Complex Time Series*, Phys. Rev. X 15, 011005 (2025). DOI: 10.1103/PhysRevX.15.011005.
- arXiv:2301.13516 — same paper, open-access preprint.
- The repo `README.md`, `GEMINI.md`, and `pyproject.toml`.
