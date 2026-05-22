# shrec — code assessment and modularization plan

Working from `src/shrec/` (≈ 60 KB of Python across `models/models.py` and seven `utils/*.py` files) and the paper *Recurrences Reveal Shared Causal Drivers of Complex Time Series*, Gilpin, **Phys. Rev. X 15, 011005 (2025)** (arXiv:2301.13516).

---

## 1. Paper-level summary (what the code must implement)

SHREC reconstructs an unobserved driver `z(t)` from `N` observed responses `x_k(t)`. The pipeline (paper Appendix B):

1. **Time-delay embed** each response: `ŷ_k(t) = [x_k(t), x_k(t-τ), …, x_k(t-(D-1)τ)]ᵀ` ∈ ℝ^(T×D).
2. **Per-response pairwise distances**: `d^(k)_ij = ‖ŷ_k(i) − ŷ_k(j)‖`.
3. **Per-response fuzzy simplicial complex** (adaptive recurrence):
   - For each row `i`, take the `M` nearest neighbours; set `ρ_i = min_m d_im`.
   - Numerically solve for `σ_i` such that `log₂ M = Σ_m exp[-ReLU(d_im − ρ_i)/σ_i]`.
   - Affinity `A^(k)_ij = exp[-ReLU(d^(k)_ij − ρ_i)/σ_i]`. Symmetrise via fuzzy union `A + Aᵀ − A∘Aᵀ`.
4. **Consensus aggregation** across responses: `A_ij = (1/K) Σ_k A^(k)_ij` (paper uses mean; the code generalises to a `p`-norm via `aggregation_order`).
5. **Driver reconstruction**:
   - **Discrete driver** → Leiden community detection on `A`; cluster labels are the driver symbols.
   - **Continuous driver** → graph Laplacian `L = D − A`; the **Fiedler eigenvector** (first non-constant eigenvector) is the reconstructed driver.

The paper also reports two baselines that live in this repo:

- `ClassicalRecurrenceClustering` — Sauer (PRL 2004) exact equivalence-class union-find on binary recurrences (the noise-free limit case).
- `HirataNomuraIsomap` — Hirata/Nomura: common-neighbour-ratio weighting + Isomap embedding.

---

## 2. Code → paper mapping

| Paper step                              | Where in `src/shrec/`                                                                                  |
|-----------------------------------------|--------------------------------------------------------------------------------------------------------|
| Time-delay embedding                    | `utils/preprocessing.py:embed_ts` / `hankel_matrix`; `models.RecurrenceModel._make_embedding`          |
| Pairwise distance                       | `cdist(...)` inside `models.data_to_connectivity*`; unused `RecurrenceModel._find_distance_matrix`     |
| ρ/σ root-find + simplicial complex      | **Two implementations**: `models.dataset_to_simplex` (custom `fsolve`) and `models.data_to_connectivity2` (calls `umap.umap_.fuzzy_simplicial_set`) |
| Consensus / p-norm aggregation          | `models.data_to_connectivity` (exp-of-distance, not the fuzzy simplex), `data_to_connectivity2`        |
| Discrete driver (Leiden)                | `models._leiden` (wraps graspologic / leidenalg / igraph / cdlib) → `RecurrenceClustering`             |
| Continuous driver (Fiedler)             | `RecurrenceManifold.fit` — **but** uses `TruncatedSVD` on `A`, not Laplacian eigh. See §3.             |
| Sauer baseline                          | `models.ClassicalRecurrenceClustering`, `models.solve_union_find`, `models.DisjointSet`                |
| Hirata–Nomura baseline                  | `models.HirataNomuraIsomap` + `utils/graph_tools.common_neighbors_ratio`                               |

The fuzzy-simplicial helpers `dataset_to_simplex` and `data_to_connectivity2` exist but are not wired into any production path — `RecurrenceManifold`/`RecurrenceClustering` call `data_to_connectivity`, which is the *exp(-d/scale)* recurrence variant, not the paper's adaptive ρ/σ construction. Worth deciding whether this is intentional.

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

### B. Algorithmic deviation from the paper

- **`RecurrenceManifold` uses TruncatedSVD on `A`**, then takes the second right singular vector (line 904–906). The paper specifies the Fiedler eigenvector of the **graph Laplacian** `L = D − A`. For an exactly symmetric, doubly-stochastic `A`, these are related but not identical; for the current code, `A` is symmetric but **not normalised**, so the SVD-based output deviates further from the paper. Commented-out alternatives in the file (`SpectralEmbedding`, `scipy.linalg.eigh`) suggest this was experimented with. **Decide and document one canonical implementation.**

- **`RecurrenceClustering` / `RecurrenceManifold` do not actually use the fuzzy simplicial complex.** They call `data_to_connectivity` (exp-of-distance, fixed scale). The paper's adaptive ρ/σ complex (`dataset_to_simplex`, `data_to_connectivity2`) is in the codebase but unused by the default models. This is a meaningful discrepancy between the paper and what `RecurrenceManifold().fit_predict(X)` actually computes.

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

### D. Performance issues

- **All-pairs `cdist` matrix** is allocated densely (`O(T²)`) for every response, then again as a `(T, T)` consensus. Paper claims `O(E)` cost; the implementation is `O(T²·N)`. Pre-pruning to a sparse k-NN graph before averaging would fix this.
- **`common_neighbors_ratio`** is a pure-Python double loop with `np.intersect1d`/`np.union1d` per pair → `O(T² · k)` Python. Vectorisable with a boolean adjacency matmul (`(A @ Aᵀ) / (k_i + k_j − A @ Aᵀ)`).
- **`compress_adjacency`** rebuilds the Pearson matrix on every merge → `O(T⁴)` for `T`-node compression. Won't scale past a few hundred nodes.
- **`data_to_connectivity2`** materialises a dense `(T, T)` `coo_matrix` (defeating the sparse type) before summing — combine sparse incrementally.

### E. Tests as they stand

`tests/test_models.py` (60 lines) covers two of the four models, with identical synthetic data (`np.sin` tiled three times), and only asserts label-vector length matches input length — not numerical correctness, not behaviour under noise, not the discrete/continuous distinction. Coverage of `utils/` is **zero**. `sys.path.insert` is used instead of relying on the installed package.

---

## 4. Proposed module layout

The principle: each file does one thing, has explicit public exports, and is unit-testable in isolation. Naming kept conservative to minimise downstream notebook churn — old import paths can be preserved via shim re-exports in `models/__init__.py` and `utils/__init__.py`.

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

---

## 5. Per-module test plan

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

## 5b. Math-correctness tests (the part that proves the algorithm is right, not just present)

The tests in §5 catch *structural* bugs — typos, signature mismatches, shape errors. They cannot tell you whether SHREC is computing what the paper says it computes. Below is a second layer of tests whose oracles come from analytical results, limiting cases, and explicit claims in the paper. These are what would let you ship a refactor with confidence.

Notation: "must-have" = catches a class of bugs that §5 cannot. "regression" = guards a property over time but won't fail on a fresh implementation if math is right.

### §5b.1 — Closed-form unit tests of the inner math

These don't need a time series at all; they probe individual operators with inputs whose answers can be written down.

| Test | Module | Oracle |
|---|---|---|
| **MM1 (must)** | `recurrence/simplicial.py:fit_rho_sigma` | The defining equation is satisfied: for arbitrary `d_row` (random, sorted), the returned `(ρ, σ)` give `\|Σ_m exp[-ReLU(d_im − ρ)/σ] − log₂ k\| < 1e-6`. |
| **MM2 (must)** | same | **Scale invariance**: for any `α > 0`, `fit_rho_sigma(α · d_row, k) = (α·ρ, α·σ)`, so the final affinity `A^(k)_ij(α·d) ≡ A^(k)_ij(d)`. This is the strongest single algebraic property of the simplicial complex; if anyone ever changes the σ root-find, this test catches it. |
| **MM3 (must)** | same | **Diagonal**: `A^(k)_ii = exp(-ReLU(0 − ρ_i)/σ_i) = 1` exactly, because `d_ii = 0 ≤ ρ_i` (nearest-neighbour distance is the *smallest non-zero* distance, but the row distance to self is 0). Verify after `hollow_matrix` is *not* applied. |
| **MM4 (must)** | `fuzzy_simplicial_complex` | **Symmetrisation range**: `A + Aᵀ − A∘Aᵀ ∈ [0, 1]^(N×N)` for any `A ∈ [0, 1]^(N×N)`. Symmetric output: `‖S − Sᵀ‖_∞ = 0`. |
| **MM5 (must)** | `recurrence/simplicial.py` vs `data_to_connectivity2` | **Cross-implementation parity**: `dataset_to_simplex(X, k=M)` and `umap.umap_.fuzzy_simplicial_set(X, M, ...)` produce affinity matrices that agree to within `atol=1e-5` on a random `(50, 3)` input. The repo carries two implementations of the *same* paper formula; if they disagree, one is wrong. This single test triages a question that the §3B audit raised but couldn't resolve statically. |
| **MM6 (must)** | `recurrence/kernel.py:exp_recurrence` | **p-norm limits**: with `ord=1` the consensus is the elementwise mean of `exp(-d^(k)/scale)` over `k`; with `ord → ∞` it approaches the *min over k* of `d^(k)` (Sauer's `d*_ij = inf_k d^(k)_ij`). Verify both with a stack of two analytically chosen distance matrices. Currently `ClassicalRecurrenceClustering` relies on this with `ord=500`; if the formula drifts, the Sauer baseline silently degrades. |
| MM7 | `pairwise_distances` | **Isometry invariance**: for random orthogonal `Q ∈ ℝ^(D×D)` and translation `c ∈ ℝ^D`, `cdist(X@Q + c, X@Q + c) == cdist(X, X)`. |
| MM8 | `pairwise_distances` | **Triangle inequality** holds on every triple `(i, j, k)` of a `cdist` matrix on random points (statistical sanity check). |
| MM9 | `sparsify_by_quantile` | Output sparsity ≥ target sparsity (with strict equality if no ties in `|a|`). Idempotent under repeated application at the same threshold. |
| **MM10 (must)** | `graph/communities.py:_leiden` | **Backend agreement**: on the barbell graph `K₅ — K₅` (two cliques joined by a bridge), every available Leiden backend returns exactly 2 communities with `adjusted_rand_score(labels_a, labels_b) == 1.0` pairwise. Paper §II.B says "performance is insensitive to the choice of community detection algorithm" — make that explicit. |
| MM11 | `graph/unionfind.py` | **Parity with `scipy.cluster.hierarchy.DisjointSet`** on random union sequences (Hypothesis-style: any sequence of `(a, b)` pairs). |
| MM12 | `graph/adjacency.py:common_neighbors_ratio` | **Closed-form values**: on `K_n` (complete graph), output is zero off-diagonal. On the disjoint union `K_a ⊔ K_b`, the inter-cluster value is exactly 1. After vectorising (see §3D), the new implementation matches the old on random binary matrices to `atol=0`. |

### §5b.2 — Algorithmic invariances and equivariances

These are properties the *whole* `fit_predict` pipeline must satisfy on principled grounds (recurrence is undirected; the model has no preferred response order; etc.). Each is a one-line test against a baseline run.

| Test | Property | Why |
|---|---|---|
| **MM13 (must)** | `model.fit_predict(X)` is **permutation-invariant in the response axis** (`X[:, perm]` → same labels up to label permutation, ARI=1). | A response is a response — order is not data. If a future refactor introduces order dependence, this catches it immediately. |
| **MM14 (must)** | **Time-reversal symmetry** for the recurrence-graph stage: `A(X[::-1]) == A(X)[::-1, ::-1]` exactly (before community detection). For `RecurrenceClustering`, `fit_predict(X[::-1])` and `fit_predict(X)[::-1]` agree up to label permutation (ARI=1). | The fuzzy simplicial complex only sees distances, not time direction. Asymmetric output ⟹ bug. |
| MM15 | **Scale invariance** of the whole pipeline w.r.t. each response: scaling `X[:, k]` by an arbitrary positive constant doesn't change the labels (because the simplicial complex is scale-invariant per MM2, and standardisation is on by default). | Cross-checks that pre-processing doesn't leak scale into the answer. |
| MM16 | **Idempotence on a single repeated response**: if all `N` responses are copies of one signal, `fit_predict` matches `fit_predict` of just that one signal (up to label permutation). | The mean-aggregation step must not amplify or attenuate identical signals. |
| MM17 | **Constant-response rejection**: a response that is constant in time is skipped (current code has this filter at `data_to_connectivity:303`). Stacking a constant response on a varying one yields the same answer as the varying one alone. | Defensive — pin the existing behaviour so a refactor doesn't accidentally include constants and produce NaNs. |
| MM18 | **Determinism w.r.t. `random_state`**: two models with the same seed produce bit-identical outputs; two models with different seeds produce different outputs on a problem where the answer is seed-sensitive (small N, noisy). | Catches the `np.random.seed` global-state bug in §3A item 6 once we move to `default_rng`. |

### §5b.3 — Limiting-case oracles (closed-form ground truth)

These are inputs constructed so the answer is provably the one we want.

| Test | Input | Oracle |
|---|---|---|
| **MM19 (must)** | **Sauer limit, period-2 driver**. Drive `N=20` distinct chaotic logistic maps (random `r^(k) ~ U(3.81, 3.97)`) with `z_t = (-1)^t`, `κ = 0.5`, zero noise, `T = 1000`. | `RecurrenceClustering().fit_predict(X)` returns exactly 2 communities with `adjusted_rand_score(labels_pred, z) == 1.0`. **Paper §II.B, line 461 and §III.D, line 570 both claim SHREC reduces to exact equivalence classes in this limit — this test makes the claim falsifiable.** |
| **MM20 (must)** | **Sauer limit, period-4 driver**. As MM19 but `z_t` cycles through 4 distinct levels. | Exactly 4 communities, ARI = 1. |
| MM21 | **Period-8 driver, with stochastic forcing** (`σ_noise = 0.04`). | ARI > 0.85 — this is no longer a hard claim but the paper's Fig. 6(a) shows the period-8 case still hits the upper plateau at this noise level. Regression test. |
| **MM22 (must)** | **Block-stochastic affinity matrix**. Skip the time series entirely: hand-construct `A = block_diag(J_p / p, J_p / p, J_p / p)` with `K = 3` blocks of size `p = 20`. | The **Fiedler eigenvector of `L = D − A`** is constant within each block and takes 3 distinct values whose signs partition the blocks. Compare `RecurrenceManifold`'s output (after `fit` on the precomputed affinity) to this analytical Fiedler vector via `|corr| > 0.99`. **This is the test that exposes the SVD-vs-Laplacian deviation flagged in §3B; the current code will fail it. Use it to drive the fix and lock in the fixed behaviour.** |
| MM23 | **Cycle graph affinity**. `A_ij = 1 if |i−j|=1 (mod n) else 0`. | Laplacian eigenvalues are `2(1 − cos(2π k/n))`; the Fiedler eigenvector is `v_i = cos(2π i / n)` (or `sin`). `RecurrenceManifold`'s output equals this up to sign with `|corr| > 0.999`. Pure closed-form check of the spectral step. |
| MM24 | **Identity-driver test**. `N = 1`, `x(t) = z(t)` where `z` is the Rössler `z₁` trajectory (continuous chaotic driver). | Spearman correlation `|ρ| > 0.95`. The synchronised regime with no measurement filter — should be the easiest possible case. |
| MM25 | **Linear measurement test**. `x_k(t) = a_k · z(t) + b_k` with random `a_k > 0, b_k`. | `|ρ| > 0.9` despite per-channel affine corruption (because standardisation removes `a_k, b_k`). |
| MM26 | **Nonlinear monotone measurement**. `x_k(t) = tanh(z(t)/σ_k)` with random `σ_k`. | `|ρ| > 0.8`. Tests that the algorithm handles channel-wise *monotone but non-affine* response filters. |

### §5b.4 — Paper-claim regression tests (quantitative scaling laws)

Two tests that turn paper figures into pass/fail checks. They are slower (involve sweeps) and so should run only under `-m slow` or in CI nightly.

| Test | Claim (paper section) | Oracle |
|---|---|---|
| **MM27 (must)** | **β-distribution accuracy scaling**, Appendix E.2 line 1751: `Acc(NT/τ) = Acc_max (1 − exp(−β √(NT/τ)))`. | For the period-4 logistic ensemble at default `σ_noise = 0.04, κ = 0.5`, run `N ∈ {2, 4, 8, 16, 32, 64}` and `T = 3000`. Fit the β-distribution form to the observed ARI vs N. Require `R² > 0.9` and `Acc_max > 0.9`. If you can ever break this curve, the algorithm has regressed in a way that single-point tests will not catch. |
| MM28 | **Percolation order parameter**, Appendix E.3 line 1771: `T_LCC/T` should show a first-order transition as `N` increases at fixed noise. | Sweep `N ∈ {2, 4, 8, 16, 32, 64}` on the period-2 logistic ensemble. Record `T_LCC/T` of the binarised consensus graph (after sparsification). Assert `T_LCC/T` is monotone non-increasing in `N` and exhibits a drop of at least 0.3 between two consecutive `N` values. Coarse but catches loss of the "glassy" recurrence structure that gives the method its name. |
| MM29 | **HN-Isomap baseline produces a valid distance matrix**, used implicitly throughout §III.A. | After `common_neighbors_ratio` computation in `HirataNomuraIsomap`, verify: symmetric, zero-diagonal, non-negative — i.e. a valid pseudo-distance for Isomap's MDS step. Catches a class of bugs where the ratio drifts below zero or symmetry is lost. |

### §5b.5 — Sklearn-contract tests

Cheap, mechanical, but worth pinning down.

- **MM30** — `sklearn.utils.estimator_checks.check_estimator` (subset; the unsupervised time-series API has known mismatches, so use `check_no_attributes_set_in_init` + `check_get_params_invariance` rather than the full battery).
- **MM31** — Constructing a `RecurrenceManifold()` does **not** change `np.random.get_state()` — pins the §3A item 6 fix.
- **MM32** — `set_params(**get_params())` is the identity on the model state.

### How §5b changes the prioritisation in §6

The math-correctness layer rearranges step 2 of §6. Suggested revision:

1. **Bug-fix PR** (§3A + §5 regression tests). No behavioural change.
2. **Canonical-algorithm PR**: add **MM5, MM19, MM20, MM22** *before* changing any model code. These three tests *will fail* against today's `models.py` (MM22 because of SVD-vs-Laplacian; MM19/20 because the default uses the kernel recurrence not the simplicial complex; MM5 because the two simplicial impls have not been compared). Use the failing tests to drive a single coherent fix that aligns the code with the paper. Lock in the green state.
3. **Modularisation PR** (§4) — now safe, because the math-correctness suite catches anything that drifts.
4. **Invariance / contract PRs** — MM13, MM14, MM18, MM30–32 enforce behaviours that should never have been allowed to drift; add them once and treat any failure as a real regression.
5. **Performance PR** (§3D) — vectorise `common_neighbors_ratio` and `compress_adjacency` only after MM12 / MM10 pin behaviour.

The "must" tests in §5b.1–§5b.3 (MM1–MM6, MM10, MM13–MM14, MM19–MM22, MM27) are the minimum I would want green before any refactor of `models.py`. They cover: (a) the inner math of the simplicial complex (MM1–MM5), (b) the consensus aggregation (MM6), (c) community-detection robustness (MM10), (d) algorithmic invariances that *cannot* be wrong (MM13–MM14), (e) the Sauer limit on which the paper's claim of exactness rests (MM19–MM20), (f) the Fiedler/Laplacian claim for the continuous model (MM22), and (g) the β-scaling that validates the method beyond a single working point (MM27).

---

## 6. Suggested order of work

1. **Land the bug-fix PR** (no behaviour changes): items A1, A2, A3, A4, A5, A7, A8, A9. Add regression tests for each.
2. **Decide and document the canonical pipeline** for `RecurrenceManifold` (Laplacian Fiedler vs SVD) and `RecurrenceClustering` (fuzzy simplex vs exp-kernel). Section B above flags both — they're behavioural, not bug fixes, so they deserve a separate PR with a regression check on the benchmark datasets.
3. **Modularise utils** first (§4) — it's the lowest-risk change; each function is pure. Land with the per-utility test files (§5).
4. **Modularise models** — extract `_make_embedding` into `embeddings.py`, extract `_leiden` into `graph/communities.py`, then split the four model classes into four files. Old import paths kept alive via shims in `models/__init__.py`.
5. **Fix performance hotspots** (`common_neighbors_ratio`, `compress_adjacency`, sparse path through `data_to_connectivity*`) — separate PR, with `pytest-benchmark` markers if you want regression guards.

---

## 7. Notes on this assessment

I could not actually run the code in this environment — the `.venv` is built for Python 3.11 but the host only has Python 3.12. The audit above is static; the bug list (§3A) is high-confidence (the typos and signature mismatches are mechanical), but the regression-test expectations in §5 should be calibrated against actual runs once the environment is set up.

Sources used for the algorithm:

- Gilpin, *Recurrences Reveal Shared Causal Drivers of Complex Time Series*, Phys. Rev. X 15, 011005 (2025). DOI: 10.1103/PhysRevX.15.011005.
- arXiv:2301.13516 — same paper, open-access preprint.
- The repo `README.md`, `GEMINI.md`, and `pyproject.toml`.
