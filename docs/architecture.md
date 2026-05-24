# shrec — architecture reference

Companion code to Gilpin, *Recurrences Reveal Shared Causal Drivers of
Complex Time Series*, **Phys. Rev. X 15, 011005 (2025)**
(arXiv:2301.13516). The local PDF is `docs/PhysRevX.15.011005.pdf`.

This file is the canonical map from the paper's algorithm to the
in-repo modules. For the math-correctness test catalog see
`docs/tests-math.md`; for the historical record of the May 2026
refactor that produced this layout, see
`docs/history/2026-05-refactor-assessment.md`.

---

## 1. Paper-level summary (what the code implements)

SHREC reconstructs an unobserved driver `z(t)` from `N` observed
responses `x_k(t)`. The pipeline (paper Appendix B):

1. **Time-delay embed** each response:
   `ŷ_k(t) = [x_k(t), x_k(t-τ), …, x_k(t-(D-1)τ)]ᵀ ∈ ℝ^(T×D)`.
2. **Per-response pairwise distances**:
   `d^(k)_ij = ‖ŷ_k(i) − ŷ_k(j)‖`.
3. **Per-response fuzzy simplicial complex** (adaptive recurrence):
   - For each row `i`, take the `M` nearest neighbours; set
     `ρ_i = min_m d_im`.
   - Numerically solve for `σ_i` such that
     `log₂ M = Σ_m exp[-ReLU(d_im − ρ_i)/σ_i]`.
   - Affinity `A^(k)_ij = exp[-ReLU(d^(k)_ij − ρ_i)/σ_i]`.
     Symmetrise via fuzzy union `A + Aᵀ − A∘Aᵀ`.
4. **Consensus aggregation** across responses:
   `A_ij = (1/K) Σ_k A^(k)_ij` (paper uses mean; the code generalises
   to a `p`-norm via `aggregation_order`).
5. **Driver reconstruction**:
   - **Discrete driver** → Leiden community detection on `A`; cluster
     labels are the driver symbols.
   - **Continuous driver** → graph Laplacian `L = D − A`; the
     **Fiedler eigenvector** (first non-constant eigenvector) is the
     reconstructed driver.

Two baselines also live in the repo:

- `ClassicalRecurrenceClustering` — Sauer (PRL 2004) exact
  equivalence-class union-find on binary recurrences (the noise-free
  limit case).
- `HirataNomuraIsomap` — Hirata/Nomura: common-neighbour-ratio
  weighting + Isomap embedding.

---

## 2. Code → paper mapping

| Paper step                              | Where in `src/shrec/`                                                                          |
|-----------------------------------------|------------------------------------------------------------------------------------------------|
| Time-delay embedding                    | `embeddings.py:embed_ts` / `hankel_matrix`; `models/base.py:RecurrenceModel._make_embedding`    |
| Pairwise distance                       | `cdist(...)` inside `recurrence/simplicial.py` and `recurrence/kernel.py`                       |
| ρ/σ root-find + simplicial complex      | `recurrence/simplicial.py:dataset_to_simplex` (canonical, self-contained `fsolve` impl)         |
| Consensus aggregation                   | `recurrence/consensus.py:data_to_connectivity2` (mean over per-response simplicial outputs)     |
| Exp-kernel recurrence (baseline only)   | `recurrence/kernel.py:data_to_connectivity` — used by `ClassicalRecurrenceClustering` only      |
| Discrete driver (Leiden)                | `graph/communities.py:_leiden` → `models/recurrence_clustering.py:RecurrenceClustering`          |
| Continuous driver (Fiedler)             | `models/recurrence_manifold.py:RecurrenceManifold` — `scipy.linalg.eigh` of `L = D − A`         |
| Sauer baseline                          | `models/classical.py:ClassicalRecurrenceClustering`, `graph/unionfind.py`                       |
| Hirata–Nomura baseline                  | `models/hirata_nomura.py` + `utils/graph_tools.common_neighbors_ratio` (vectorised matmul)      |

Top-level public API (`from shrec import …`):

- `RecurrenceClustering`, `RecurrenceManifold`,
  `ClassicalRecurrenceClustering`, `HirataNomuraIsomap`
- `embed_ts`, `hankel_matrix`, `make_embedding`

Sub-packages still available for advanced use:

- `shrec.recurrence` — `simplicial`, `kernel`, `consensus` primitives
- `shrec.graph` — `_leiden`, `DisjointSet`, `solve_union_find`
- `shrec.embeddings` — `embed_ts`, `hankel_matrix`, `make_embedding`
- `shrec.utils` — preprocessing, signal, surrogates, metrics,
  transforms (legacy organisation, preserved via shims)

---

## 3. Design decisions

These are the canonical choices fixed by the 2026-05 refactor.
See `docs/history/2026-05-refactor-assessment.md` for the reasoning
trail.

- **Fiedler eigenvector over TruncatedSVD.** `RecurrenceManifold`
  computes the smallest non-zero eigenvector of `L = D − A` via
  `scipy.linalg.eigh`. The earlier `TruncatedSVD(A).components_[1]`
  only coincides with the Fiedler vector for regular graphs; oracle
  MM22 pins the canonical choice.
- **Simplicial complex over exp kernel.** `RecurrenceClustering` and
  `RecurrenceManifold` build their consensus affinity via the
  per-response adaptive fuzzy simplicial complex
  (`data_to_connectivity2`). The fixed-σ exp-kernel
  (`data_to_connectivity`) is retained for the Sauer baseline only.
  Oracles MM19, MM20 motivate the switch.
- **One simplicial implementation.** `dataset_to_simplex` is the
  canonical implementation; the umap-based variant in the old
  `data_to_connectivity2` was dropped and `umap-learn` is no longer a
  core dependency. Reconciling the two σ-solver conventions (MM5) is
  an open follow-up.
- **`np.random.Generator`, not `np.random.seed`.** No model constructor
  or `fit` ever mutates the global numpy RNG. Noise injection uses a
  local `np.random.default_rng(self.random_state)`. MM18 pins this.
- **`fit(X, y=None)` returning `self`.** All four models follow the
  sklearn contract: behavioural knobs live on `__init__`,
  `fit` is nullary, and returns `self` for chaining. MM30 pins
  `get_params`/`set_params` round-trip.
- **Vectorised `common_neighbors_ratio`.** Replaces an O(T²·k)
  Python double-loop with the boolean-adjacency matmul
  `(A @ Aᵀ) / (k_i + k_j − A @ Aᵀ)`. MM12 pins parity with the
  legacy implementation.

---

## 4. Sub-package layout (current)

```
src/shrec/
├── __init__.py             # version + top-level public API
├── embeddings.py           # embed_ts, hankel_matrix, make_embedding
├── recurrence/
│   ├── __init__.py
│   ├── simplicial.py       # dataset_to_simplex, relu (canonical)
│   ├── kernel.py           # data_to_connectivity (Sauer baseline)
│   └── consensus.py        # data_to_connectivity2 (mean aggregation)
├── graph/
│   ├── __init__.py
│   ├── communities.py      # _leiden multi-backend adapter
│   └── unionfind.py        # DisjointSet, solve_union_find
├── models/
│   ├── __init__.py         # public model surface
│   ├── models.py           # back-compat re-export shim
│   ├── base.py             # RecurrenceModel (sklearn-clean)
│   ├── recurrence_clustering.py
│   ├── recurrence_manifold.py
│   ├── classical.py
│   └── hirata_nomura.py
└── utils/                  # legacy organisation, preserved via shims
    ├── __init__.py
    ├── preprocessing.py    # standardise, detrend, NaN handling
    ├── graph_tools.py      # adjacency, association, connectivity
    ├── signal.py           # discretise, find_psd, group_consecutives
    ├── surrogates.py
    ├── metrics.py          # sparsify, otsu_threshold, evaluate_*
    ├── transforms.py       # RigidTransform
    └── io.py
```

---

## 5. Where to add new behaviour

- **A new model**: add `models/<name>.py` subclassing
  `RecurrenceModel` (composition of `recurrence/`, `graph/`,
  `embeddings.py`). Re-export from `models/__init__.py` and, if
  user-facing, from `shrec/__init__.py`.
- **A new recurrence primitive**: prefer adding to
  `recurrence/{simplicial,kernel,consensus}.py`. New consensus
  schemes go in `consensus.py`.
- **A new community-detection backend**: extend `_leiden` dispatch in
  `graph/communities.py` and gate its import with `pytest.importorskip`
  in tests.
- **A new math-correctness test**: claim a new `MM<n>` id, add to
  `docs/tests-math.md`, and cite the id in the test docstring.

---

## 6. Sources

- Gilpin, *Recurrences Reveal Shared Causal Drivers of Complex Time
  Series*, Phys. Rev. X 15, 011005 (2025). DOI:
  10.1103/PhysRevX.15.011005.
- arXiv:2301.13516 — same paper, open-access preprint.
- `docs/history/2026-05-refactor-assessment.md` — original audit and
  modularisation plan, frozen on completion.
