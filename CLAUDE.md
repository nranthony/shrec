# CLAUDE.md — agent guidance for the shrec repo

This file is the canonical onboarding doc for any AI coding agent working on
this project. `GEMINI.md` and `AGENTS.md` are thin pointers to this file.

## What this project is

`shrec` (shared recurrence) is the companion code to Gilpin,
*Recurrences Reveal Shared Causal Drivers of Complex Time Series*,
**Phys. Rev. X 15, 011005 (2025)** (arXiv:2301.13516). The local PDF is
`docs/PhysRevX.15.011005.pdf`. The SHREC algorithm reconstructs an
unobserved driver `z(t)` from `N` measured response time series `x_k(t)`.

The clearest summary of the algorithm is the paper's **Appendix B**
("Shared Reconstruction Algorithm Implementation"), starting roughly at
PDF page 11.

## Where to look

- `docs/architecture.md` — paper-to-code mapping, current sub-package
  layout, and the design decisions fixed by the 2026-05 refactor
  (Fiedler vs SVD, simplicial vs exp kernel, RNG plumbing, etc.).
  **Read this first** before changing algorithm code.
- `docs/tests-math.md` — live catalog of math-correctness oracles
  (`MM<n>` ids). Every new math-test should claim a fresh MM id and
  update this file.
- `docs/history/2026-05-refactor-assessment.md` — frozen record of the
  original audit + modularisation plan. Useful for *why* the code
  looks the way it does; not guidance for new work.
- `docs/pre-release-notes.md` — packaging / release punch list for
  the `0.2.0` PyPI publish.

## Current state

The 2026-05 refactor is complete. The library is structured as a
proper Python package with the public API at the top level:

```python
from shrec import RecurrenceClustering, RecurrenceManifold
```

Test suite (`uv run pytest`): **66 passed, 6 skipped (optional
`igraph`/`leidenalg`/`cdlib` backends), 2 documented xfailed**. The
math-correctness catalog (`docs/tests-math.md`) now has all 14 "must"
tests green or documented-xfail.

The two xfails are tracked in `docs/tests-math.md`:

- **MM5** — `dataset_to_simplex` vs `umap.umap_.fuzzy_simplicial_set`
  diverge on σ-solver conventions. The refactor chose
  `dataset_to_simplex` (paper-faithful, no umap dependency). Lifting
  the xfail requires reconciling the conventions.
- **MM20** — period-4 Sauer-limit ARI ≈ 0.50. **This is a
  representational limit, not a Leiden-resolution artifact** (don't
  chase resolution tuning — it's been ruled out). A modularity sweep
  jumps 2 communities → ~1000 singletons with no stable 4-community
  regime; an *oracle* spectral k-means=4 on the consensus Laplacian
  also gives ARI≈0.5; the continuous `RecurrenceManifold` gives
  Spearman |ρ|≈0.38. The four levels collapse into a low/high 2-way
  split in the recurrence graph. Closing it needs a richer recurrence
  representation. Pinned by `test_period_four_is_not_separable_in_graph`.

## Canonical SHREC pipeline (Appendix B)

1. Time-delay embed each response:
   `ŷ_k(t) = [x_k(t), …, x_k(t-(D-1)τ)]ᵀ`.
   — `shrec.embeddings.embed_ts` / `make_embedding`.
2. Per-response pairwise distance `d^(k)_ij = ‖ŷ_k(i) − ŷ_k(j)‖`.
   — `scipy.spatial.distance.cdist` inside the recurrence primitives.
3. Per-response **fuzzy simplicial complex** with adaptive `ρ_i, σ_i`
   (root-solve `log₂ M = Σ_m exp[-ReLU(d_im − ρ_i)/σ_i]` per row).
   — `shrec.recurrence.simplicial.dataset_to_simplex`.
4. Consensus aggregation `A = mean_k A^(k)`.
   — `shrec.recurrence.consensus.data_to_connectivity2`.
5. **Discrete driver** → Leiden community detection on `A`
   (`shrec.graph.communities._leiden` → `RecurrenceClustering`).
   **Continuous driver** → **Fiedler eigenvector** of `L = D − A`
   (`scipy.linalg.eigh` inside `RecurrenceManifold`).

`ClassicalRecurrenceClustering` (Sauer baseline) uses the simpler
exp-kernel recurrence at `shrec.recurrence.kernel.data_to_connectivity`
with `ord=500.` to approximate the Sauer `inf_k d^(k)`.

## Layout

```
src/shrec/
├── __init__.py             # version + top-level public API
├── embeddings.py           # embed_ts, hankel_matrix, make_embedding
├── recurrence/
│   ├── simplicial.py       # dataset_to_simplex (canonical)
│   ├── kernel.py           # data_to_connectivity (Sauer baseline)
│   └── consensus.py        # data_to_connectivity2 (mean aggregation)
├── graph/
│   ├── communities.py      # _leiden multi-backend adapter
│   └── unionfind.py        # DisjointSet, solve_union_find
├── models/
│   ├── base.py             # RecurrenceModel (sklearn-clean)
│   ├── recurrence_clustering.py
│   ├── recurrence_manifold.py
│   ├── classical.py
│   ├── hirata_nomura.py
│   └── models.py           # back-compat re-export shim
└── utils/                  # legacy organisation, preserved via shims
benchmarks/                 # driver-response synthetic systems
tests/                      # pytest suite
docs/                       # architecture / tests-math / history / paper
```

Old import paths (`from shrec.models.models import …`,
`from shrec.utils import …`) continue to work via shims.

## Testing

The suite is **pytest**. Run from the repo root:

```
uv run pytest                 # everything
uv run pytest tests/test_preprocessing.py -v
uv run pytest -m "not slow"   # skip the math-correctness sweeps
```

Conventions:

- Each `tests/test_*.py` maps to one source module (or one math-
  correctness MM id, or one bug-fix group).
- Use the fixtures in `tests/conftest.py` (`rng`, `tiled_sine`,
  `square_wave_responses`). Never call `np.random.seed` — use a
  local `np.random.default_rng(seed)` instead.
- Gate backend-dependent tests with `pytest.importorskip("cdlib")`
  etc. so the core suite still runs without optional packages.
- New regression tests should cite the relevant bug ID or `MM<n>`
  math-test ID in their docstring, so the traceability holds up.

The legacy `tests/test_models.py` is a smoke test only; new tests
should not extend it.

## Working style for this repo

- Prefer editing existing files over creating new ones.
- Don't add error handling, fallbacks, or backwards-compatibility
  shims unless explicitly requested. The repo is research code with a
  single upstream maintainer — keep changes minimal and direct.
- Don't add comments that just describe what the code does. Comments
  belong only where the *why* is non-obvious (an algorithmic
  invariant, a known limitation, a deferred bug).
- Random state: always plumb a local
  `np.random.default_rng(self.random_state)`. Never call
  `np.random.seed(...)` (audit item A6, fixed in the refactor;
  reintroducing it would silently mutate global state).
- `fit(X, y=None)` returning `self` is the contract for every model.
  Per-call kwargs belong on `__init__`.
- When changing the algorithm, cite the paper section
  (e.g. "per Appendix B, eq. B3") in the commit message.

## Environment

`uv` manages the venv. From the repo root:

```
uv sync --dev          # install runtime + dev deps
uv run pytest          # run the tests
```

The paper PDF is large; `pypdf` is in the dev deps so
`pypdf.PdfReader` works for text extraction. There is no
`poppler-utils` in the container.

## When in doubt

1. Re-read the relevant section of `docs/architecture.md` for the
   canonical layout / design decisions, and `docs/tests-math.md`
   for the test catalog.
2. Check the paper's Appendix B for the canonical algorithm step.
3. Add the regression test before the fix; let it fail, then make it
   pass. Cite a `MM<n>` id (claim a new one if needed) and update
   `docs/tests-math.md`.
