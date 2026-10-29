# Decisions still in force

One line per decision. The long form and the rejected alternatives live in the
commit that carries the decision. Numbers are stable and never reused.

0005, 0006, 0007 and 0009 are absent because they are history: findings of the
2026-06 test audit and a coverage plan, all applied, whose durable content now
lives in the tests, their docstrings and `docs/tests-math.md`. That audit
(0001–0007) covered pipeline steps 2–4 and the Fiedler reconstructor; time-delay
embedding and Leiden were outside it, so its verdict that the suite tests the
right math is bounded to those. The original records, these included, were one
file each under `docs/decisions/`: `git log -- docs/decisions/` finds them in any
clone, and in the checkout that migrated them,
`git show pre-migrate-conventions:docs/decisions/<file>`.

Two layers. Numbers say nothing about the layer; a line moves between sections by
being moved. If something outside this repo relies on it, it is an invariant; if
changing it needs only mechanical updates in here, it is a current choice; when
unsure, it is an invariant until a human says otherwise.

## Invariants

Costly to reverse, or something outside this repo relies on it. Pause and ask
before breaking one. Where a check enforces it, the line names the check.

- **0002** The consensus affinity is the elementwise mean over responses (`1/K`), as Appendix B's prose says ("elementwise average"); its printed equation, a bare sum, is a typo, so the code is not changed to match it. Leiden and the Fiedler vector are invariant to a global scale, so the driver is the same either way, but the stored `A` is not. Check: `uv run pytest tests/test_models_invariances.py -k consensus_mean_is_idempotent`.
- **0003** ρᵢ is the distance to the nearest neighbour (`[:k]` of the sorted row once the self-distance is inf-filled), per Appendix B's `ρᵢ ≡ minₘ dᵢₘ`; the paper's formula is not a typo, and the old `[1:k+1]` slice, which picked the second-nearest, was the bug. Check: `uv run pytest tests/test_recurrence_simplicial.py -k test_rho_matches_umap_exactly` (MM5).
- **0004** The σ root-solve sums over the `k` nearest real neighbours, nearest included (`M = k`), as Appendix B does; umap counts the point itself, and the ~4% σ gap that leaves is a convention, not a target. Check: `uv run pytest tests/test_recurrence_simplicial.py` (MM1–MM5).
- **0010** The default Leiden backend calls `graspologic_native.leiden` directly and never imports `graspologic`, whose numpy<2 pin would stop this package installing beside numpy 2, which a downstream consumer requires; the method keeps the name `"graspologic"` so callers' saved configs need no change. Check: `uv run pytest tests/test_graph_communities.py`.
- **0011** `shrec.models.models` and `shrec.utils` keep re-exporting every name they export today (the import surface from before the 2026-05 split), and the four preset constructors keep their scalar signatures; the in-tree notebooks and 0.1.x callers import through them. Drop only at a release boundary, with a deprecation cycle. Check: `uv run pytest tests/test_back_compat_shims.py`.
- **0012** `RecurrenceManifold` takes the Fiedler eigenvector of `L = D − A` (`scipy.linalg.eigh`), not `TruncatedSVD(A).components_[1]`, which coincides with it only on regular graphs; reproducing the paper's continuous driver depends on it. Check: `uv run pytest tests/test_models_recurrence_manifold.py` (MM22).
- **0013** The two canonical presets, `RecurrenceClustering` and `RecurrenceManifold`, default to the adaptive fuzzy simplicial complex (`SimplicialConnectivity`) as their connectivity stage; the fixed-σ exp kernel is the Sauer baseline's stage only. Check: `uv run pytest tests/test_models_recurrence_clustering.py` (MM19).
- **0015** No model constructor or `fit` touches the global numpy RNG: randomness comes from a local `np.random.default_rng(self.random_state)`, never `np.random.seed`. Check: `uv run pytest tests/test_models_base.py -k TestRngIsolation` (MM18, MM31).
- **0016** Every model follows the sklearn contract: behavioural knobs live on `__init__`, which only assigns them, and `fit(X, y=None)` returns `self`. Check: `uv run pytest tests/test_models_base.py -k TestSklearnContract` (MM30).

## Current choices

The approach in use. The code wins: when work moves away from one, edit or delete
its line in the same commit. No supersession record, no pause.

- **0001** A claim about the canonical algorithm is settled before the code changes by checking it against the paper's text (Appendix B/E, extracted with `pypdf`) and against the original author's intent in the pre-refactor git history, plus `umap.umap_` for the simplicial step alone, then checking the readings agree; act only when they converge.
- **0008** The four public models are presets over `ShrecPipeline`, one `Connectivity` stage and one `Reconstructor` stage each; new behaviour is a new stage strategy, not a new model class, and any combination is one `ShrecPipeline(connectivity=…, reconstructor=…)` (MM37 pins preset ≡ pipeline).
- **0014** `dataset_to_simplex` is the one fuzzy-simplicial implementation; `umap-learn` is an optional extra (`shrec[sim]`), imported only by the MM5 parity tests, which skip without it.
