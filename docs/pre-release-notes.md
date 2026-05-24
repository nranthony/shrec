# Pre-release notes — `shrec` packaging & usability

Working notes captured before the first PyPI release. Not a public changelog;
this is the punch-list for *getting* to a release. Once `0.2.0` ships, the
relevant parts of this file get promoted into `CHANGELOG.md` / the README and
the rest is deleted.

## Status today (2026-05-23)

`pyproject.toml` already declares a setuptools build backend with the
`src/` layout, so `pip install git+https://…/shrec.git@<tag>` works *today*
for collaborators. The real blockers to outside use are dependency hygiene,
public API surface, and the in-flight refactor — not packaging mechanics.

## Update (2026-05-23, end of refactor)

All the punch-list items below are now applied:

- Version bumped to `0.2.0.dev0`. Local-version segment `+linear-hamster`
  removed. `requires-python` loosened to `>=3.10`. `numpy` / `scipy` /
  `scikit-learn` / `networkx` / `pandas` pinned to compatible ranges
  rather than exact versions.
- Heavy/optional deps split into extras:
  - `shrec[viz]` — `matplotlib`, `seaborn`
  - `shrec[notebook]` — `jupyterlab`, `ipykernel`, `ipywidgets`
  - `shrec[graph-extras]` — `leidenalg`, `python-igraph`, `cdlib`
  - `shrec[bio]` — `scanpy`
  - `shrec[sim]` — `umap-learn` (legacy fuzzy-simplicial impl)
  - `shrec[all]` — union of the above, plus `h5py`
- Top-level public API exposed: `from shrec import RecurrenceManifold,
  RecurrenceClustering, ClassicalRecurrenceClustering, HirataNomuraIsomap,
  embed_ts, hankel_matrix, make_embedding` all work.
- `umap-learn` is no longer imported by core (was unused on production
  paths; now lives in `shrec[sim]` as a back-compat install).
- Test suite: 39 passed, 2 skipped (`cdlib` in optional extra), 2
  xfailed (documented in CLAUDE.md).

Remaining for the actual TestPyPI → PyPI release:

1. Tag `v0.2.0`.
2. `uv build`; smoke-test the wheel in a fresh venv.
3. `uv publish --repository testpypi`; verify the install link.
4. `uv publish` to PyPI proper.

## Minimum-to-publish punch list

These are the items that don't depend on the canonical-algorithm refactor
(`docs/history/2026-05-refactor-assessment.md` §6 step 2) and can land at any time.

1. **Loosen version pins.**
   - `numpy==1.26.4`, `scipy==1.12.0`, `requires-python = "==3.12.*"` will
     collide with every downstream app.
   - Move to ranges: `numpy>=1.26,<3`, `scipy>=1.12,<2`, `python>=3.10`.
   - Drop `+linear-hamster` local version segment — PyPI rejects it.

2. **Split heavy deps into `[project.optional-dependencies]` extras.**
   - **Core** (always installed): `numpy`, `scipy`, `scikit-learn`,
     `networkx`, `graspologic`, `numba`, `pandas`, `statsmodels`.
   - **`shrec[notebook]`**: `jupyterlab`, `ipykernel`, `ipywidgets`.
   - **`shrec[viz]`**: `matplotlib`, `seaborn`.
   - **`shrec[bio]`**: `scanpy` (only used by one benchmark dataset).
   - **`shrec[all]`**: union of the above.
   - **`umap-learn` is a special case** — see "umap caveat" below.

3. **Define the top-level public API in `src/shrec/__init__.py`.**
   Today users have to know `from shrec.models.models import
   RecurrenceManifold`. Re-export the four canonical models so
   `from shrec import RecurrenceManifold, RecurrenceClustering,
   ClassicalRecurrenceClustering, HirataNomuraIsomap` just works.
   This is the change that lets the library be discovered.

4. **Publish workflow: TestPyPI → tag → PyPI.**
   Use `uv build` + `uv publish --repository testpypi` for a dry run on a
   throwaway version (e.g. `0.2.0rc1`). Only push to PyPI proper once the
   install works clean in a fresh venv.

## Tradeoff: when to actually ship

`docs/history/2026-05-refactor-assessment.md` is mid-refactor. Step 2 (canonical-algorithm PR) will
change what `RecurrenceManifold().fit_predict(X)` *computes* — switching
from TruncatedSVD on `A` to the Fiedler eigenvector of `L = D − A`, and
from the exp-kernel recurrence to the fuzzy simplicial complex. Step 4
(modularisation PR) will reshape `fit` signatures.

If we publish `0.1.x` to PyPI now, every early adopter inherits churn when
those land. Cleaner play:

- Do the dep + extras + top-level-export cleanup *now* — none of it
  depends on the refactor.
- Hold off on PyPI until **after** §6 step 2 (canonical algorithm) lands.
  Freeze the public API at `0.2.0` and tag it.
- Until then, collaborators use `pip install git+…@<tag>`.

## umap caveat

`umap-learn` is currently a hard runtime dep (top-level `from umap.umap_
import fuzzy_simplicial_set` at `src/shrec/models/models.py:25`), but the
only function that uses it (`data_to_connectivity2`) is **not on any
production path** — every `fit` calls `data_to_connectivity` (the simple
exp-kernel variant) instead. So today, umap is imported-but-unused.

After §6 step 2 lands, the fuzzy simplicial complex moves *onto* the
production path, and we have to choose:

- **Keep `umap-learn` in core** — convenient, lets us delegate to
  `umap.umap_.fuzzy_simplicial_set` which is well-tested upstream.
- **Drop `umap-learn` entirely** — use the in-repo
  `models.dataset_to_simplex` (≈ 20 lines, same root-solve). Trades a
  heavy dep for self-contained code, at the cost of owning the
  implementation.

Decision deferred to the §6 step 2 PR. MM5 in `docs/tests-math.md` §5b.1
is the parity test that lets us prove the two implementations are
equivalent before making the call.

## Usability strand (separate from packaging)

Out of scope for the first release, but worth scoping:

- `examples/quickstart.ipynb` — one notebook that goes from
  `X = load_lorenz()` to a recovered driver plot in ≤ 20 cells. Single
  highest-leverage docs artifact.
- `shrec.plotting` module — two functions to start:
  `plot_driver_overlay(z_true, z_pred)` and
  `plot_recurrence_matrix(A)`. Reduces the boilerplate notebook users
  have to write to see anything.
- Sphinx site — deferred until there's demand. The existing
  `docs/conf.py` skeleton is enough to bootstrap when needed; a clean
  README + the quickstart notebook covers 90% of users until then.
