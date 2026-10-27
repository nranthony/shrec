# 0010 — Leiden backend: call graspologic-native directly, drop graspologic

**Status:** applied · **Date:** 2026-10-27

## Context

The default Leiden backend was `graspologic.partition.leiden`, the only use
of the `graspologic` package in this repo (`graph/communities.py`).
graspologic 3.4.4, its latest release and its `main` branch, requires
`numpy<2`. Downstream, the numerai host needs `numerai-tools>=0.7.0`, which
requires `numpy>=2` (numerai ADR-0009 d.9 and its 2026-10-27 addendum); the
two pins cannot be satisfied in one environment.

`graspologic.partition.leiden` is a thin wrapper: it converts the adjacency
matrix to a `(str(i), str(j), float(w))` edge list over the upper triangle,
calls `graspologic_native.leiden` (the Rust core, a separate package by the
same publisher, `numpy>=1.24`), and maps the returned `{str: int}` back. The
`is_unweighted` scan it runs first only rewrites a weight of 1 to 1.0.

## Decision

1. `graph/communities.py` builds the edge list itself (`_adjacency_to_edge_list`,
   same entries in the same order as graspologic 3.4) and calls
   `graspologic_native.leiden` with graspologic's defaults: one iteration,
   one trial, randomness 0.001, no starting communities. The symmetry check
   (`abs(A − Aᵀ).max() ≤ 1e-15`) and the positive-seed rule are kept so the
   same inputs are accepted and rejected. Isolates are omitted from the
   result with the same warning; `LeidenReconstructor` already fills them
   with −1.
2. The method keeps its name, `"graspologic"`, so `RecurrenceClustering`,
   `LeidenReconstructor` and downstream configs need no change.
3. `pyproject.toml`: `graspologic>=3.4,<4` → `graspologic-native>=1.2.1,<2`;
   `future` (the `past` shim for graspologic → hyppo) and `numba` (never
   imported in `src/`) are removed. `statsmodels` stays (stationarity tests).
4. Tests that gated on `importorskip("graspologic")` gate on
   `graspologic_native`; `test_graph_communities.py` adds the native-leg
   regressions (runs without `graspologic` importable, dense and sparse
   agree, directed input rejected, non-positive seed rejected, isolates
   warned, CPM resolution forwarded).

## Consequences

- Any scipy sparse matrix is now accepted on the default path (graspologic
  took only `csr_array`); dense inputs behave exactly as before.
- Partitions for a given seed are expected to be identical to graspologic's,
  since the native call and the edge order are the same. The numerai host's
  regime step still recomputes once, because its signature includes the
  Leiden package name and version.
- The repo's own `uv.lock` must be regenerated on a host (`uv lock`,
  `uv sync --dev`, `uv run pytest`); the sandbox has no registry access.
