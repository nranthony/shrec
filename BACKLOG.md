# Backlog

One line per wanted, **unexamined** thing. Examining something promotes it to a
`work/NNNN` item and deletes its line here. Done also means the line is deleted.

- Real DTW distance: `metric="dtw"` is named in the `recurrence/kernel.py` docstring, but only `scipy.spatial.distance.cdist` metrics work.
- The classical union-find label quirk: the 2026-06 coverage plan (decision 0009) says `UnionFindReconstructor` labels stay muddied on `Simplicial+UnionFind` even with `sparsify=True`, but never says how; reproduce it before anything else.
