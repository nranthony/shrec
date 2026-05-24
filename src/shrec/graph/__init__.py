"""Graph primitives for SHREC: community detection, union-find."""
from shrec.graph.communities import _leiden
from shrec.graph.unionfind import DisjointSet, solve_union_find

__all__ = ["_leiden", "DisjointSet", "solve_union_find"]
