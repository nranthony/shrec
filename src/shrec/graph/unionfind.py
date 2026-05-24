"""Hand-rolled disjoint-set (union-find) used by `ClassicalRecurrenceClustering`.

Predates the introduction of `scipy.cluster.hierarchy.DisjointSet`
(≥ scipy 1.6); kept for parity with the legacy behaviour. The
modularisation PR could swap in the scipy implementation behind a thin
adapter once an equivalence test (MM11) lands.
"""


class DisjointSet:
    """A disjoint set data structure.

    Adapted from open-source code:
        https://stackoverflow.com/questions/67805907
    """

    class Element:
        def __init__(self):
            self.parent = self
            self.rank = 0

    def __init__(self):
        self.elements = {}

    def find(self, key):
        el = self.elements.get(key, None)
        if not el:
            el = self.Element()
            self.elements[key] = el
        else:  # Path splitting algorithm
            while el.parent != el:
                el, el.parent = el.parent, el.parent.parent
        return el

    def union(self, key=None, *otherkeys):
        if key is None:
            return
        root = self.find(key)
        for otherkey in otherkeys:
            el = self.find(otherkey)
            if el != root:
                if root.rank < el.rank:
                    root, el = el, root
                el.parent = root
                if root.rank == el.rank:
                    root.rank += 1

    def groups(self):
        result = {el: [] for el in self.elements.values() if el.parent == el}
        for key in self.elements:
            result[self.find(key)].append(key)
        return result


def solve_union_find(lists):
    """Given a list of merge groups, return each group's full equivalence class."""
    disjoint = DisjointSet()
    for lst in lists:
        disjoint.union(*lst)
    groups = disjoint.groups()
    return [lst and groups[disjoint.find(lst[0])] for lst in lists]
