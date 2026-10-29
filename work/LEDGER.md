# Ledger

One row per closed work item. The folder is gone; this says where its durable content
went, what the folder held that nothing else does, and what was rejected.

```
NNNN · slug · opened → closed · <distilled to> — <the folder's unique content, and what was rejected>
```

No commit hashes, because a row cannot name the commit that contains it and git finds
both from the folder's name:

```
git log -1 --format=%h -- work/NNNN-slug/      # the closing commit: read it to see what survived
git show <closing>^:work/NNNN-slug/<file>      # its parent still holds the folder: what was thought
git log --oneline -- work/NNNN-slug/           # the item's whole arc
```

A closed item's number stays taken: this file is where the next free number is checked.
