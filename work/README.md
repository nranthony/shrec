# work/

One numbered folder per unit of work. `NNNN` is the next free number; numbers are
never reused, including by closed items, so the next number is one more than the
highest in this directory **and in [LEDGER.md](LEDGER.md)**. This is the only place
a plan, proposal or in-flight investigation lives.

Work numbers are their own sequence, separate from the decision numbers in
[DECISIONS.md](../DECISIONS.md). "Plan 0009" in older docstrings is decision 0009,
a coverage plan written before this directory existed, not a work item.

`BACKLOG.md` at the repo root holds what has not been thought about yet: one line,
no folder. The moment something is examined it gets a number and its line is
deleted. Promotion is the only way in; deletion is the only way out.

## Inside an item

Whichever of these the work needs — the one that starts it is the only required file:

- `proposal.md` — "should we do this?", when the answer is not yet settled
- `spec.md` — the pinned what and why, for work already decided
- `plan.md` — the implementation plan
- `notes.md` — running notes while executing

Front-matter is YAML, at the top of the item's first file:

```yaml
---
number: 0001
status: proposed | active | blocked | done
opened: YYYY-MM-DD
touches: [paths this item will change]
---
```

Four statuses, no others. `blocked` names what it is blocked on in the body.

## Closing an item: distill, delete, tombstone

An item may close only when everything durable in it has been **distilled somewhere
live** — a `DECISIONS.md` line, a rule in an `AGENTS.md`, a skill, a code comment.
Nothing durable may live only in `work/`.

Then the folder is deleted and one row is added to [LEDGER.md](LEDGER.md):

```
NNNN · slug · opened → closed · <where it distilled to> — <what the folder held that nothing else does, and what was rejected>
```

No commit hashes: a row cannot name the commit that contains it, and git finds both
from the folder's name. `git log -1 --format=%h -- work/NNNN-slug/` is the closing
commit; its parent still holds the folder
(`git show <closing>^:work/NNNN-slug/<file>`).

There is no archive directory. Git holds the full text; the ledger row says what came
out and what was turned down. Before proposing an approach in an area, look in the
ledger for one already tried or rejected. An agent proposes the deletion by naming the path; a human
performs it or approves it.

## work/inbox/

Tracked. One file per inbound report from an agent that hit something wrong here: a
skill whose instructions did not fit, a footgun nobody documented, a convention that
failed in practice. The agent commits the file locally; a human pushes. A report is a
claim, not a patch: it is read during triage, which appends a `Disposition:` line and
leaves the file in place. If the repo ignores a root `inbox/`, or denies agents it in a
permission rule, anchor each such rule at the root (`/inbox/`, `Read(/inbox/**)`) so this
directory stays tracked and writable.

## work/outbox/

Tracked, created when first needed. One file per report this repo sends about a shared
skill it uses but does not ship, in the filename and envelope the owner's `work/inbox/`
takes. The agent commits it; a human carries it to the owner's `work/inbox/`, then
appends `Delivered: <date>` to this copy. Nothing in it is deleted, and none of it is
triaged in this repo.
