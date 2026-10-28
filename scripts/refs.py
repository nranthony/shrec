#!/usr/bin/env python3
"""Every path this repo refers to must exist.

Run `python3 scripts/refs.py` and `python3 scripts/refs_test.py` before and after
any deletion; the repo has no CI gate, and pytest's `testpaths` is `tests/` only,
so neither runs on its own. stdlib only.

Six checks over the files git knows about (`git ls-files`: committed or
staged). An untracked new file is not checked until it is `git add`ed.

  links      relative Markdown links resolve
  paths      repo-shaped paths in code comments resolve
  backticks  file paths quoted in Markdown resolve
  cites      a `work/NNNN` cite names an open item
  contents   a long Markdown file lists its `## ` sections under `## Contents`
  tokens     names of things that no longer exist do not appear

Deterministic by design: it answers "does this path exist?", never "is this doc
still worth keeping". Run it before and after any deletion — closing a work item
deletes its folder, and this is what finds the references that deletion left
behind.

Usage: python3 scripts/refs.py [--verbose]
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Trees to leave out: generated output, vendored code, fixtures that are fake
# repos, templates written for another filesystem. Each entry is a path prefix
# from the repo root, with its reason beside it.
SKIP_PREFIXES: tuple[str, ...] = (
    "docs/spbuild/",      # generated Sphinx output
    "docs/decisions/",    # awaiting deletion; its lines are in DECISIONS.md
)

# Top-level directories of this repo. A path-shaped string in a comment is only
# checked when it starts with one of these — that keeps URLs, container paths and
# example strings out of the check. Empty means the comment check does nothing.
REPO_DIRS: tuple[str, ...] = (
    "src/", "tests/", "benchmarks/", "examples/", "scripts/", "docs/", "work/",
)

# Names of things that no longer exist anywhere in this repo. A hit is a stale
# reference even when it is not shaped like a path. Each entry says why it died;
# this is also how a decision gets a check (name the entry in its line).
DEAD_TOKENS: dict[str, str] = {
    "docs/decisions/": "decision records are lines in DECISIONS.md",
}

# Files allowed to name a dead token, because naming it is their job.
TOKEN_ALLOW: tuple[str, ...] = (
    "AGENTS.md",
    "DECISIONS.md",
    "work/",
    "scripts/refs.py",
    "scripts/refs_test.py",
)

# Which files have their comments checked for repo paths, keyed by how a comment
# starts there. A suffix matches the end of the path, so a bare filename works.
# Trim to the languages this repo has.
COMMENT_DIALECTS: dict[str, tuple[str, ...]] = {
    "#": (".py", ".sh", ".toml", ".yml", ".yaml", "Makefile"),
}

# Quoted paths in Markdown that live in another tree on purpose: a file in a
# sibling repo, on a host, or inside a container image. Each with where it is.
FOREIGN: dict[str, str] = {
    "scripts/with-egress.sh": "a sandbox tool's script, named inside the managed "
                              "sandbox-notice block in AGENTS.md",
}
# Prefixes of quoted paths that are never in this tree: state directories on a
# host, mount points inside a container. A leading `/`, `~` or `-` is skipped
# already.
FOREIGN_PREFIXES: tuple[str, ...] = (
    # "state/",
)

# The records: files whose job is to name closed work items, and paths that have
# since moved, on purpose. Everything else that cites `work/<NNNN>` must name an
# open one. The quoted-path check skips them too, so a directory of dated records
# (a journal, run logs) is exempted from both by adding its prefix here.
CITE_RECORDS: tuple[str, ...] = (
    "work/", "scripts/refs.py", "scripts/refs_test.py",
    "docs/history/",                    # frozen records of finished work
)
CITE_RECORD_NAMES: tuple[str, ...] = (
    "DECISIONS.md", "BACKLOG.md", "CHANGELOG.md",
    "math-learning-notes.md",           # dated, append-only journal
)

# A long document needs a contents list. An agent's file reader returns a bounded
# slice per call, and agents often read only a file's head or a line range, so a
# list near the top lets them find a section and read it by line range. Lines can
# be wide, so bytes count as well as lines.
CONTENTS_BYTES = 20_000
CONTENTS_LINES = 300
CONTENTS_MIN_SECTIONS = 5
CONTENTS_WITHIN = 40                    # lines from the top
# Exempt by default: transient work items and the flat-list records.
CONTENTS_EXEMPT: tuple[str, ...] = ("work/",)
CONTENTS_EXEMPT_NAMES: tuple[str, ...] = (
    "LEDGER.md", "BACKLOG.md", "DECISIONS.md", "CHANGELOG.md",
    "math-learning-notes.md",           # read by round; its round index is the map
)
# Path prefixes for vendored or upstream trees whose docs this repo does not
# own, each with its reason beside it. Empty by default.
CONTENTS_EXEMPT_PREFIXES: tuple[str, ...] = (
    "docs/history/",      # frozen records, never navigated as references
)
CONTENTS_HEADING = re.compile(r"^## Contents\s*$")
H2 = re.compile(r"^## (\S.*?)\s*#*\s*$")
CONTENTS_ITEM = re.compile(r"^\s*[-*] \[([^\]]*)\]\(#([^)\s]*)\)\s*$")

MD_LINK = re.compile(r"\]\(([^)\s]+)\)")
FENCE = re.compile(r"^\s*(```|~~~)")
INLINE_CODE = re.compile(r"`[^`]*`")
# A comment body. The marker must follow whitespace or start the line, which
# keeps the `//` in a URL from reading as a comment.
COMMENTS = {
    marker: re.compile(rf"(?:^|\s){re.escape(marker)}\s?(.*)$")
    for marker in COMMENT_DIALECTS
}
PATHISH = re.compile(r"[A-Za-z0-9_.][A-Za-z0-9_./-]*/[A-Za-z0-9_./-]*")
# A path quoted in Markdown, `like/this.md`. Only a token with a `/` and a
# repo-file suffix, or a bare `name.md`, is checked: a bare script name is as
# often a file on a host or in an image as one here.
BACKTICK = re.compile(r"`([^`\s]+)`")
DOC_SUFFIX = re.compile(r"\.(md|sh|py|yml|yaml|toml)$")
NOT_A_PATH = set("<>*{}$:@=…[]|\\%")
# A cite of a work item by number. A closed item has a ledger row and its
# durable content is a decision line, which is what to cite instead.
WORK_CITE = re.compile(r"(?<![\w./-])work/(\d{4})(?!\d)")
# `work/0015` with no slug cites an item by number, not a path; check_cites
# decides whether it is open. A live item's files are cited with the slug, and
# those are checked as paths.
PROVENANCE = re.compile(r"^work/\d{4}$")


def ls_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return [p for p in out.stdout.split("\n") if p]


def tracked(every: list[str]) -> list[str]:
    return [p for p in every if not p.startswith(SKIP_PREFIXES)]


def open_items() -> set[str]:
    """A work item is open while its folder exists. No work/ means none is."""
    work = ROOT / "work"
    if not work.is_dir():
        return set()
    return {p.name[:4] for p in work.iterdir() if p.is_dir() and re.match(r"\d{4}-", p.name)}


def resolves(rel: str, cand: str) -> bool:
    """A comment names a path from wherever its author was standing — the repo root,
    a package root, or the file's directory. Any of them counts."""
    base = (ROOT / rel).parent
    while True:
        if (base / cand).exists():
            return True
        if base == ROOT:
            return False
        base = base.parent


def outside_html_comments(line: str, in_comment: bool) -> tuple[str, bool]:
    """The parts of `line` that are not inside `<!-- -->`, and whether a comment
    is still open at its end. A link someone commented out is not a reference."""
    kept = []
    while line:
        if in_comment:
            end = line.find("-->")
            if end < 0:
                break
            line, in_comment = line[end + 3:], False
        else:
            start = line.find("<!--")
            if start < 0:
                kept.append(line)
                break
            kept.append(line[:start])
            line, in_comment = line[start + 4:], True
    return "".join(kept), in_comment


def check_links(rel: str, text: str, fail: list[str]) -> None:
    in_fence = in_comment = False
    for n, line in enumerate(text.splitlines(), 1):
        # A fence marker inside an HTML comment is commented out with the rest.
        if not in_comment and FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        # Inline code first: `<!--` quoted in backticks opens no comment.
        line, in_comment = outside_html_comments(INLINE_CODE.sub("", line), in_comment)
        for target in MD_LINK.findall(line):
            target = target.split("#", 1)[0]
            if not target or re.match(r"^[a-z][a-z0-9+.-]*:", target):
                continue
            if not (ROOT / rel).parent.joinpath(target).exists():
                fail.append(f"{rel}:{n}: link -> {target}")


def comment_pattern(rel: str) -> re.Pattern[str] | None:
    for marker, suffixes in COMMENT_DIALECTS.items():
        if rel.endswith(suffixes):
            return COMMENTS[marker]
    return None


def check_paths(rel: str, text: str, comment: re.Pattern[str], fail: list[str]) -> None:
    """Repo-shaped paths named in comments, where a wrong one misleads the next
    reader and nothing else would catch it."""
    for n, line in enumerate(text.splitlines(), 1):
        m = comment.search(line)
        if not m:
            continue
        for cand in PATHISH.findall(m.group(1)):
            cand = cand.rstrip(".,:;)\"'")
            if not cand.startswith(REPO_DIRS):
                continue
            if "*" in cand or "<" in cand or cand.endswith("/"):
                continue
            if PROVENANCE.match(cand):
                continue
            if not resolves(rel, cand):
                fail.append(f"{rel}:{n}: path -> {cand}")


def check_backticks(rel: str, text: str, fail: list[str], every: list[str]) -> None:
    """Paths quoted in Markdown. A doc names a file from wherever its author was
    standing, so a path counts if it resolves from the doc's directory upward or
    is the tail of any tracked path; a bare `name.md` counts if any tracked file
    has that name. A link checker cannot see this class: a quoted path to a file
    deleted long ago reads exactly like a live one."""
    if rel.startswith(CITE_RECORDS) or rel.endswith(CITE_RECORD_NAMES):
        return
    names = {p.rsplit("/", 1)[-1] for p in every}
    in_fence = False
    for n, line in enumerate(text.splitlines(), 1):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for cand in BACKTICK.findall(line):
            cand = cand.rstrip(".,:;)")
            if not DOC_SUFFIX.search(cand) or cand in FOREIGN:
                continue
            if cand.startswith(("/", "~", "-") + FOREIGN_PREFIXES) or NOT_A_PATH & set(cand):
                continue
            if "/" not in cand:
                if cand.endswith(".md") and cand not in names:
                    fail.append(f"{rel}:{n}: quoted -> {cand}")
                continue
            if "." in cand.split("/", 1)[0].lstrip("."):
                continue                                # a host name: example.com/x.sh
            tail = cand.removeprefix("./")
            if resolves(rel, cand) or any(p == tail or p.endswith("/" + tail) for p in every):
                continue
            fail.append(f"{rel}:{n}: quoted -> {cand}")


def check_cites(rel: str, text: str, fail: list[str], open_now: set[str]) -> None:
    """A `work/NNNN` outside the records names an item that must be open. Once it
    closes, cite the decision line it distilled to; the ledger keeps the rest."""
    if rel.startswith(CITE_RECORDS) or rel.endswith(CITE_RECORD_NAMES):
        return
    for n, line in enumerate(text.splitlines(), 1):
        for m in WORK_CITE.finditer(line):
            if m.group(1) not in open_now:
                fail.append(f"{rel}:{n}: cite -> {m.group(0)} (closed; "
                            "cite the decision line it distilled to)")


def slug(title: str) -> str:
    """GitHub's anchor for a heading: lowercase, punctuation dropped, spaces to
    hyphens. Backticks and link brackets go with the rest of the punctuation."""
    return re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")


def check_contents(rel: str, text: str, fail: list[str]) -> None:
    """A long file's `## Contents` must list its `## ` sections, in order. The list
    is checked so a renamed or added heading cannot leave it stale."""
    if rel.startswith(CONTENTS_EXEMPT + CONTENTS_EXEMPT_PREFIXES) or rel.endswith(
        CONTENTS_EXEMPT_NAMES
    ):
        return
    lines = text.splitlines()
    if len(text.encode("utf-8")) <= CONTENTS_BYTES and len(lines) <= CONTENTS_LINES:
        return
    heads: list[tuple[int, str]] = []
    in_fence = False
    for n, line in enumerate(lines, 1):
        if FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence and (m := H2.match(line)):
            heads.append((n, m.group(1)))
    at = next((n for n, t in heads if CONTENTS_HEADING.match(f"## {t}")), 0)
    heads = [(n, t) for n, t in heads if n != at]
    if len(heads) < CONTENTS_MIN_SECTIONS:
        return
    if not at or at > CONTENTS_WITHIN:
        fail.append(f"{rel}: contents -> no `## Contents` within the first "
                    f"{CONTENTS_WITHIN} lines ({len(heads)} sections)")
        return
    want = [slug(t) for _, t in heads]
    got: list[str] = []
    for line in lines[at:]:
        if not line.strip():
            continue
        if not (m := CONTENTS_ITEM.match(line)):
            break
        got.append(m.group(2))
    if got == want:
        return
    missing = [a for a in want if a not in got]
    extra = [a for a in got if a not in want]
    if missing or extra:
        fail.append(f"{rel}:{at}: contents -> missing {missing or 'none'}, "
                    f"extra {extra or 'none'}")
    else:
        fail.append(f"{rel}:{at}: contents -> links are out of document order")


def check_tokens(rel: str, text: str, fail: list[str]) -> None:
    if rel.startswith(TOKEN_ALLOW) or rel.endswith("/AGENTS.md"):
        return
    for n, line in enumerate(text.splitlines(), 1):
        for token, why in DEAD_TOKENS.items():
            if token in line:
                fail.append(f"{rel}:{n}: token -> {token} ({why})")


def main() -> int:
    verbose = "--verbose" in sys.argv
    fail: list[str] = []
    checked = 0
    every = ls_files()
    open_now = open_items()
    for rel in tracked(every):
        path = ROOT / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        checked += 1
        comment = comment_pattern(rel)
        if rel.endswith(".md"):
            check_links(rel, text, fail)
            check_backticks(rel, text, fail, every)
            check_contents(rel, text, fail)
        if comment:
            check_paths(rel, text, comment, fail)
        if rel.endswith(".md") or comment:
            check_cites(rel, text, fail, open_now)
        check_tokens(rel, text, fail)

    if fail:
        print("\n".join(sorted(fail)), file=sys.stderr)
        print(
            f"\n{len(fail)} stale reference(s) across {checked} tracked files.",
            file=sys.stderr,
        )
        return 1
    if verbose:
        print(f"{checked} tracked files checked")
    print("refs: every referenced path exists")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
