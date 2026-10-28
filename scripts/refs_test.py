#!/usr/bin/env python3
"""Offline tests for refs.py's quoted-path and work-cite checks.

Both checks take the text and the tree as arguments, so these tests hand them
synthetic files and a synthetic `git ls-files` list rather than this repo: the
rules are the subject, not today's tree. stdlib only, matching refs.py.

STARTING POINT: it passes against the unadapted refs.py. After filling in
FOREIGN or FOREIGN_PREFIXES there, add a case here for each entry.

Usage: python3 scripts/refs_test.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refs  # noqa: E402

EVERY = [
    "app/scripts/deploy.sh",
    "app/docs/debug-recipes.md",
    "lib/README.md",
]


def quoted(text: str, rel: str = "app/README.md") -> list[str]:
    fail: list[str] = []
    refs.check_backticks(rel, text, fail, EVERY)
    return fail


def cites(text: str, rel: str = "app/README.md", open_now=("0021",)) -> list[str]:
    fail: list[str] = []
    refs.check_cites(rel, text, fail, set(open_now))
    return fail


class Backticks(unittest.TestCase):
    def test_tail_of_a_tracked_path_counts(self):
        # Docs name a script from wherever they stand.
        self.assertEqual(quoted("run `scripts/deploy.sh` first"), [])

    def test_a_path_that_is_nowhere_fails(self):
        self.assertEqual(len(quoted("see `docs/retired-plan.md`")), 1)

    def test_a_bare_md_name_must_be_some_tracked_file(self):
        self.assertEqual(quoted("`debug-recipes.md`"), [])
        self.assertEqual(len(quoted("`retired-plan.md`")), 1)

    def test_a_bare_script_name_is_not_checked(self):
        # A bare script name is as often on a host or in an image as here.
        self.assertEqual(quoted("`install.sh` and `setup.py`"), [])

    def test_host_absolute_and_placeholder_paths_are_skipped(self):
        text = ("`example.com/install.sh` `/usr/local/bin/x.sh` `~/notes/x.md` "
                "`<repo>/x.md` `docs/*.md`")
        self.assertEqual(quoted(text), [])

    def test_foreign_entries_are_skipped(self):
        saved = refs.FOREIGN, refs.FOREIGN_PREFIXES
        try:
            refs.FOREIGN = {"docs/elsewhere.md": "in a sibling repo"}
            refs.FOREIGN_PREFIXES = ("state/",)
            self.assertEqual(quoted("`docs/elsewhere.md` `state/run/log.md`"), [])
        finally:
            refs.FOREIGN, refs.FOREIGN_PREFIXES = saved

    def test_fenced_code_is_skipped(self):
        self.assertEqual(quoted("```\n`gone/file.md`\n```\n"), [])

    def test_records_are_skipped(self):
        for rel in ("work/0009-x/plan.md", "DECISIONS.md", "lib/DECISIONS.md", "BACKLOG.md"):
            self.assertEqual(quoted("`gone/file.md`", rel=rel), [], rel)

    def test_a_records_prefix_added_to_the_setting_is_skipped(self):
        # One setting for both checks: a repo exempts its dated records once.
        saved = refs.CITE_RECORDS
        refs.CITE_RECORDS = saved + ("journal/",)
        try:
            self.assertEqual(quoted("`gone/file.md`", rel="journal/2026-10-01.md"), [])
        finally:
            refs.CITE_RECORDS = saved
        self.assertEqual(len(quoted("`gone/file.md`", rel="journal/2026-10-01.md")), 1)


class Cites(unittest.TestCase):
    def test_an_open_item_passes_bare_or_with_its_slug(self):
        self.assertEqual(cites("(work/0021 stage 3) and work/0021-some-slug/"), [])

    def test_a_closed_item_fails(self):
        self.assertEqual(len(cites("measured in work/0014")), 1)

    def test_records_may_name_closed_items(self):
        for rel in ("work/LEDGER.md", "DECISIONS.md", "lib/DECISIONS.md",
                    "BACKLOG.md", "lib/CHANGELOG.md"):
            self.assertEqual(cites("work/0014", rel=rel), [], rel)

    def test_only_a_work_cite_is_matched(self):
        # A path segment that merely ends in work/ is not a cite of this repo's work/.
        self.assertEqual(cites("homework/0014 and other-work/0003 and work/00145"), [])


def long_doc(heads=("Install", "Use", "Config", "Debug", "Notes"), contents=None,
             pad=0) -> str:
    """A document over the threshold: `pad` filler lines after the sections. With
    `contents` None the list is the right one for `heads`."""
    if contents is None:
        contents = [refs.slug(h) for h in heads]
    out = ["# Title", "", "Intro.", "", "## Contents", ""]
    out += [f"- [{c}](#{c})" for c in contents]
    out += [""]
    for h in heads:
        out += [f"## {h}", "", "text", ""]
    out += ["filler"] * (pad or 400)
    return "\n".join(out) + "\n"


def contents(text: str, rel: str = "app/README.md") -> list[str]:
    fail: list[str] = []
    refs.check_contents(rel, text, fail)
    return fail


class Contents(unittest.TestCase):
    def test_a_matching_list_passes(self):
        self.assertEqual(contents(long_doc()), [])

    def test_a_long_file_without_one_fails_naming_the_file(self):
        text = long_doc().replace("## Contents", "## Overview")
        (msg,) = contents(text)
        self.assertIn("app/README.md", msg)
        self.assertIn("no `## Contents`", msg)

    def test_a_list_below_the_first_40_lines_fails(self):
        text = long_doc().replace("Intro.", "Intro.\n" * 45)
        self.assertEqual(len(contents(text)), 1)

    def test_a_heading_the_list_lacks_fails(self):
        (msg,) = contents(long_doc(contents=["install", "use", "config", "debug"]))
        self.assertIn("missing ['notes']", msg)

    def test_a_link_to_no_heading_fails(self):
        (msg,) = contents(long_doc(contents=["install", "use", "config", "debug",
                                             "notes", "gone"]))
        self.assertIn("extra ['gone']", msg)

    def test_a_renamed_heading_fails(self):
        text = long_doc().replace("## Debug", "## Debugging")
        (msg,) = contents(text)
        self.assertIn("missing ['debugging']", msg)
        self.assertIn("extra ['debug']", msg)

    def test_out_of_order_fails(self):
        (msg,) = contents(long_doc(contents=["use", "install", "config", "debug", "notes"]))
        self.assertIn("out of document order", msg)

    def test_punctuation_in_a_heading_follows_github_anchors(self):
        heads = ("Run `just x`", "Why (and when)", "A/B", "Four", "Five")
        text = long_doc(heads=heads, contents=["run-just-x", "why-and-when", "ab",
                                               "four", "five"])
        self.assertEqual(contents(text), [])

    def test_under_both_thresholds_is_ignored(self):
        self.assertEqual(contents(long_doc(pad=1).replace("## Contents", "## Overview")), [])

    def test_size_alone_triggers_it(self):
        wide = long_doc(pad=1).replace("text", "x" * 5000).replace("## Contents", "## Overview")
        self.assertEqual(len(contents(wide)), 1)

    def test_fewer_than_five_sections_is_ignored(self):
        text = long_doc(heads=("One", "Two", "Three", "Four")).replace("## Contents\n", "")
        self.assertEqual(contents(text), [])

    def test_default_exempt_paths_are_ignored(self):
        text = long_doc().replace("## Contents", "## Overview")
        for rel in ("work/0009-x/plan.md", "work/LEDGER.md", "BACKLOG.md", "DECISIONS.md",
                    "lib/DECISIONS.md", "lib/CHANGELOG.md"):
            self.assertEqual(contents(text, rel=rel), [], rel)

    def test_an_exempt_prefix_setting_is_honoured(self):
        text = long_doc().replace("## Contents", "## Overview")
        self.assertEqual(len(contents(text, rel="vendor/x/README.md")), 1)
        saved = refs.CONTENTS_EXEMPT_PREFIXES
        try:
            refs.CONTENTS_EXEMPT_PREFIXES = ("vendor/",)
            self.assertEqual(contents(text, rel="vendor/x/README.md"), [])
        finally:
            refs.CONTENTS_EXEMPT_PREFIXES = saved

    def test_headings_inside_a_code_fence_are_not_sections(self):
        fenced = "```\n" + "".join(f"## Example {i}\n" for i in range(6)) + "```\n"
        text = long_doc(heads=("Install", "Use", "Config", "Debug", "Notes"))
        text = text.replace("filler", fenced, 1)
        self.assertEqual(contents(text), [])
        bare = "# T\n\n" + fenced + "filler\n" * 400
        self.assertEqual(contents(bare), [])


class OpenItems(unittest.TestCase):
    def test_no_work_dir_means_no_open_item(self):
        saved = refs.ROOT
        try:
            refs.ROOT = Path(__file__).resolve().parent / "no-such-dir"
            self.assertEqual(refs.open_items(), set())
        finally:
            refs.ROOT = saved


def tokens(text: str, rel: str) -> list[str]:
    fail: list[str] = []
    refs.check_tokens(rel, text, fail)
    return fail


class ThisRepo(unittest.TestCase):
    """One case per setting this repo fills in refs.py."""

    def test_the_sandbox_script_is_foreign(self):
        self.assertEqual(quoted("(or via `scripts/with-egress.sh`)"), [])

    def test_history_and_the_journal_may_name_old_paths(self):
        for rel in ("docs/history/2026-05-refactor-assessment.md",
                    "docs/math-learning-notes.md"):
            self.assertEqual(quoted("`gone/file.md`", rel=rel), [], rel)
            self.assertEqual(cites("work/0014", rel=rel), [], rel)

    def test_history_and_the_journal_need_no_contents(self):
        text = long_doc().replace("## Contents", "## Overview")
        for rel in ("docs/history/2026-05-refactor-assessment.md",
                    "docs/math-learning-notes.md"):
            self.assertEqual(contents(text, rel=rel), [], rel)
        self.assertEqual(len(contents(text, rel="docs/tests-math.md")), 1)

    def test_the_old_decision_directory_is_a_dead_token(self):
        self.assertEqual(len(tokens("See `docs/decisions/0008-x.md`.", "src/shrec/x.py")), 1)
        self.assertEqual(tokens("under `docs/decisions/`", "DECISIONS.md"), [])

    def test_generated_tree_is_skipped(self):
        every = ["docs/spbuild/html/index.html", "src/a.py"]
        self.assertEqual(refs.tracked(every), ["src/a.py"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
