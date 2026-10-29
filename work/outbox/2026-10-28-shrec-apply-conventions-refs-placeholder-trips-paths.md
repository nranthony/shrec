# Skill feedback: apply-conventions — template refs.py's own comment fails its paths check once `work/` is in REPO_DIRS

- From: shrec (2026-10-28)
- Owner: nranthony/the-vault, from `repository` in the plugin's `.claude-plugin/plugin.json`
- Version: `myconv 0.16.0 skill:cced494f6e37` (apply-conventions/VERSION)
- Install mode: vendored container copy (seeded at `~/.claude/skills/myconv/`, plugin tree)
- Invocation: `/myconv:migrate-conventions` (no arguments), adding `scripts/refs.py` from `apply-conventions/templates/`
- Artifact: template (`templates/scripts/refs.py`)
- Broke at: migrate-conventions "3. The additive commit (nothing deleted)": "`scripts/refs.py` and `scripts/refs_test.py` from the templates, settings filled in"
- Frequency: every run (any repo that fills REPO_DIRS with `work/`)
- Symptom: none-fit:self-flagging-template
- Verdict: conditional on REPO_DIRS containing `"work/"`, which the template's own commented example (`# "src/", "tests/", "scripts/", "docs/", "work/",`) suggests; this repo has a work tier, so it did
- Risk class: mechanical
- Workaround applied locally: rewrote the comment's placeholder as `work/<NNNN>`, which `check_paths` skips on `<`

With `REPO_DIRS = (..., "work/")`, `python3 scripts/refs.py` fails on the template itself:

```
scripts/refs.py:87: path -> work/NNNN
```

The comment above `CITE_RECORDS` reads "Everything else that cites `work/NNNN` must name an open one." `check_paths` skips candidates containing `*` or `<`, ending in `/`, or matching `PROVENANCE` (`^work/\d{4}$`), but not the literal placeholder `NNNN`. `CITE_RECORDS` and `TOKEN_ALLOW` exempt `scripts/refs.py` from the cite and token checks, not from the comment-path check. `refs_test.py` passes, because no case runs `check_paths` over refs.py's own source.

## Proposed edit

In `templates/scripts/refs.py`, the comment above `CITE_RECORDS`:

```diff
-# since moved, on purpose. Everything else that cites `work/NNNN` must name an
+# since moved, on purpose. Everything else that cites `work/<NNNN>` must name an
```

In `templates/scripts/refs_test.py`, a guard so the template cannot trip itself again:

```python
class SelfCheck(unittest.TestCase):
    def test_refs_py_passes_its_own_comment_check_with_work_in_repo_dirs(self):
        saved = refs.REPO_DIRS
        try:
            refs.REPO_DIRS = ("src/", "tests/", "scripts/", "docs/", "work/")
            fail: list[str] = []
            src = Path(refs.__file__).read_text(encoding="utf-8")
            refs.check_paths("scripts/refs.py", src, refs.COMMENTS["#"], fail)
            self.assertEqual(fail, [])
        finally:
            refs.REPO_DIRS = saved
```
