"""Reproduce the text evidence for the consensus-equation typo (decision 0002).

Appendix B's prose calls the consensus matrix an "elementwise average", but the
printed equation is a bare sum. pypdf keeps a leading `1/` elsewhere in this
paper (it extracts as `1=`), so a typeset `1/K` would have survived extraction.
This prints both passages so a reader can check that without trusting the record.

Written after the 2026-06 audit, whose own extraction was never committed; it
searches every page rather than assuming page numbers.

Usage: uv run python docs/history/2026-06-consensus-typo-extract.py
"""
import re
from pathlib import Path

from pypdf import PdfReader

PDF = Path(__file__).resolve().parents[1] / "PhysRevX.15.011005.pdf"
WINDOW = 160


def show(label, pattern, pages):
    for n, text in enumerate(pages, 1):
        m = re.search(pattern, text)
        if m:
            lo, hi = max(0, m.start() - WINDOW), m.end() + WINDOW
            print(f"--- {label}: PDF page {n}\n{text[lo:hi]!r}\n")
            return
    print(f"--- {label}: not found")


pages = [" ".join(p.extract_text().split()) for p in PdfReader(PDF).pages]
show("consensus prose and equation", r"elementwise\s+average", pages)
show("a leading 1/ that pypdf keeps", r"¼\s*1=", pages)
