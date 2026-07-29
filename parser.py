"""
parser.py — Column-aware PDF parsing for two-column academic papers (ACL/arXiv style).

Why not plain pdfplumber extract_text()?
Two-column papers get interleaved (left line, right line, left line...) which
destroys reading order. Fix: split each page at the vertical midline, extract
left column then right column.

Output: list of page dicts:
    {"page": int, "text": str, "sections": [str, ...]}
where sections = numbered headings detected on that page (e.g. "2.1 Problem Setup").
"""

import re
import pdfplumber

# Numbered section headings like "2 Knowledge Graph Language Model" or "5.1 Evaluation Setup"
SECTION_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+([A-Z][^\n]{2,80})$")


def _extract_columns(page, gutter_tolerance: float = 6.0) -> str:
    """Extract text from a two-column page in correct reading order."""
    mid = page.width / 2
    left = page.crop((0, 0, mid + gutter_tolerance, page.height))
    right = page.crop((mid - gutter_tolerance, 0, page.width, page.height))
    left_text = left.extract_text(x_tolerance=1.5) or ""
    right_text = right.extract_text(x_tolerance=1.5) or ""
    return (left_text + "\n" + right_text).strip()


def _is_single_column(page) -> bool:
    """Heuristic: if many words cross the midline, treat the page as single-column."""
    mid = page.width / 2
    words = page.extract_words()
    if not words:
        return True
    crossing = sum(1 for w in words if w["x0"] < mid - 20 and w["x1"] > mid + 20)
    return crossing / len(words) > 0.10


def _clean(text: str) -> str:
    # De-hyphenate line breaks: "lan-\nguage" -> "language"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Drop page-number-only lines
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
    # Collapse 3+ newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _find_sections(text: str) -> list[str]:
    sections = []
    for line in text.split("\n"):
        m = SECTION_RE.match(line.strip())
        if m:
            # Reject dates ("21 April 1989") and copyright lines ("2019 Association...")
            if int(m.group(1).split(".")[0]) > 20:
                continue
            sections.append(f"{m.group(1)} {m.group(2).strip()}")
    return sections


def parse_pdf(path: str) -> list[dict]:
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            if _is_single_column(page):
                text = page.extract_text(x_tolerance=1.5) or ""
            else:
                text = _extract_columns(page)
            text = _clean(text)
            pages.append({
                "page": i,
                "text": text,
                "sections": _find_sections(text),
            })
    return pages


if __name__ == "__main__":
    import sys, json
    pages = parse_pdf(sys.argv[1])
    for p in pages:
        print(f"--- page {p['page']} | sections: {p['sections']} | {len(p['text'])} chars")
    # Sample: print first 800 chars of page 2 (dense two-column page) for QA
    print("\n=== PAGE 2 SAMPLE ===")
    print(pages[1]["text"][:800])
