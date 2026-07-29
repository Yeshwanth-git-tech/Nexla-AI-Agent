"""
table_normalizer.py — Normalize tables at ingestion time using Claude.

Why: pdfplumber's table engine fails on borderless academic tables, and the
text layer flattens them into whitespace rows that embed poorly. So once, at
indexing time, pages containing a "Table N:" caption are passed to Claude,
which rewrites each table as explicit cell statements:

    Table 2: Linked WikiText-2 Corpus Statistics.
    Documents — Train: 600, Dev: 60, Test: 60
    Tokens — Train: 2,019,195, Dev: 207,982, Test: 236,062
    ...

Each normalized table becomes its own retrieval chunk (section = "Table N,
page P"). Results are cached in data/index/tables_cache.json keyed by a hash
of the page text, so re-indexing is free.
"""

import os
import re
import io
import json
import base64
import hashlib
import anthropic
import pdfplumber

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MODEL = "claude-sonnet-4-6"
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "index", "tables_cache.json")
CAPTION_RE = re.compile(r"Table \d+\s*:")

PROMPT = """This is a page image from an academic paper. Rewrite EVERY table on
this page as explicit cell statements, one line per row:

Table N: <caption text>
<RowLabel> — <ColHeader1>: <value1>, <ColHeader2>: <value2>, ...

Rules:
- Read column boundaries VISUALLY. Spanning headers (one label over several
  sub-columns, e.g. "KGLM" over "Oracle" and "NEL") must be expanded into
  compound names ("KGLM Oracle", "KGLM NEL"). Every data row must yield
  exactly as many values as your header list.
- Preserve values exactly as printed. Empty/unreadable cells: "-".
- Only tables with a "Table N:" caption. Ignore prose, figures, equations.
- Output ONLY the normalized tables. If no table exists, output exactly: NONE"""


def _load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        return json.load(open(CACHE_PATH))
    return {}


def _save_cache(cache: dict):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    json.dump(cache, open(CACHE_PATH, "w"), indent=1)


def _render_page_b64(pdf_path: str, page_number: int, resolution: int = 150) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        img = pdf.pages[page_number - 1].to_image(resolution=resolution)
        buf = io.BytesIO()
        img.original.save(buf, format="PNG")
        return base64.standard_b64encode(buf.getvalue()).decode()


def normalize_tables(pages: list[dict], doc_name: str, pdf_path: str,
                     client: anthropic.Anthropic | None = None) -> list[dict]:
    """Returns extra chunk dicts, one per page that contains tables.
    Sends the RENDERED PAGE IMAGE to Claude so column alignment is read
    visually — flattened text is ambiguous for tables with spanning headers."""
    cache = _load_cache()
    client = client or anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    chunks = []

    for page in pages:
        text = page["text"]
        if not CAPTION_RE.search(text):
            continue
        key = f"{doc_name}:p{page['page']}:v2img:{hashlib.sha256(text.encode()).hexdigest()[:16]}"
        if key in cache:
            normalized = cache[key]
        else:
            img_b64 = _render_page_b64(pdf_path, page["page"])
            resp = client.messages.create(
                model=MODEL, max_tokens=1500,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64",
                     "media_type": "image/png", "data": img_b64}},
                    {"type": "text", "text": PROMPT},
                ]}],
            )
            normalized = resp.content[0].text.strip()
            cache[key] = normalized
            _save_cache(cache)

        if normalized == "NONE":
            continue
        captions = CAPTION_RE.findall(normalized)
        label = ", ".join(dict.fromkeys(c.rstrip(":").strip() for c in captions)) or "Tables"
        chunks.append({
            "id": f"{doc_name}_p{page['page']}_tables",
            "doc_name": doc_name,
            "page": page["page"],
            "section": f"{label} (normalized)",
            "text": normalized,
        })
    return chunks


if __name__ == "__main__":
    import sys
    from parser import parse_pdf
    pages = parse_pdf(sys.argv[1])
    doc = sys.argv[1].split("/")[-1]
    table_chunks = normalize_tables(pages, doc, sys.argv[1])
    for c in table_chunks:
        print(f"=== {c['id']} | {c['section']}")
        print(c["text"][:500], "\n")