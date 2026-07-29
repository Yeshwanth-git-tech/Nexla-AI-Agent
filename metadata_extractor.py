"""
metadata_extractor.py — Extract document-level metadata at ingestion time.

Uses Claude (vision) on page 1 to reliably extract: title, authors, venue,
affiliations. Adds deterministic counts: pages, words, references.

Each document gets one metadata chunk stored alongside text/table chunks,
so questions like "who is the last author" or "how many pages" hit it
directly via retrieval.

Results cached in data/index/metadata_cache.json (same pattern as tables).
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
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "index", "metadata_cache.json")

PROMPT = """This is the FIRST PAGE of an academic paper. Extract the following
metadata and return ONLY a JSON object (no markdown fences):

{
  "title": "full paper title",
  "authors": ["First Last", "First Last", ...],
  "affiliations": ["affiliation 1", ...],
  "venue": "conference or journal name with year, e.g. ACL 2019",
  "emails": ["email1", "email2"]
}

Rules:
- List authors in the EXACT order they appear on the page.
- If venue/conference appears in a footer or header, include it.
- If any field is not visible, use null."""


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


def _count_references(pages: list[dict]) -> int:
    """Count references by detecting the References section and counting
    entries that start with author-name patterns."""
    in_refs = False
    count = 0
    for page in pages:
        text = page["text"]
        if re.search(r"^References\s*$", text, re.MULTILINE):
            in_refs = True
            # count from the line after "References"
            after = text[text.index("References") + len("References"):]
            count += len(re.findall(r"\n[A-Z][a-z]+(?:,|\s+[A-Z])", after))
            continue
        if in_refs:
            count += len(re.findall(r"\n[A-Z][a-z]+(?:,|\s+[A-Z])", text))
    return count


def extract_metadata(pages: list[dict], doc_name: str, pdf_path: str,
                     client: anthropic.Anthropic | None = None) -> list[dict]:
    """Returns a single metadata chunk for the document."""
    cache = _load_cache()
    client = client or anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    full_text = "\n".join(p["text"] for p in pages)
    key = f"{doc_name}:metadata:{hashlib.sha256(pages[0]['text'].encode()).hexdigest()[:16]}"

    if key in cache:
        meta = cache[key]
    else:
        img_b64 = _render_page_b64(pdf_path, 1)
        resp = client.messages.create(
            model=MODEL, max_tokens=500,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                 "media_type": "image/png", "data": img_b64}},
                {"type": "text", "text": PROMPT},
            ]}],
        )
        raw = resp.content[0].text.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            meta = json.loads(raw)
        except json.JSONDecodeError:
            meta = {"title": None, "authors": [], "affiliations": [],
                    "venue": None, "emails": []}
        cache[key] = meta
        _save_cache(cache)

    # Add deterministic counts
    word_count = len(re.findall(r"\b\w+\b", full_text))
    page_count = len(pages)
    ref_count = _count_references(pages)

    # Build a human-readable metadata text block for retrieval
    authors = meta.get("authors") or []
    lines = [
        f"Document: {doc_name}",
        f"Title: {meta.get('title') or 'Unknown'}",
        f"Authors: {', '.join(authors)}",
        f"First author: {authors[0] if authors else 'Unknown'}",
        f"Last author: {authors[-1] if authors else 'Unknown'}",
        f"Number of authors: {len(authors)}",
        f"Affiliations: {', '.join(meta.get('affiliations') or [])}",
        f"All authors same affiliation: {'yes' if len(set(meta.get('affiliations') or [])) <= 1 and meta.get('affiliations') else 'unknown'}",
        f"Venue/Published at: {meta.get('venue') or 'Unknown'}",
        f"Total pages: {page_count}",
        f"Approximate word count: {word_count}",
        f"Approximate number of references: {ref_count}",
    ]
    text = "\n".join(lines)

    return [{
        "id": f"{doc_name}_metadata",
        "doc_name": doc_name,
        "page": 1,
        "section": "Document Metadata",
        "text": text,
    }]


if __name__ == "__main__":
    import sys
    from parser import parse_pdf
    path = sys.argv[1]
    pages = parse_pdf(path)
    doc = path.split("/")[-1]
    chunks = extract_metadata(pages, doc, path)
    for c in chunks:
        print(c["text"])