"""
chunker.py — Recursive chunking with metadata for RAG.

- Target ~512 tokens per chunk (~4 chars/token heuristic -> 2048 chars)
- ~15% overlap between consecutive chunks
- Split priority: paragraph (\n\n) > line (\n) > sentence (". ")
- Every chunk carries: doc_name, page, section (the heading in effect at that point)

Chunks never cross page boundaries — page-accurate attribution is a hard
requirement, and losing a little cross-page continuity is the cheaper cost.
"""

import re

CHUNK_CHARS = 2048     # ~512 tokens
OVERLAP_CHARS = 300    # ~75 tokens
SEPARATORS = ["\n\n", "\n", ". "]


def _split_recursive(text: str, seps: list[str]) -> list[str]:
    """Split text into pieces no larger than CHUNK_CHARS using separator hierarchy."""
    if len(text) <= CHUNK_CHARS:
        return [text]
    if not seps:
        # Hard split as last resort
        return [text[i:i + CHUNK_CHARS] for i in range(0, len(text), CHUNK_CHARS)]

    sep, rest = seps[0], seps[1:]
    parts = text.split(sep)
    pieces, buf = [], ""
    for part in parts:
        candidate = (buf + sep + part) if buf else part
        if len(candidate) <= CHUNK_CHARS:
            buf = candidate
        else:
            if buf:
                pieces.append(buf)
            buf = part if len(part) <= CHUNK_CHARS else ""
            if len(part) > CHUNK_CHARS:
                pieces.extend(_split_recursive(part, rest))
    if buf:
        pieces.append(buf)
    return pieces


def _add_overlap(pieces: list[str]) -> list[str]:
    out = []
    for i, p in enumerate(pieces):
        if i > 0 and OVERLAP_CHARS > 0:
            tail = pieces[i - 1][-OVERLAP_CHARS:]
            # Snap overlap to a word boundary
            tail = tail[tail.find(" ") + 1:]
            p = tail + " " + p
        out.append(p)
    return out


def chunk_pages(pages: list[dict], doc_name: str) -> list[dict]:
    """pages: output of parser.parse_pdf(). Returns chunk dicts with metadata."""
    chunks = []
    current_section = "Preamble"
    for page in pages:
        # Sections in effect: last section seen carries into pages with none
        page_sections = page["sections"]
        text = page["text"]
        if not text:
            continue

        # Map char offsets -> section, so each chunk gets the right heading
        boundaries = []  # (offset, section_name)
        for sec in page_sections:
            idx = text.find(sec)
            if idx >= 0:
                boundaries.append((idx, sec))
        boundaries.sort()

        pieces = _add_overlap(_split_recursive(text, SEPARATORS))

        offset = 0
        for piece in pieces:
            # Locate piece start (approx: search from last offset)
            idx = text.find(piece[:80], max(0, offset - OVERLAP_CHARS - 50))
            if idx < 0:
                idx = offset
            offset = idx + len(piece)

            sec = current_section
            for b_off, b_name in boundaries:
                if b_off <= idx:
                    sec = b_name
            chunks.append({
                "id": f"{doc_name}_p{page['page']}_c{len(chunks)}",
                "doc_name": doc_name,
                "page": page["page"],
                "section": sec,
                "text": piece.strip(),
            })
        if boundaries:
            current_section = boundaries[-1][1]
    return chunks


if __name__ == "__main__":
    import sys
    from parser import parse_pdf
    pages = parse_pdf(sys.argv[1])
    chunks = chunk_pages(pages, doc_name=sys.argv[1].split("/")[-1])
    print(f"{len(chunks)} chunks")
    for c in chunks:
        print(f"  {c['id']:<28} p{c['page']} | {c['section'][:40]:<40} | {len(c['text'])} chars")
    print("\n=== SAMPLE CHUNK ===")
    sample = chunks[len(chunks) // 2]
    print(f"[{sample['doc_name']} | page {sample['page']} | {sample['section']}]")
    print(sample["text"][:600])
