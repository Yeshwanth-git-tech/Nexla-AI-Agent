"""
rag.py — Retrieval + answer generation with Claude.

Flow:
  1. Hybrid search (store.py) -> top-k chunks
  2. Gate 1: if best dense similarity < SIM_THRESHOLD, don't call the LLM.
     Return "not available" + pointer to the closest section.
  3. Gate 2: Claude is instructed to return answerable=false if the context
     doesn't actually contain the answer -> same fallback.
  4. Otherwise: grounded answer + sources [{doc, page, section}].

Requires: ANTHROPIC_API_KEY env var.
"""

import os
import re
import json
import anthropic
from dotenv import load_dotenv

load_dotenv()  # reads ANTHROPIC_API_KEY from a .env file in the project root

SIM_THRESHOLD = 0.35   # tune against gold set in evaluate.py
MODEL = "claude-sonnet-4-6"

SYSTEM = """You answer questions strictly from the provided document excerpts.

Rules:
- Use ONLY the excerpts. No outside knowledge.
- Excerpts come from PDFs of academic papers, so tables appear FLATTENED as
  whitespace-separated rows. The first such line is usually the column header
  (e.g. "Train Dev Test") and following lines are "RowLabel val1 val2 val3" —
  map values to columns by position. A nearby "Table N: ..." caption names the
  table. Treat these flattened tables as valid, citable evidence.
- Answer when the excerpts contain sufficient evidence, even if the phrasing
  differs from the question. Set "answerable" to false ONLY when the needed
  information is genuinely absent.
- Answer in 1-3 complete sentences, restating what the value refers to.
- Be direct. When an excerpt explicitly states the answer (e.g. the question
  asks for "the primary challenge" and an excerpt says "the primary barrier
  is..."), answer with that excerpt's framing instead of synthesizing across
  all excerpts. Broad synthesis is only for questions no single excerpt answers.
- Cite which excerpt(s) support the answer by their [n] markers.

Respond with ONLY a JSON object, no markdown fences:
{"answerable": true|false, "answer": "...", "used_excerpts": [1, 2]}"""


def _format_context(chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(
            f"[{i}] (doc: {c['doc_name']}, page {c['page']}, section: {c['section']})\n{c['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def _fallback(chunks: list[dict], reason: str) -> dict:
    """reason: 'unrelated' (low retrieval confidence) or 'not_found' (good
    retrieval, but the answer is absent from the documents)."""
    best = chunks[0] if chunks else None
    if reason == "unrelated" or not best:
        return {
            "answer": ("This question does not appear to be related to the "
                       "indexed documents, so I can't answer it from them."),
            "answerable": False,
            "sources": [],
        }
    return {
        "answer": (f"Sorry, the answer is not available in the indexed documents. "
                   f"The closest matching section is '{best['section']}' in "
                   f"{best['doc_name']} (page {best['page']}) — you may want to check there."),
        "answerable": False,
        "sources": [{"doc_name": best["doc_name"], "page": best["page"],
                     "section": best["section"]}],
    }


COUNT_TRIGGER_RE = re.compile(
    r"how many time|number of times|how often", re.IGNORECASE)
TERM_AFTER_MENTION_RE = re.compile(
    r"mentions?\s+[\"']?(.+?)[\"']?\s*\??\s*$", re.IGNORECASE)
TERM_BEFORE_VERB_RE = re.compile(
    r"(?:does|do|is)\s+[\"']?(.+?)[\"']?\s+(?:appear|occur|mentioned)", re.IGNORECASE)


def _count_route(question: str, store) -> dict | None:
    """Deterministic path for 'how many times is X mentioned' questions.
    Retrieval cannot count across a whole document; regex over the full
    parsed text can. Hyphens are optional in the pattern because the PDF
    cleaner de-hyphenates line breaks (WikiText-\n2 -> WikiText2)."""
    if not COUNT_TRIGGER_RE.search(question) or not getattr(store, "doc_texts", None):
        return None
    m = TERM_AFTER_MENTION_RE.search(question) or TERM_BEFORE_VERB_RE.search(question)
    if not m:
        return None
    term = m.group(1).strip().strip('.?"\'')
    pattern = re.escape(term).replace(r"\-", "-?")
    per_doc = {}
    for doc, text in store.doc_texts.items():
        n = len(re.findall(pattern, text, re.IGNORECASE))
        if n:
            per_doc[doc] = n
    total = sum(per_doc.values())
    if len(per_doc) <= 1:
        doc = next(iter(per_doc), None)
        answer = (f"The term '{term}' appears {total} times"
                  + (f" in {doc}." if doc else " in the indexed documents."))
    else:
        parts = ", ".join(f"{d}: {n}" for d, n in per_doc.items())
        answer = f"The term '{term}' appears {total} times across the indexed documents ({parts})."
    return {
        "answer": answer,
        "answerable": total > 0,
        "sources": [{"doc_name": d, "page": None, "section": f"full-text count ({n} occurrences)"}
                    for d, n in per_doc.items()],
    }


METADATA_TRIGGERS = re.compile(
    r"who.*(author|wrote|write)|last author|first author|"
    r"how many (authors|pages|words|references)|"
    r"(affilia|same (affilia|institu|organi))|"
    r"where.*(publish|appear|present)|"
    r"which (conference|journal|venue|workshop)|"
    r"(published|presented) (at|in)|"
    r"what.*(venue|conference|journal)",
    re.IGNORECASE)


def _metadata_route(question, store, client):
    """For metadata questions, search only metadata chunks, bypass sim gate."""
    if not METADATA_TRIGGERS.search(question):
        return None
    meta_chunks = [c for c in store.chunks if c.get("section") == "Document Metadata"]
    if not meta_chunks:
        return None
    for mc in meta_chunks:
        if mc["doc_name"].replace(".pdf", "").lower() in question.lower():
            meta_chunks = [mc]
            break
    context_blocks = []
    for i, c in enumerate(meta_chunks, 1):
        header = "[" + str(i) + "] (doc: " + c["doc_name"] + ")"
        context_blocks.append(header + "\n" + c["text"])
    context = "\n\n---\n\n".join(context_blocks)
    resp = client.messages.create(
        model=MODEL, max_tokens=500, system=SYSTEM,
        messages=[{"role": "user",
                   "content": "Excerpts:\n\n" + context + "\n\nQuestion: " + question}],
    )
    raw = resp.content[0].text.strip()
    parsed = None
    for m in re.finditer(r"\{.*?\}(?=[^{}]*$)|\{.*\}", raw, re.DOTALL):
        try:
            cand = json.loads(m.group())
            if "answer" in cand:
                parsed = cand
        except json.JSONDecodeError:
            continue
    if parsed is None:
        parsed = {"answerable": True, "answer": raw,
                  "used_excerpts": list(range(1, len(meta_chunks) + 1))}
    if not parsed.get("answerable", False):
        # Claude was unsure (e.g. "the paper" is ambiguous across 5 PDFs).
        # Instead of falling through, list all papers' relevant metadata.
        all_lines = []
        for c in meta_chunks:
            all_lines.append(c["doc_name"] + ":\n" + c["text"])
        combined = "\n\n".join(all_lines)
        # Retry with explicit instruction to list all
        resp2 = client.messages.create(
            model=MODEL, max_tokens=500,
            messages=[{"role": "user",
                       "content": ("The user asked: " + question +
                                   "\n\nSince multiple documents are indexed and the question "
                                   "does not specify which one, list the answer for ALL documents "
                                   "below.\n\n" + combined +
                                   "\n\nProvide a concise answer covering all documents. "
                                   "Respond with ONLY a JSON object: "
                                   '{"answerable": true, "answer": "...", "used_excerpts": [1]}')}],
        )
        raw2 = resp2.content[0].text.strip()
        parsed2 = None
        for m2 in re.finditer(r"\{.*?\}(?=[^{}]*$)|\{.*\}", raw2, re.DOTALL):
            try:
                cand2 = json.loads(m2.group())
                if "answer" in cand2:
                    parsed2 = cand2
            except json.JSONDecodeError:
                continue
        if parsed2 and parsed2.get("answerable", False):
            sources = [{"doc_name": c["doc_name"], "page": 1,
                        "section": "Document Metadata"} for c in meta_chunks]
            return {"answer": parsed2["answer"], "answerable": True,
                    "sources": sources, "retrieved": []}
        # If still fails, return all metadata as-is
        sources = [{"doc_name": c["doc_name"], "page": 1,
                    "section": "Document Metadata"} for c in meta_chunks]
        return {"answer": combined, "answerable": True,
                "sources": sources, "retrieved": []}
    used = parsed.get("used_excerpts") or list(range(1, len(meta_chunks) + 1))
    sources = []
    for i in used:
        if 1 <= i <= len(meta_chunks):
            c = meta_chunks[i - 1]
            sources.append({"doc_name": c["doc_name"], "page": 1,
                            "section": "Document Metadata"})
    return {"answer": parsed["answer"], "answerable": True,
            "sources": sources, "retrieved": []}


def answer_question(question: str, store, k: int = 8,
                    client: anthropic.Anthropic | None = None) -> dict:
    counted = _count_route(question, store)
    if counted is not None:
        counted["retrieved"] = []
        return counted

    client = client or anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    meta_result = _metadata_route(question, store, client)
    if meta_result is not None:
        return meta_result

    chunks = store.search(question, k=k)
    retrieved = [{"doc_name": c["doc_name"], "page": c["page"]} for c in chunks]
    if os.environ.get("RAG_DEBUG"):
        for c in chunks:
            print(f"  [retrieved] {c['id']} sim={c['dense_sim']} rrf={c['rrf_score']}")

    # Gate 1: retrieval confidence
    if not chunks or chunks[0]["dense_sim"] < SIM_THRESHOLD:
        out = _fallback(chunks, "unrelated")
        out["retrieved"] = retrieved
        return out

    resp = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Excerpts:\n\n{_format_context(chunks)}\n\nQuestion: {question}",
        }],
    )
    raw = resp.content[0].text.strip()
    # Claude sometimes prepends prose despite instructions — extract the last
    # JSON object rather than requiring the whole response to be JSON.
    parsed = None
    for m in re.finditer(r"\{.*?\}(?=[^{}]*$)|\{.*\}", raw, re.DOTALL):
        try:
            cand = json.loads(m.group())
            if "answer" in cand:
                parsed = cand
        except json.JSONDecodeError:
            continue
    if parsed is None:
        parsed = {"answerable": True, "answer": raw,
                  "used_excerpts": list(range(1, len(chunks) + 1))}

    # Gate 2: LLM abstention
    if not parsed.get("answerable", False):
        out = _fallback(chunks, "not_found")
        out["retrieved"] = retrieved
        return out

    used = parsed.get("used_excerpts") or list(range(1, len(chunks) + 1))
    sources, seen = [], set()
    for i in used:
        if 1 <= i <= len(chunks):
            c = chunks[i - 1]
            key = (c["doc_name"], c["page"], c["section"])
            if key not in seen:
                seen.add(key)
                sources.append({"doc_name": c["doc_name"], "page": c["page"],
                                "section": c["section"]})
    return {"answer": parsed["answer"], "answerable": True, "sources": sources,
            "retrieved": retrieved}


if __name__ == "__main__":
    from parser import parse_pdf
    from chunker import chunk_pages
    from store import HybridStore

    pages = parse_pdf("data/P19-1598.pdf")
    store = HybridStore()
    store.index(chunk_pages(pages, "P19-1598.pdf"))

    for q in [
        "What perplexity does KGLM achieve compared to AWD-LSTM?",
        "What is the capital of France?",
    ]:
        print(f"\nQ: {q}")
        print(json.dumps(answer_question(q, store), indent=2))
