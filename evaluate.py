"""
evaluate.py — Run the hardcoded gold-standard Q&A set through the pipeline.

Usage:
  python evaluate.py data/P19-1598.pdf [more.pdf ...]

Reports per-question: predicted vs gold answer, sources, token-overlap score.
Final judgment is manual — read the side-by-side.
"""

import sys
import re
import json
import glob
import os

from ingest import build_index
from rag import answer_question

GOLD = [
    {
        "question": "What is the primary challenge addressed by the introduction of the Linked WikiText-2 dataset?",
        "answer": "The primary challenge addressed is incorporating factual knowledge into language models due to difficulty in obtaining training data that describes which entities or facts each token is referring to.",
        "type": "text-only",
    },
    {
        "question": "What is the top-1 accuracy of the Oracle KGLM on birthdate prediction?",
        "answer": "The top-1 accuracy of the Oracle KGLM on birthdate prediction is 65%.",
        "type": "multimodal-t",  # Table 4, page 7
    },
    {
        "question": "How many documents are there in the training set of the Linked WikiText-2 Corpus?",
        "answer": "There are 600 documents in the training set.",
        "type": "multimodal-t",  # Table 2, page 5
    },
    # Fallback-path check: not in any paper, must trigger the unrelated fallback
    {
        "question": "What is the capital of France?",
        "answer": "not available",
        "type": "unanswerable",
    },
    {
        "question": "What is the unknown-penalized perplexity (UPP) of EntityCopyNet on Linked WikiText-2?",
        "answer": "EntityCopyNet achieves an unknown-penalized perplexity of 144.0.",
        "type": "multimodal-t",  # Table 3, page 7
    },
    {
        "question": "What is the vocabulary size of the Linked WikiText-2 corpus?",
        "answer": "The vocabulary size is 33,558.",
        "type": "multimodal-t",  # Table 2, page 5
    },
    {
        "question": "What is GPT-2's top-1 accuracy on the nation-capital relation in fact completion?",
        "answer": "GPT-2 achieves 6% top-1 accuracy on the nation-capital relation.",
        "type": "multimodal-t",  # Table 4, page 7 — spanning-header stress test
    },
    {
        "question": "What is the top-5 accuracy of the NEL KGLM on the spouse relation?",
        "answer": "The NEL KGLM achieves 19% top-5 accuracy on the spouse relation.",
        "type": "multimodal-t",  # Table 4 — hard: NEL column, second value of the pair
    },
    {
        "question": "Which token did GPT-2 predict to complete 'Bob Dylan was born in', and what was the gold answer?",
        "answer": "GPT-2 predicted 'New' while the gold answer was Duluth.",
        "type": "multimodal-t",  # Table 5, page 8 — hard: two cells from one row
    },
    {
        "question": "Which relation is introduced so the model can refer to an entity it has already mentioned?",
        "answer": "A Reflexive relation that self-relates, i.e. p = e for (p, Reflexive, e).",
        "type": "text-only",  # p3, Section 2.2
    },
    {
        "question": "Which method is used to pre-train the entity and relation embeddings, and on what data?",
        "answer": "The embeddings are pre-trained using TransE on Wikidata.",
        "type": "text-only",  # p6, Section 4
    },
    {
        "question": "Which optimizer and learning rate are used to train the KGLM?",
        "answer": "Adam with a learning rate of 1e-3, instead of NT-ASGD, as it was found to be more stable.",
        "type": "text-only",  # p6, Section 5.1
    },
    {
        "question": "What happens to KGLM's predictions for 'Barack Obama was born on' when the birth date in the knowledge graph is changed to 2013-03-21?",
        "answer": "The top three decoded tokens change from 'August', '4', '1961' to 'March', '21', '2013', showing the model's predictions are directly controllable via the knowledge graph.",
        "type": "text-only",  # p8, Effect of changing the KG
    },
    {
        "question": "Which entity linker and coreference tool are used to expand the entity annotations?",
        "answer": "The neural-el entity linker is used to identify additional links to Wikidata, and Stanford CoreNLP is used for coreference resolution.",
        "type": "text-only",  # p4, Section 3
    },
]


# ---------------- metrics ----------------

def _norm_tokens(s: str) -> list[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s.%-]", " ", s)
    return [t for t in s.split() if t not in {"the", "a", "an"}]


def token_f1(pred: str, gold: str) -> float:
    """SQuAD-style token F1: precision penalizes padding, recall penalizes omission."""
    p, g = _norm_tokens(pred), _norm_tokens(gold)
    if not p or not g:
        return 0.0
    common = {}
    for t in p:
        if g.count(t) > common.get(t, 0):
            common[t] = min(p.count(t), g.count(t))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def numeric_match(pred: str, gold: str) -> float | None:
    """For table questions the numbers ARE the answer. Returns None when the
    gold has no numbers (metric not applicable)."""
    nums = lambda s: set(re.findall(r"\d[\d,]*\.?\d*", s))
    g = nums(gold)
    if not g:
        return None
    return 1.0 if g <= nums(pred) else 0.0


def hit_at_k(retrieved: list[dict], gold_pages: set[tuple]) -> float | None:
    """Was any gold-evidence (doc, page) among the retrieved chunks?"""
    if not gold_pages:
        return None
    got = {(r["doc_name"], r["page"]) for r in retrieved}
    return 1.0 if got & gold_pages else 0.0


def find_evidence_pages(evidence: str, doc_texts_pages: dict) -> set[tuple]:
    """Locate the gold evidence quote in the parsed pages -> {(doc, page)}.
    Matches on a normalized 60-char slice from the middle of the quote."""
    ev = re.sub(r"\s+", " ", evidence.replace('\\"', '"')).strip(' ."')
    if len(ev) < 30:
        return set()
    mid = len(ev) // 2
    probe = ev[max(0, mid - 30):mid + 30].lower()
    hits = set()
    for doc, pages in doc_texts_pages.items():
        for pg in pages:
            if probe in re.sub(r"\s+", " ", pg["text"]).lower():
                hits.add((doc, pg["page"]))
    return hits


JUDGE_PROMPT = """Question: {q}
Gold answer: {gold}
Predicted answer: {pred}

Score the predicted answer against the gold answer. Rules:
- If the predicted answer refuses to answer or says "not available" / "can't answer"
  but the gold answer contains a real answer, that is INCORRECT.
- If the predicted answer gives a DIFFERENT number, name, or fact than the gold, INCORRECT.
- If the predicted answer gives the same core fact as gold (possibly with extra
  correct detail or different wording), CORRECT.
- If the predicted answer is partially right but misses the key fact, INCORRECT.

Reply with exactly one word: CORRECT or INCORRECT."""


def llm_judge(client, q: str, gold: str, pred: str) -> float:
    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=5,
        messages=[{"role": "user",
                   "content": JUDGE_PROMPT.format(q=q, gold=gold, pred=pred)}])
    return 1.0 if "CORRECT" in resp.content[0].text.strip().upper() else 0.0


def _load_gold() -> list[dict]:
    """Prefer data/gold/*.jsonl (one file per PDF, official format); fall back
    to the hardcoded GOLD above. The France fallback check is always appended."""
    files = sorted(glob.glob(os.path.join("data", "gold", "*.jsonl")))
    if not files:
        return GOLD
    items = []
    for f in files:
        for line in open(f):
            line = line.strip()
            if line:
                items.append(json.loads(line))
    items.append({"question": "What is the capital of France?",
                  "answer": "not available", "type": "unanswerable"})
    return items


def main(pdf_paths: list[str]):
    import anthropic
    from parser import parse_pdf
    store, docs = build_index(pdf_paths)
    doc_texts_pages = {}
    for p in pdf_paths:
        doc = p.split("/")[-1]
        doc_texts_pages[doc] = parse_pdf(p)
    print(f"Indexed {len(store.chunks)} chunks from {len(pdf_paths)} PDF(s)\n")

    client = anthropic.Anthropic()
    agg = {"hit@8": [], "numeric": [], "f1": [], "judge": []}
    gold = _load_gold()
    for i, item in enumerate(gold, 1):
        res = answer_question(item["question"], store)
        print(f"[{i}] ({item['type']}) Q: {item['question']}")
        if item["type"] == "unanswerable":
            ok = 1.0 if not res["answerable"] else 0.0
            agg["judge"].append(ok)
            print(f"    fallback triggered: {bool(ok)}  ->  {res['answer']}\n")
            continue

        hit = hit_at_k(res.get("retrieved", []),
                       find_evidence_pages(item.get("evidence", ""), doc_texts_pages))
        num = numeric_match(res["answer"], item["answer"])
        f1 = token_f1(res["answer"], item["answer"])
        # Auto-fail: system abstained but gold has a real answer
        if not res.get("answerable", True):
            jdg = 0.0
        else:
            jdg = llm_judge(client, item["question"], item["answer"], res["answer"])
        for key, val in (("hit@8", hit), ("numeric", num), ("f1", f1), ("judge", jdg)):
            if val is not None:
                agg[key].append(val)

        print(f"    gold:      {item['answer']}")
        print(f"    predicted: {res['answer']}")
        print(f"    sources:   {res['sources']}")
        fmt = lambda v: "n/a" if v is None else f"{v:.2f}"
        print(f"    hit@8={fmt(hit)}  numeric={fmt(num)}  f1={f1:.2f}  judge={fmt(jdg)}\n")

    print("=" * 60)
    for key, label in [("hit@8", "Retrieval hit-rate@8"), ("numeric", "Numeric exact-match"),
                       ("f1", "Token F1"), ("judge", "LLM-judge correctness")]:
        vals = agg[key]
        if vals:
            avg = sum(vals) / len(vals)
            print(f"{label:<24} {avg:.2f}  (n={len(vals)})")

    # Summary
    judge_vals = agg["judge"]
    correct = int(sum(judge_vals))
    total = len(gold)
    answerable = sum(1 for g in gold if g["type"] != "unanswerable")
    print("")
    print(f"Correctly answered:   {correct}/{total}")
    print(f"Answerable accuracy:  {correct}/{answerable} ({correct/answerable*100:.0f}%)")


if __name__ == "__main__":
    import glob
    main(sys.argv[1:] or sorted(glob.glob("data/pdfs/*.pdf")))