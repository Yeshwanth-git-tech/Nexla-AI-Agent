import os
"""
store.py — Hybrid retrieval index: Chroma (dense, ONNX MiniLM) + BM25 (lexical).

Why hybrid: dense embeddings handle paraphrase ("how well does it do" -> results),
BM25 handles exact terms common in paper Q&A ("KGLM", "perplexity", "AWD-LSTM").
Scores fused with Reciprocal Rank Fusion (RRF) — robust, no score normalization needed.
"""

import re
import chromadb
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridStore:
    DEFAULT_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "vectorstore")

    def __init__(self, persist_dir: str | None = None):
        self.persist_dir = persist_dir or self.DEFAULT_PERSIST_DIR
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.col = self.client.get_or_create_collection(
            "papers", metadata={"hnsw:space": "cosine"}
        )
        self.chunks: list[dict] = []
        self.bm25: BM25Okapi | None = None
        self.doc_texts: dict[str, str] = {}  # doc_name -> full parsed text

    def index(self, chunks: list[dict], doc_texts: dict[str, str] | None = None):
        self.chunks = chunks
        self.doc_texts = doc_texts or {}
        self.col.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{"doc_name": c["doc_name"], "page": c["page"],
                        "section": c["section"]} for c in chunks],
        )
        self.bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])

    def search(self, query: str, k: int = 6, rrf_k: int = 60) -> list[dict]:
        """Hybrid search via Reciprocal Rank Fusion. Returns chunks with fused score
        and the dense cosine similarity of the top dense hit (used for thresholding)."""
        n = len(self.chunks)
        fetch = min(n, max(k * 3, 12))

        # Dense
        dres = self.col.query(query_texts=[query], n_results=fetch)
        dense_ranked = dres["ids"][0]
        dense_sim = {i: 1 - d for i, d in zip(dres["ids"][0], dres["distances"][0])}

        # Lexical
        scores = self.bm25.get_scores(_tokenize(query))
        bm25_ranked = [self.chunks[i]["id"] for i in
                       sorted(range(n), key=lambda i: -scores[i])[:fetch]]

        # RRF fusion
        fused: dict[str, float] = {}
        for ranked in (dense_ranked, bm25_ranked):
            for rank, cid in enumerate(ranked):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)

        # Guaranteed slots: a #1 hit in either ranker must never be squeezed
        # out by chunks that are merely mediocre in both. Reserve top-4 of each.
        guaranteed = list(dict.fromkeys(dense_ranked[:4] + bm25_ranked[:4]))[:k]
        by_rrf = sorted(fused, key=fused.get, reverse=True)
        top = guaranteed + [cid for cid in by_rrf if cid not in guaranteed]
        top = top[:k]

        by_id = {c["id"]: c for c in self.chunks}
        return [{**by_id[cid],
                 "rrf_score": round(fused[cid], 5),
                 "dense_sim": round(dense_sim.get(cid, 0.0), 4)}
                for cid in top]


if __name__ == "__main__":
    from parser import parse_pdf
    from chunker import chunk_pages

    pages = parse_pdf("data/P19-1598.pdf")
    chunks = chunk_pages(pages, "P19-1598.pdf")
    store = HybridStore()
    store.index(chunks)

    for q in ["What perplexity does KGLM achieve on Linked WikiText-2?",
              "How is the dataset annotated with entities?",
              "What is the capital of France?"]:
        print(f"\nQ: {q}")
        for r in store.search(q, k=3):
            print(f"  [{r['dense_sim']:.3f} sim | rrf {r['rrf_score']}] "
                  f"p{r['page']} {r['section'][:35]} :: {r['text'][:90]!r}")
