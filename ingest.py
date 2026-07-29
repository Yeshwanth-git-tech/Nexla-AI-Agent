"""
ingest.py — Build the hybrid index from all PDFs in data/pdfs/.

Standalone entry point for ingestion:
  python ingest.py

Does three things:
  1. Parses all PDFs (column-aware text extraction)
  2. Chunks text + normalizes tables (vision-based, cached)
  3. Returns a ready-to-query HybridStore with all chunks indexed

Also used as a module by evaluate.py and server.py:
  from ingest import build_index
  store, docs = build_index()
"""

import os
import sys
import glob

from parser import parse_pdf
from chunker import chunk_pages
from table_normalizer import normalize_tables
from metadata_extractor import extract_metadata
from store import HybridStore

PDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pdfs")


def build_index(pdf_paths: list[str] | None = None, clear: bool = False) -> tuple:
    """Parse, chunk, normalize tables, and index all PDFs.

    Args:
        pdf_paths: explicit list of PDF paths. If None, uses data/pdfs/*.pdf.

    Returns:
        (store, docs) where:
          store  — HybridStore with all chunks indexed, ready for search
          docs   — list of dicts with per-document metadata
    """
    if pdf_paths is None:
        pdf_paths = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))

    if not pdf_paths:
        print(f"WARNING: no PDFs found in {PDF_DIR}", file=sys.stderr)

    store = HybridStore()
    if clear:
        try:
            store.client.delete_collection("papers")
            store.col = store.client.get_or_create_collection(
                "papers", metadata={"hnsw:space": "cosine"})
        except Exception:
            pass

    # If store already has data and no clear requested, rebuild only
    # BM25 and doc_texts (in-memory) from the PDFs, skip embedding
    existing_count = store.col.count()
    if existing_count > 0 and not clear:
        print(f"vector store already has {existing_count} chunks, "
              f"rebuilding in-memory indexes only", file=sys.stderr)
        all_chunks = []
        doc_texts = {}
        docs = []
        for path in pdf_paths:
            doc = os.path.basename(path)
            pages = parse_pdf(path)
            chunks = chunk_pages(pages, doc)
            table_chunks = normalize_tables(pages, doc, path)
            meta_chunks = extract_metadata(pages, doc, path)
            all_chunks += chunks + table_chunks + meta_chunks
            doc_texts[doc] = "\n".join(pg["text"] for pg in pages)
            docs.append({"doc_name": doc, "pages": len(pages),
                         "text_chunks": len(chunks),
                         "table_chunks": len(table_chunks),
                         "meta_chunks": len(meta_chunks)})
        store.chunks = all_chunks
        store.doc_texts = doc_texts
        from rank_bm25 import BM25Okapi
        from store import _tokenize
        store.bm25 = BM25Okapi([_tokenize(c["text"]) for c in all_chunks])
        print(f"index ready: {len(all_chunks)} chunks from {len(pdf_paths)} PDFs",
              file=sys.stderr)
        return store, docs

    print(f"vector store path: {store.persist_dir}", file=sys.stderr)
    all_chunks = []
    doc_texts = {}
    docs = []

    for path in pdf_paths:
        doc = os.path.basename(path)
        pages = parse_pdf(path)
        chunks = chunk_pages(pages, doc)
        table_chunks = normalize_tables(pages, doc, path)
        meta_chunks = extract_metadata(pages, doc, path)
        all_chunks += chunks + table_chunks + meta_chunks
        doc_texts[doc] = "\n".join(pg["text"] for pg in pages)
        docs.append({
            "doc_name": doc,
            "pages": len(pages),
            "text_chunks": len(chunks),
            "table_chunks": len(table_chunks),
            "meta_chunks": len(meta_chunks),
        })
        print(f"ingested {doc}: {len(pages)} pages, "
              f"{len(chunks)} text + {len(table_chunks)} table + {len(meta_chunks)} meta chunks",
              file=sys.stderr)

    if all_chunks:
        store.index(all_chunks, doc_texts)
    print(f"index ready: {len(all_chunks)} total chunks from "
          f"{len(pdf_paths)} PDFs", file=sys.stderr)

    return store, docs


if __name__ == "__main__":
    clear = "--clear" in sys.argv
    paths = [a for a in sys.argv[1:] if a != "--clear"] or None
    store, docs = build_index(paths, clear=clear)
    print(f"\n{len(store.chunks)} chunks indexed across {len(docs)} documents:")
    for d in docs:
        print(f"  {d['doc_name']}: {d['pages']} pages, "
              f"{d['text_chunks']} text + {d['table_chunks']} table chunks")