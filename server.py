"""
server.py — MCP server exposing Q&A over the indexed PDFs.

Tools:
  query_documents(question) -> grounded answer + source attribution
  list_documents()          -> what's indexed (doc names, pages, chunk counts)

Ingestion runs once at startup: every PDF in data/pdfs/ is parsed, chunked,
table-normalized (cached), and indexed into the hybrid store.

Run locally:
  python server.py           # stdio transport (what MCP clients expect)

Claude Desktop config (claude_desktop_config.json):
  {
    "mcpServers": {
      "pdf-qa": {
        "command": "/path/to/your/venv/bin/python",
        "args": ["/path/to/project/server.py"]
      }
    }
  }
"""

import os
import sys
import json

from mcp.server.fastmcp import FastMCP

from ingest import build_index
from rag import answer_question

mcp = FastMCP("pdf-qa")
_store = None
_docs: list[dict] = []


def _ingest():
    global _store, _docs
    _store, _docs = build_index()


@mcp.tool()
def query_documents(question: str) -> str:
    """Answer a natural-language question using the indexed PDF documents.

    Returns a JSON object with:
      answer     - grounded answer drawn only from the documents
      answerable - false if the documents don't contain the answer
      sources    - list of {doc_name, page, section} the answer was drawn from
    """
    if _store is None or not _store.chunks:
        return json.dumps({"answer": "No documents are indexed.",
                           "answerable": False, "sources": []})
    result = answer_question(question, _store)
    return json.dumps(result, indent=2)


@mcp.tool()
def list_documents() -> str:
    """List the indexed PDF documents with page and chunk counts."""
    return json.dumps(_docs, indent=2)


if __name__ == "__main__":
    _ingest()
    mcp.run(transport="stdio")
