"""
pipeline.py — Full Pipeline Orchestrator

WHAT IT DOES:
  Wires together all 8 stages into a single `PageIndexPipeline` class
  with two high-level methods:
    - build(doc_path)  → builds and persists the index
    - query(question)  → retrieves and answers

This is also the best place to understand how the stages interact
and what data flows between them.

FLOW SUMMARY:

  BUILD:
    Document  →[1. ingest]→  Document
    Document  →[2. toc_detect]→  TocDetectionResult
    (Document, TocDetectionResult)  →[3. build_tree]→  DocumentIndex
    (DocumentIndex, Document)  →[4. summarize]→  DocumentIndex (+ summaries)
    DocumentIndex  →[5. save JSON]→  disk

  QUERY:
    disk  →[5. load JSON]→  DocumentIndex
    (query, DocumentIndex, Document)  →[6. retrieve]→  RetrievalState
    (RetrievalState, DocumentIndex)  →[8. generate]→  answer string
"""

import os
from pathlib import Path

from .ingestion import ingest
from .toc_detector import detect_toc
from .tree_builder import build_tree
from .summarizer import summarize_index_sync
from .retriever import retrieve, explain_retrieval
from .generator import generate_answer
from .models import DocumentIndex, Document
from . import llm_client


class PageIndexPipeline:
    """
    A complete PageIndex pipeline instance.

    Usage:
        pipeline = PageIndexPipeline(model="gpt-4o", index_dir="./indexes")
        pipeline.build("my_report.pdf")
        answer = pipeline.query("my_report.pdf", "What was the revenue growth?")
        print(answer)
    """

    def __init__(
        self,
        model: str = llm_client.DEFAULT_MODEL,
        index_dir: str = "./pageindex_cache",
        toc_check_pages: int = 20,
        max_pages_per_node: int = 10,
        max_iterations: int = 5,
        skip_summarization: bool = False,
    ):
        self.model = model
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.toc_check_pages = toc_check_pages
        self.max_pages_per_node = max_pages_per_node
        self.max_iterations = max_iterations
        self.skip_summarization = skip_summarization

        # Cache loaded documents and indexes in memory
        self._docs: dict[str, Document] = {}
        self._indexes: dict[str, DocumentIndex] = {}

    # ─────────────────────────────────────────────
    # Phase 1: Build the index
    # ─────────────────────────────────────────────

    def build(self, doc_path: str, force_rebuild: bool = False) -> DocumentIndex:
        """
        Build and persist the PageIndex for a document.

        If an index already exists on disk for this document,
        load it instead (unless force_rebuild=True).
        """
        index_path = self._index_path_for(doc_path)

        if not force_rebuild and index_path.exists():
            print(f"[pipeline] Loading existing index from {index_path}")
            index = DocumentIndex.load(str(index_path))
            self._indexes[doc_path] = index
            return index

        # ── Stage 1: Document Ingestion ──────────
        print(f"\n[pipeline] ── Stage 1: Ingesting {doc_path}")
        doc = ingest(doc_path)
        self._docs[doc_path] = doc
        print(f"[pipeline] Loaded {doc.total_pages} pages")

        # ── Stage 2: TOC Detection ───────────────
        print(f"\n[pipeline] ── Stage 2: TOC Detection")
        toc_result = detect_toc(doc, self.toc_check_pages, model=self.model)
        if toc_result.has_toc:
            print(f"[pipeline] Found ToC with {len(toc_result.entries)} entries")
        else:
            print("[pipeline] No ToC found — will use incremental scan")

        # ── Stage 3: Tree Construction ───────────
        print(f"\n[pipeline] ── Stage 3: Tree Construction")
        index = build_tree(
            doc=doc,
            toc_result=toc_result,
            model=self.model,
            max_pages_per_node=self.max_pages_per_node,
        )
        print(f"[pipeline] Built tree with {self._count_nodes(index)} nodes")
        print("\nTree preview:")
        print(index.to_compact_repr())

        # ── Stage 4: Node Summarization ──────────
        if not self.skip_summarization:
            print(f"\n[pipeline] ── Stage 4: Node Summarization")
            index = summarize_index_sync(index, doc, model=self.model)
        else:
            print("[pipeline] Skipping summarization (skip_summarization=True)")

        # ── Stage 5: Persist ─────────────────────
        print(f"\n[pipeline] ── Stage 5: Persisting index to {index_path}")
        index.save(str(index_path))
        self._indexes[doc_path] = index

        print(f"\n[pipeline] Build complete!")
        return index

    # ─────────────────────────────────────────────
    # Phase 2: Query
    # ─────────────────────────────────────────────

    def query(
        self,
        doc_path: str,
        question: str,
        explain: bool = False,
    ) -> str:
        """
        Answer a question about a document using tree search retrieval.

        Args:
            doc_path: path to the document (must have been built first)
            question: natural language question
            explain: if True, append a retrieval trace to the answer
        """
        # Ensure we have index and doc loaded
        index = self._load_index(doc_path)
        doc = self._load_doc(doc_path)

        print(f"\n[pipeline] Query: '{question}'")

        # ── Stage 6: Tree Search ─────────────────
        print(f"\n[pipeline] ── Stage 6: Tree Search Retrieval")
        state = retrieve(
            query=question,
            index=index,
            doc=doc,
            model=self.model,
            max_iterations=self.max_iterations,
        )

        # ── Stage 8: Answer Generation ───────────
        print(f"\n[pipeline] ── Stage 8: Answer Generation")
        answer = generate_answer(state, index, model=self.model)

        if explain:
            trace = "\n\n---\nRetrieval trace:\n" + explain_retrieval(state, index)
            answer += trace

        return answer

    # ─────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────

    def _index_path_for(self, doc_path: str) -> Path:
        """Derive index JSON path from document path."""
        stem = Path(doc_path).stem
        return self.index_dir / f"{stem}_pageindex.json"

    def _load_index(self, doc_path: str) -> DocumentIndex:
        if doc_path in self._indexes:
            return self._indexes[doc_path]
        index_path = self._index_path_for(doc_path)
        if not index_path.exists():
            raise FileNotFoundError(
                f"No index found for '{doc_path}'. Run pipeline.build('{doc_path}') first."
            )
        index = DocumentIndex.load(str(index_path))
        self._indexes[doc_path] = index
        return index

    def _load_doc(self, doc_path: str) -> Document:
        if doc_path in self._docs:
            return self._docs[doc_path]
        doc = ingest(doc_path)
        self._docs[doc_path] = doc
        return doc

    def _count_nodes(self, index: DocumentIndex) -> int:
        count = 0
        def walk(node):
            nonlocal count
            count += 1
            for child in node.nodes:
                walk(child)
        for root in index.root_nodes:
            walk(root)
        return count
