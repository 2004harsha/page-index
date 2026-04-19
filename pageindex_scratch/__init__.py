"""
pageindex_scratch — PageIndex pipeline built from scratch.

A learning implementation of the VectifyAI PageIndex system:
  https://github.com/VectifyAI/PageIndex

Stages:
  1. ingestion.py     → Document → list[PageContent]
  2. toc_detector.py  → detect existing ToC
  3. tree_builder.py  → build hierarchical DocumentIndex
  4. summarizer.py    → add summaries to each TreeNode
  5. (storage)        → DocumentIndex.save() / .load()
  6. retriever.py     → agentic tree search → RetrievalState
  8. generator.py     → RetrievalState → answer string
  pipeline.py         → orchestrates all stages
"""

from .models import PageContent, Document, TreeNode, DocumentIndex, RetrievalState
from .pipeline import PageIndexPipeline
from .ingestion import ingest, create_test_document

__all__ = [
    "PageContent",
    "Document",
    "TreeNode",
    "DocumentIndex",
    "RetrievalState",
    "PageIndexPipeline",
    "ingest",
    "create_test_document",
]
