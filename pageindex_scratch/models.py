"""
models.py — Core data structures for the PageIndex pipeline.

Every stage of the pipeline speaks in these types.
Understanding them is understanding the whole system.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import json


# ─────────────────────────────────────────────────
# 1. Document representation
# ─────────────────────────────────────────────────

@dataclass
class PageContent:
    """
    A single physical page extracted from a document.

    PageIndex wraps each page's text in <physical_index_N> tags
    before sending to the LLM.  That tag is how the LLM knows
    where page N starts and ends — critical for building accurate
    start_index / end_index in each TreeNode.
    """
    page_num: int           # 0-based physical page number
    text: str               # raw extracted text

    def to_tagged_string(self) -> str:
        """
        Wrap text in physical_index tags the way PageIndex does it:
          <physical_index_5>...text...</physical_index_5>
        This is the unit the LLM sees when building the tree.
        """
        n = self.page_num
        return f"<physical_index_{n}>\n{self.text}\n</physical_index_{n}>"


@dataclass
class Document:
    """
    A parsed document — just an ordered list of pages.
    """
    path: str
    pages: list[PageContent] = field(default_factory=list)

    def get_pages(self, start: int, end: int) -> list[PageContent]:
        """Return pages in [start, end) range (end is exclusive)."""
        return [p for p in self.pages if start <= p.page_num < end]

    def to_tagged_text(self, start: int, end: int) -> str:
        """Concatenate tagged pages in a range — what we feed the LLM."""
        return "\n".join(p.to_tagged_string() for p in self.get_pages(start, end))

    @property
    def total_pages(self) -> int:
        return len(self.pages)


# ─────────────────────────────────────────────────
# 2. Tree index node — the heart of PageIndex
# ─────────────────────────────────────────────────

@dataclass
class TreeNode:
    """
    A single node in the hierarchical document tree.

    DESIGN INSIGHT:
      - node_id  → a stable key to look up raw page content later
      - start_index / end_index  → the page range this node covers
      - nodes  → child nodes (recursive; this is what makes it a tree)
      - summary  → optional LLM-generated summary stored AT index time
                   so retrieval can reason over it WITHOUT re-reading pages

    The node represents a *logical* section of the document (chapter,
    sub-section, appendix) not an arbitrary fixed-size chunk.
    That is the fundamental difference from chunking-based RAG.
    """
    node_id: str                            # e.g. "0003"
    title: str                              # section title as it appears in the doc
    start_index: int                        # first page (inclusive)
    end_index: int                          # last page (exclusive)
    summary: Optional[str] = None          # filled in Stage 4
    nodes: list[TreeNode] = field(default_factory=list)   # children

    # ── Helpers ──────────────────────────────────

    def is_leaf(self) -> bool:
        return len(self.nodes) == 0

    def page_count(self) -> int:
        return max(0, self.end_index - self.start_index)

    def all_leaf_nodes(self) -> list[TreeNode]:
        """Flatten the subtree, returning only leaves."""
        if self.is_leaf():
            return [self]
        result = []
        for child in self.nodes:
            result.extend(child.all_leaf_nodes())
        return result

    def find_by_id(self, node_id: str) -> Optional[TreeNode]:
        """Depth-first search for a node by its id."""
        if self.node_id == node_id:
            return self
        for child in self.nodes:
            found = child.find_by_id(node_id)
            if found:
                return found
        return None

    # ── Serialization ────────────────────────────

    def to_dict(self) -> dict:
        d = {
            "node_id": self.node_id,
            "title": self.title,
            "start_index": self.start_index,
            "end_index": self.end_index,
        }
        if self.summary:
            d["summary"] = self.summary
        if self.nodes:
            d["nodes"] = [child.to_dict() for child in self.nodes]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TreeNode:
        return cls(
            node_id=d["node_id"],
            title=d["title"],
            start_index=d["start_index"],
            end_index=d["end_index"],
            summary=d.get("summary"),
            nodes=[cls.from_dict(child) for child in d.get("nodes", [])],
        )

    def to_compact_repr(self, depth: int = 0) -> str:
        """
        Human-readable tree display — also used as the 'index string'
        fed to the LLM during tree search.  It must be compact enough
        to fit in the context window while conveying enough structure
        to let the LLM reason about WHERE to look.
        """
        indent = "  " * depth
        summary_preview = f"  [{self.summary[:60]}...]" if self.summary else ""
        line = f"{indent}[{self.node_id}] {self.title} (pp {self.start_index}-{self.end_index}){summary_preview}"
        child_lines = [child.to_compact_repr(depth + 1) for child in self.nodes]
        return "\n".join([line] + child_lines)


@dataclass
class DocumentIndex:
    """
    The complete index for a document — a list of top-level TreeNodes
    that together cover the entire document.

    This is the object saved to disk as JSON and loaded at query time.
    """
    doc_path: str
    description: Optional[str] = None      # LLM-generated document description
    root_nodes: list[TreeNode] = field(default_factory=list)

    def find_node(self, node_id: str) -> Optional[TreeNode]:
        for root in self.root_nodes:
            found = root.find_by_id(node_id)
            if found:
                return found
        return None

    def to_compact_repr(self) -> str:
        """The full tree as a compact string for the LLM context."""
        header = f"Document: {self.doc_path}"
        if self.description:
            header += f"\nDescription: {self.description}"
        node_strs = [n.to_compact_repr() for n in self.root_nodes]
        return header + "\n\n" + "\n".join(node_strs)

    def to_dict(self) -> dict:
        return {
            "doc_path": self.doc_path,
            "description": self.description,
            "root_nodes": [n.to_dict() for n in self.root_nodes],
        }

    @classmethod
    def from_dict(cls, d: dict) -> DocumentIndex:
        return cls(
            doc_path=d["doc_path"],
            description=d.get("description"),
            root_nodes=[TreeNode.from_dict(n) for n in d.get("root_nodes", [])],
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> DocumentIndex:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ─────────────────────────────────────────────────
# 3. Retrieval state — tracks agentic search
# ─────────────────────────────────────────────────

@dataclass
class RetrievalState:
    """
    Tracks progress of an agentic tree search session.

    The retriever doesn't do one shot — it loops:
      while not enough context:
          pick next node → fetch pages → accumulate → decide
    This object is the loop's memory.
    """
    query: str
    collected_context: list[str] = field(default_factory=list)
    visited_node_ids: list[str] = field(default_factory=list)
    is_sufficient: bool = False
    iteration: int = 0
    max_iterations: int = 5

    def add_context(self, node_id: str, text: str) -> None:
        self.visited_node_ids.append(node_id)
        self.collected_context.append(text)
        self.iteration += 1

    def full_context(self) -> str:
        return "\n\n---\n\n".join(self.collected_context)

    def budget_remaining(self) -> bool:
        return self.iteration < self.max_iterations


# ─────────────────────────────────────────────────
# 4. Node ID counter — simple but important
# ─────────────────────────────────────────────────

class NodeIDCounter:
    """
    Generates sequential 4-digit node IDs: "0001", "0002", ...

    In the real PageIndex these are used as stable keys to map
    node_id → page content.  We keep them zero-padded strings
    (not ints) so they sort lexicographically and stay consistent
    when serialized to JSON.
    """
    def __init__(self, start: int = 1):
        self._counter = start

    def next(self) -> str:
        val = f"{self._counter:04d}"
        self._counter += 1
        return val
