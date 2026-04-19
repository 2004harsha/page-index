"""
tests.py — Unit tests for every pipeline stage.

Run with:  python -m pytest tests.py -v
  or:      python tests.py

These tests are designed to run WITHOUT an LLM API key by mocking
all LLM calls.  This lets you verify the data structures, algorithms,
and wiring are correct before spending API tokens.

TEST STRATEGY:
  Each stage gets its own test class.
  We test:
    1. Data structures behave correctly (TreeNode, DocumentIndex, etc.)
    2. Stage logic works with mocked LLM responses
    3. Edge cases: empty documents, single-page docs, malformed LLM output
    4. The full pipeline end-to-end with mocked LLM
"""

import json
import sys
import os
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from pageindex_scratch.models import (
    PageContent, Document, TreeNode, DocumentIndex,
    RetrievalState, NodeIDCounter
)
from pageindex_scratch.ingestion import create_test_document, _clean_pdf_text
from pageindex_scratch.toc_detector import (
    TocDetectionResult, toc_entries_to_tree, _parse_detection_result
)
from pageindex_scratch.tree_builder import (
    _merge_trees, _dicts_to_tree_nodes, _build_incremental_prompt
)
from pageindex_scratch.summarizer import _collect_all_nodes
from pageindex_scratch.retriever import explain_retrieval


# ─────────────────────────────────────────────────
# Test runner (no pytest dependency needed)
# ─────────────────────────────────────────────────

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name):
        self.passed += 1
        print(f"  ✓  {name}")

    def fail(self, name, reason):
        self.failed += 1
        self.errors.append((name, reason))
        print(f"  ✗  {name}")
        print(f"     {reason}")

results = TestResults()

def test(name):
    """Decorator for test functions."""
    def decorator(fn):
        try:
            fn()
            results.ok(name)
        except AssertionError as e:
            results.fail(name, str(e))
        except Exception as e:
            results.fail(name, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        return fn
    return decorator


# ─────────────────────────────────────────────────
# Stage 0: Data Models
# ─────────────────────────────────────────────────

print("\n── Stage 0: Data Models ────────────────────────────────")

@test("PageContent.to_tagged_string wraps text in physical_index tags")
def _():
    page = PageContent(page_num=5, text="Hello world")
    tagged = page.to_tagged_string()
    assert "<physical_index_5>" in tagged
    assert "Hello world" in tagged
    assert "</physical_index_5>" in tagged


@test("Document.get_pages returns correct range (end exclusive)")
def _():
    doc = create_test_document(num_pages=10)
    pages = doc.get_pages(start=2, end=5)
    assert len(pages) == 3
    assert all(p.page_num in [2, 3, 4] for p in pages)


@test("Document.to_tagged_text concatenates page tags correctly")
def _():
    doc = create_test_document(num_pages=5)
    text = doc.to_tagged_text(start=0, end=3)
    assert "<physical_index_0>" in text
    assert "<physical_index_1>" in text
    assert "<physical_index_2>" in text
    assert "<physical_index_3>" not in text  # end is exclusive


@test("TreeNode.is_leaf is True for nodes without children")
def _():
    leaf = TreeNode("0001", "Section 1", 0, 5)
    assert leaf.is_leaf()
    parent = TreeNode("0002", "Chapter 1", 0, 20, nodes=[leaf])
    assert not parent.is_leaf()


@test("TreeNode.page_count is correct")
def _():
    node = TreeNode("0001", "Section", 5, 12)
    assert node.page_count() == 7


@test("TreeNode.find_by_id finds nested nodes")
def _():
    leaf1 = TreeNode("0002", "Leaf 1", 0, 3)
    leaf2 = TreeNode("0003", "Leaf 2", 3, 6)
    root = TreeNode("0001", "Root", 0, 6, nodes=[leaf1, leaf2])
    assert root.find_by_id("0003") is leaf2
    assert root.find_by_id("0099") is None


@test("TreeNode.all_leaf_nodes returns only leaves")
def _():
    leaf1 = TreeNode("0002", "Leaf 1", 0, 3)
    leaf2 = TreeNode("0003", "Leaf 2", 3, 6)
    inner = TreeNode("0004", "Inner", 6, 10)
    root = TreeNode("0001", "Root", 0, 10, nodes=[
        TreeNode("0005", "Parent", 0, 6, nodes=[leaf1, leaf2]),
        inner
    ])
    leaves = root.all_leaf_nodes()
    assert len(leaves) == 3
    assert inner in leaves
    assert leaf1 in leaves
    assert leaf2 in leaves


@test("TreeNode serialization round-trips correctly")
def _():
    original = TreeNode(
        node_id="0001",
        title="Chapter 1",
        start_index=0,
        end_index=10,
        summary="A summary.",
        nodes=[TreeNode("0002", "Section 1.1", 0, 5)]
    )
    d = original.to_dict()
    restored = TreeNode.from_dict(d)
    assert restored.node_id == "0001"
    assert restored.title == "Chapter 1"
    assert restored.summary == "A summary."
    assert len(restored.nodes) == 1
    assert restored.nodes[0].node_id == "0002"


@test("DocumentIndex.find_node searches across all roots")
def _():
    n1 = TreeNode("0001", "A", 0, 5)
    n2 = TreeNode("0002", "B", 5, 10)
    index = DocumentIndex("test.pdf", root_nodes=[n1, n2])
    assert index.find_node("0001") is n1
    assert index.find_node("0002") is n2
    assert index.find_node("9999") is None


@test("DocumentIndex JSON round-trip (save/load)")
def _():
    import tempfile
    node = TreeNode("0001", "Introduction", 0, 3, summary="Intro text.")
    index = DocumentIndex("report.pdf", description="A report.", root_nodes=[node])

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
        tmp_path = f.name

    try:
        index.save(tmp_path)
        loaded = DocumentIndex.load(tmp_path)
        assert loaded.doc_path == "report.pdf"
        assert loaded.description == "A report."
        assert len(loaded.root_nodes) == 1
        assert loaded.root_nodes[0].title == "Introduction"
        assert loaded.root_nodes[0].summary == "Intro text."
    finally:
        os.unlink(tmp_path)


@test("NodeIDCounter generates zero-padded sequential IDs")
def _():
    counter = NodeIDCounter(start=1)
    assert counter.next() == "0001"
    assert counter.next() == "0002"
    counter2 = NodeIDCounter(start=99)
    assert counter2.next() == "0099"
    assert counter2.next() == "0100"


@test("RetrievalState.add_context accumulates correctly")
def _():
    state = RetrievalState("What is X?", max_iterations=5)
    state.add_context("0001", "Context A")
    state.add_context("0002", "Context B")
    assert state.visited_node_ids == ["0001", "0002"]
    assert "Context A" in state.full_context()
    assert "Context B" in state.full_context()
    assert state.iteration == 2


@test("RetrievalState.budget_remaining respects max_iterations")
def _():
    state = RetrievalState("Q", max_iterations=3)
    assert state.budget_remaining()
    state.iteration = 3
    assert not state.budget_remaining()


# ─────────────────────────────────────────────────
# Stage 1: Ingestion
# ─────────────────────────────────────────────────

print("\n── Stage 1: Document Ingestion ─────────────────────────")

@test("create_test_document creates correct number of pages")
def _():
    doc = create_test_document(num_pages=15)
    assert doc.total_pages == 15
    assert all(isinstance(p, PageContent) for p in doc.pages)
    assert doc.pages[0].page_num == 0
    assert doc.pages[14].page_num == 14


@test("create_test_document pages have non-empty text")
def _():
    doc = create_test_document(num_pages=5)
    assert all(p.text.strip() for p in doc.pages)


@test("_clean_pdf_text fixes ligatures")
def _():
    cleaned = _clean_pdf_text("The ﬁrst ﬂoor has ﬀective use")
    assert "fi" in cleaned
    assert "fl" in cleaned
    assert "ff" in cleaned
    assert "ﬁ" not in cleaned


@test("_clean_pdf_text removes hyphenated line breaks")
def _():
    cleaned = _clean_pdf_text("impor-\ntant information")
    assert "important" in cleaned
    assert "-\n" not in cleaned


@test("txt ingestion splits on blank lines")
def _():
    import tempfile, os
    content = "Page one content.\n\n\nPage two content.\n\n\nPage three."
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode='w') as f:
        f.write(content)
        tmp = f.name
    try:
        from pageindex_scratch.ingestion import _ingest_txt
        doc = _ingest_txt(tmp)
        assert doc.total_pages == 3
        assert "Page one" in doc.pages[0].text
        assert "Page two" in doc.pages[1].text
    finally:
        os.unlink(tmp)


# ─────────────────────────────────────────────────
# Stage 2: TOC Detection
# ─────────────────────────────────────────────────

print("\n── Stage 2: TOC Detection ──────────────────────────────")

@test("_parse_detection_result: no ToC → empty entries")
def _():
    raw = {"has_toc": False, "page_index_given_in_toc": "no", "toc_entries": []}
    result = _parse_detection_result(raw)
    assert not result.has_toc
    assert result.entries == []


@test("_parse_detection_result: ToC without page numbers → treated as no ToC")
def _():
    raw = {
        "has_toc": True,
        "page_index_given_in_toc": "no",
        "toc_entries": [{"title": "Intro", "page": 0, "level": 1}]
    }
    result = _parse_detection_result(raw)
    assert not result.has_toc   # page nums required


@test("_parse_detection_result: valid ToC → correct entries")
def _():
    raw = {
        "has_toc": True,
        "page_index_given_in_toc": "yes",
        "toc_entries": [
            {"title": "Introduction", "page": 2, "level": 1},
            {"title": "Background", "page": 5, "level": 1},
            {"title": "Related Work", "page": 7, "level": 2},
        ]
    }
    result = _parse_detection_result(raw)
    assert result.has_toc
    assert len(result.entries) == 3
    assert result.entries[0]["title"] == "Introduction"
    assert result.entries[0]["page"] == 2


@test("toc_entries_to_tree: flat entries build correct tree")
def _():
    entries = [
        {"title": "Chapter 1", "page": 0, "level": 1},
        {"title": "Section 1.1", "page": 3, "level": 2},
        {"title": "Section 1.2", "page": 7, "level": 2},
        {"title": "Chapter 2", "page": 10, "level": 1},
    ]
    counter = NodeIDCounter()
    roots = toc_entries_to_tree(entries, total_pages=15, id_counter=counter)

    assert len(roots) == 2
    assert roots[0].title == "Chapter 1"
    assert len(roots[0].nodes) == 2
    assert roots[0].nodes[0].title == "Section 1.1"
    assert roots[0].nodes[1].title == "Section 1.2"
    assert roots[1].title == "Chapter 2"


@test("toc_entries_to_tree: page ranges computed correctly")
def _():
    entries = [
        {"title": "Chapter 1", "page": 0, "level": 1},
        {"title": "Chapter 2", "page": 8, "level": 1},
    ]
    counter = NodeIDCounter()
    roots = toc_entries_to_tree(entries, total_pages=20, id_counter=counter)

    assert roots[0].start_index == 0
    assert roots[0].end_index == 8     # next chapter's start
    assert roots[1].start_index == 8
    assert roots[1].end_index == 20    # end of document


@test("toc_entries_to_tree: empty entries returns empty list")
def _():
    counter = NodeIDCounter()
    roots = toc_entries_to_tree([], total_pages=10, id_counter=counter)
    assert roots == []


# ─────────────────────────────────────────────────
# Stage 3: Tree Construction
# ─────────────────────────────────────────────────

print("\n── Stage 3: Tree Construction ──────────────────────────")

@test("_merge_trees: non-overlapping → concatenation")
def _():
    existing = [{"title": "A", "start_index": 0}]
    new_nodes = [{"title": "B", "start_index": 5}]
    merged = _merge_trees(existing, new_nodes, batch_start=5, batch_end=10)
    assert len(merged) == 2


@test("_merge_trees: overlapping → new replaces old")
def _():
    existing = [
        {"title": "A", "start_index": 0},
        {"title": "B", "start_index": 3},
    ]
    # new_nodes starts at 2, overlapping with B
    new_nodes = [{"title": "B revised", "start_index": 2}]
    merged = _merge_trees(existing, new_nodes, batch_start=2, batch_end=6)
    # "A" survives (start=0 < new_min=2), "B" is replaced
    assert len(merged) == 2
    titles = [n["title"] for n in merged]
    assert "A" in titles
    assert "B revised" in titles
    assert "B" not in titles


@test("_merge_trees: empty existing → returns new_nodes")
def _():
    new_nodes = [{"title": "A", "start_index": 0}]
    merged = _merge_trees([], new_nodes, 0, 5)
    assert merged == new_nodes


@test("_dicts_to_tree_nodes: assigns end_index from next sibling")
def _():
    raw = [
        {"title": "Section 1", "start_index": 0, "nodes": []},
        {"title": "Section 2", "start_index": 5, "nodes": []},
    ]
    counter = NodeIDCounter()
    nodes = _dicts_to_tree_nodes(raw, counter, total_pages=12)
    assert nodes[0].end_index == 5   # next sibling start
    assert nodes[1].end_index == 12  # last → total_pages


@test("_dicts_to_tree_nodes: sorts by start_index")
def _():
    # Out-of-order input
    raw = [
        {"title": "Section B", "start_index": 5, "nodes": []},
        {"title": "Section A", "start_index": 0, "nodes": []},
    ]
    counter = NodeIDCounter()
    nodes = _dicts_to_tree_nodes(raw, counter, total_pages=10)
    assert nodes[0].title == "Section A"
    assert nodes[1].title == "Section B"


@test("_dicts_to_tree_nodes: recursively builds child nodes")
def _():
    raw = [
        {
            "title": "Chapter 1", "start_index": 0,
            "nodes": [
                {"title": "Section 1.1", "start_index": 0, "nodes": []},
                {"title": "Section 1.2", "start_index": 3, "nodes": []},
            ]
        }
    ]
    counter = NodeIDCounter()
    nodes = _dicts_to_tree_nodes(raw, counter, total_pages=10)
    assert nodes[0].title == "Chapter 1"
    assert len(nodes[0].nodes) == 2
    assert nodes[0].nodes[0].title == "Section 1.1"


@test("_build_incremental_prompt includes physical_index tags context")
def _():
    # Verify the prompt instructs the LLM about physical_index tags
    prompt = _build_incremental_prompt(
        pages_text="<physical_index_0>text</physical_index_0>",
        current_tree=[],
        start_index=0,
    )
    assert "physical_index" in prompt
    assert "start_index" in prompt
    assert "JSON" in prompt


# ─────────────────────────────────────────────────
# Stage 4: Summarization
# ─────────────────────────────────────────────────

print("\n── Stage 4: Summarization ──────────────────────────────")

@test("_collect_all_nodes returns all nodes including nested")
def _():
    leaf1 = TreeNode("0002", "Leaf 1", 0, 3)
    leaf2 = TreeNode("0003", "Leaf 2", 3, 6)
    inner = TreeNode("0004", "Inner", 0, 6, nodes=[leaf1, leaf2])
    root = TreeNode("0001", "Root", 0, 10, nodes=[inner])
    index = DocumentIndex("test.pdf", root_nodes=[root])

    all_nodes = _collect_all_nodes(index)
    assert len(all_nodes) == 4
    node_ids = {n.node_id for n in all_nodes}
    assert node_ids == {"0001", "0002", "0003", "0004"}


@test("_collect_all_nodes: DFS order (parent before children)")
def _():
    child = TreeNode("0002", "Child", 0, 3)
    root = TreeNode("0001", "Root", 0, 10, nodes=[child])
    index = DocumentIndex("test.pdf", root_nodes=[root])
    all_nodes = _collect_all_nodes(index)
    # Root should come before child in DFS
    ids = [n.node_id for n in all_nodes]
    assert ids.index("0001") < ids.index("0002")


# ─────────────────────────────────────────────────
# Stage 6: Retrieval
# ─────────────────────────────────────────────────

print("\n── Stage 6: Retrieval ──────────────────────────────────")

@test("explain_retrieval produces human-readable trace")
def _():
    state = RetrievalState("What is revenue?")
    state.visited_node_ids = ["0001", "0002"]
    state.collected_context = ["Context A", "Context B"]
    state.iteration = 2
    state.is_sufficient = True

    n1 = TreeNode("0001", "Financial Summary", 5, 10)
    n2 = TreeNode("0002", "Revenue Details", 10, 15)
    index = DocumentIndex("report.pdf", root_nodes=[n1, n2])

    trace = explain_retrieval(state, index)
    assert "What is revenue?" in trace
    assert "Financial Summary" in trace
    assert "Revenue Details" in trace
    assert "Iterations: 2" in trace


@test("RetrievalState: is_sufficient starts False")
def _():
    state = RetrievalState("Q")
    assert not state.is_sufficient


@test("RetrievalState: full_context joins with separator")
def _():
    state = RetrievalState("Q")
    state.add_context("0001", "Part A")
    state.add_context("0002", "Part B")
    full = state.full_context()
    assert "Part A" in full
    assert "Part B" in full
    assert "---" in full  # separator


# ─────────────────────────────────────────────────
# Stage 7: Tree compact repr (used by retriever)
# ─────────────────────────────────────────────────

print("\n── Tree compact repr (retriever input) ─────────────────")

@test("TreeNode.to_compact_repr shows node_id, title, page range")
def _():
    node = TreeNode("0003", "Market Analysis", 10, 20,
                    summary="Covers global market trends.")
    repr_str = node.to_compact_repr()
    assert "0003" in repr_str
    assert "Market Analysis" in repr_str
    assert "10" in repr_str
    assert "20" in repr_str
    assert "global market" in repr_str  # summary preview


@test("DocumentIndex.to_compact_repr includes all nodes")
def _():
    n1 = TreeNode("0001", "Intro", 0, 3)
    n2 = TreeNode("0002", "Body", 3, 8)
    n3 = TreeNode("0003", "Conclusion", 8, 10)
    index = DocumentIndex("report.pdf", description="A report.", root_nodes=[n1, n2, n3])
    repr_str = index.to_compact_repr()
    assert "Intro" in repr_str
    assert "Body" in repr_str
    assert "Conclusion" in repr_str
    assert "A report." in repr_str


@test("to_compact_repr handles nested nodes with indentation")
def _():
    child = TreeNode("0002", "Sub-section", 0, 3)
    root = TreeNode("0001", "Section", 0, 5, nodes=[child])
    index = DocumentIndex("doc.pdf", root_nodes=[root])
    repr_str = index.to_compact_repr()
    lines = repr_str.split("\n")
    # Root line should have less indentation than child
    root_line = next(l for l in lines if "Section" in l and "Sub" not in l)
    child_line = next(l for l in lines if "Sub-section" in l)
    root_indent = len(root_line) - len(root_line.lstrip())
    child_indent = len(child_line) - len(child_line.lstrip())
    assert child_indent > root_indent


# ─────────────────────────────────────────────────
# End-to-end: pipeline with synthetic doc (no LLM)
# ─────────────────────────────────────────────────

print("\n── End-to-end (no LLM) ─────────────────────────────────")

@test("Full index build and query on synthetic doc with manual tree")
def _():
    """
    Build a DocumentIndex manually (bypassing LLM) and run a query.
    This verifies the pipeline plumbing without API calls.
    """
    doc = create_test_document(num_pages=20)

    # Manually construct what the LLM would have produced
    nodes = [
        TreeNode("0001", "Abstract", 0, 1),
        TreeNode("0002", "1. Introduction", 1, 4,
                 summary="Introduces the market analysis scope and methodology.",
                 nodes=[
                     TreeNode("0003", "1.1 Scope", 2, 3),
                     TreeNode("0004", "1.2 Methodology", 3, 4),
                 ]),
        TreeNode("0005", "2. Market Overview", 4, 8,
                 summary="Total addressable market reached $4.2 trillion.",
                 nodes=[
                     TreeNode("0006", "2.1 North America", 5, 6),
                     TreeNode("0007", "2.2 Europe", 6, 7),
                     TreeNode("0008", "2.3 Asia Pacific", 7, 8),
                 ]),
        TreeNode("0009", "3. Technology Sector", 8, 12,
                 summary="AI spend tripled; semiconductor shortages continue."),
        TreeNode("0010", "4. Financial Analysis", 12, 16,
                 summary="Revenue grew 8.4%; EBITDA fell from 22% to 19%."),
        TreeNode("0011", "5. Risk Factors", 16, 17),
        TreeNode("0012", "6. Outlook", 17, 18),
    ]
    index = DocumentIndex(
        doc_path=doc.path,
        description="Market analysis report covering 2024 global trends.",
        root_nodes=nodes
    )

    # Verify tree structure
    assert index.find_node("0007") is not None
    assert index.find_node("0007").title == "2.2 Europe"

    # Verify compact repr
    repr_str = index.to_compact_repr()
    assert "Market Overview" in repr_str
    assert "$4.2 trillion" in repr_str  # from summary

    # Verify page fetch works
    page_text = doc.to_tagged_text(start=4, end=8)
    assert "<physical_index_4>" in page_text
    assert "<physical_index_7>" in page_text

    # Verify serialization
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
        tmp = f.name
    try:
        index.save(tmp)
        loaded = DocumentIndex.load(tmp)
        assert loaded.find_node("0010").summary == "Revenue grew 8.4%; EBITDA fell from 22% to 19%."
    finally:
        os.unlink(tmp)


# ─────────────────────────────────────────────────
# Print summary
# ─────────────────────────────────────────────────

print(f"\n{'─'*55}")
print(f"Results: {results.passed} passed, {results.failed} failed")
if results.errors:
    print("\nFailed tests:")
    for name, reason in results.errors:
        print(f"  • {name}")
        print(f"    {reason[:200]}")
print(f"{'─'*55}\n")

if __name__ == "__main__":
    sys.exit(0 if results.failed == 0 else 1)
