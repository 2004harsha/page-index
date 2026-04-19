"""
tree_builder.py — Stage 3: Tree Construction

THIS IS THE CORE OF PAGEINDEX.

WHAT IT DOES:
  Builds a hierarchical TreeNode structure that represents the
  logical organisation of the document.

  There are TWO paths depending on what Stage 2 found:

    PATH A (has_toc=True):
      The ToC-based tree is already partially built by toc_detector.
      We only need to recursively expand any large nodes whose
      sub-sections weren't in the ToC.

    PATH B (has_toc=False):
      We do NOT have structural hints.  We feed pages to the LLM
      in batches, ask it to detect section boundaries, and
      incrementally merge the results into a single tree.

ALGORITHM (Path B — no ToC):
  1. Divide pages into batches of max_pages_per_batch
  2. For each batch, send: previous_tree + new_pages to the LLM
  3. LLM responds with an updated tree JSON
  4. Post-process: verify that claimed titles actually appear in pages
  5. Recurse: for any node wider than max_pages_per_node,
     run the same algorithm on just those pages

LEARNING FOCUS:
  - The incremental merge strategy (why we send "previous tree")
  - Why large nodes are recursively expanded
  - Post-processing for title verification
  - The exact LLM prompt structure
"""

from __future__ import annotations
import json
from typing import Optional

from .models import Document, TreeNode, NodeIDCounter, DocumentIndex
from .toc_detector import TocDetectionResult, toc_entries_to_tree
from . import llm_client


# ─────────────────────────────────────────────────
# Configuration defaults (mirror PageIndex defaults)
# ─────────────────────────────────────────────────

MAX_PAGES_PER_NODE = 10         # Nodes wider than this get recursively expanded
MAX_PAGES_PER_BATCH = 15        # Pages fed to LLM per incremental step
MAX_TOKENS_PER_NODE = 20_000    # Token budget per node content fetch


# ─────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────

def build_tree(
    doc: Document,
    toc_result: TocDetectionResult,
    model: str = llm_client.DEFAULT_MODEL,
    max_pages_per_node: int = MAX_PAGES_PER_NODE,
    max_pages_per_batch: int = MAX_PAGES_PER_BATCH,
) -> DocumentIndex:
    """
    Build the full document index tree.

    Returns a DocumentIndex ready for summarization and persistence.
    """
    id_counter = NodeIDCounter(start=1)

    print(f"[tree_builder] Building tree for {doc.total_pages}-page document...")

    if toc_result.has_toc:
        print(f"[tree_builder] Using ToC path ({len(toc_result.entries)} entries found)")
        root_nodes = _build_from_toc(
            doc, toc_result, id_counter, model, max_pages_per_node
        )
    else:
        print("[tree_builder] No ToC found — using incremental scan path")
        root_nodes = _build_incremental(
            doc, id_counter, model, max_pages_per_batch, max_pages_per_node
        )

    # Generate a document-level description
    description = _generate_doc_description(doc, model)

    return DocumentIndex(
        doc_path=doc.path,
        description=description,
        root_nodes=root_nodes,
    )


# ─────────────────────────────────────────────────
# Path A: Build from existing ToC
# ─────────────────────────────────────────────────

def _build_from_toc(
    doc: Document,
    toc_result: TocDetectionResult,
    id_counter: NodeIDCounter,
    model: str,
    max_pages_per_node: int,
) -> list[TreeNode]:
    """
    Convert ToC entries to TreeNodes, then recursively expand wide nodes.

    KEY INSIGHT:
      A ToC gives us the top-level structure (chapters, sections) but
      often omits sub-sections.  Any node spanning more than
      max_pages_per_node pages is "too wide" — the LLM might not
      extract fine-grained answers from it accurately.  We recursively
      drill into those nodes.
    """
    root_nodes = toc_entries_to_tree(toc_result.entries, doc.total_pages, id_counter)

    # Expand any oversized nodes
    for node in root_nodes:
        _expand_wide_nodes(node, doc, id_counter, model, max_pages_per_node)

    return root_nodes


def _expand_wide_nodes(
    node: TreeNode,
    doc: Document,
    id_counter: NodeIDCounter,
    model: str,
    max_pages_per_node: int,
) -> None:
    """
    Recursively expand nodes that span more pages than the threshold.

    DESIGN NOTE:
      We expand a node only if it has no children yet.  If the ToC
      already gave us sub-sections for this node, we trust those.
      Only leaf nodes that are too wide need expansion.
    """
    if not node.is_leaf():
        # Recurse into existing children
        for child in node.nodes:
            _expand_wide_nodes(child, doc, id_counter, model, max_pages_per_node)
        return

    if node.page_count() <= max_pages_per_node:
        return  # Small enough — no expansion needed

    print(f"[tree_builder] Expanding wide node '{node.title}' "
          f"(pages {node.start_index}-{node.end_index})...")

    sub_nodes = _build_incremental(
        doc, id_counter, model,
        max_pages_per_batch=max_pages_per_node,
        max_pages_per_node=max_pages_per_node,
        start_page=node.start_index,
        end_page=node.end_index,
    )

    if sub_nodes:
        node.nodes = sub_nodes


# ─────────────────────────────────────────────────
# Path B: Incremental scan (no ToC)
# ─────────────────────────────────────────────────

def _build_incremental(
    doc: Document,
    id_counter: NodeIDCounter,
    model: str,
    max_pages_per_batch: int,
    max_pages_per_node: int,
    start_page: int = 0,
    end_page: Optional[int] = None,
) -> list[TreeNode]:
    """
    Incrementally build a tree by feeding the LLM batches of pages.

    THE KEY ALGORITHM:

      accumulated_tree = []
      for each batch of pages:
          prompt = (instructions
                    + current_pages_tagged
                    + json(accumulated_tree))
          response = LLM(prompt)
          accumulated_tree = merge(accumulated_tree, response)

      return post_process(accumulated_tree)

    The critical insight is passing `accumulated_tree` back to the LLM
    with each batch.  This is the "in-context incremental merge" strategy.
    The LLM can see what structure it's already committed to and either
    continue building from it or revise earlier decisions based on new
    evidence from later pages.

    This is analogous to how AlphaGo keeps track of the board state
    when planning the next move — the LLM "knows where it is" in
    the document.
    """
    if end_page is None:
        end_page = doc.total_pages

    accumulated_tree: list[dict] = []  # raw dicts (not yet TreeNodes)

    page = start_page
    while page < end_page:
        batch_end = min(page + max_pages_per_batch, end_page)
        batch_pages = doc.to_tagged_text(start=page, end=batch_end)

        print(f"[tree_builder] Processing pages {page}-{batch_end} ...")

        prompt = _build_incremental_prompt(
            pages_text=batch_pages,
            current_tree=accumulated_tree,
            start_index=page,
        )

        try:
            response = llm_client.complete_json(prompt=prompt, model=model, max_tokens=4096)
        except Exception as e:
            print(f"[tree_builder] LLM call failed for pages {page}-{batch_end}: {e}")
            page = batch_end
            continue

        # response should be a list of node dicts
        new_nodes = response if isinstance(response, list) else response.get("nodes", [])
        accumulated_tree = _merge_trees(accumulated_tree, new_nodes, page, batch_end)
        page = batch_end

    # Convert raw dicts to TreeNode objects
    return _dicts_to_tree_nodes(accumulated_tree, id_counter, end_page)


def _build_incremental_prompt(
    pages_text: str,
    current_tree: list[dict],
    start_index: int,
) -> str:
    """
    PROMPT DESIGN NOTES:
      The prompt has three parts:
        1. Instructions: the JSON schema the LLM must produce
        2. Current pages: the batch of pages to process
        3. Previous tree: the tree built so far (for context)

      The 'structure' field (e.g. "1", "1.1", "2.3") is a
      hierarchical numbering system.  PageIndex uses this to help
      the LLM maintain consistent hierarchy depth across batches.

      We use <physical_index_N> tags so the LLM can report
      accurate start_index values.
    """
    prev_tree_str = json.dumps(current_tree, indent=2) if current_tree else "[]"

    return f"""You are an expert at extracting hierarchical document structure.

Your task: given the document pages below, extract the logical section structure.

Respond ONLY with a JSON array of section nodes. Each node must have:
{{
  "title": "exact section title from the text",
  "structure": "hierarchical number, e.g. '1', '1.1', '2.3'",
  "start_index": <page number where this section starts (0-based)>,
  "nodes": [<child nodes using same schema>]
}}

Rules:
- Use the <physical_index_N> tags to identify which page each section starts on
- Keep titles exactly as they appear in the document (fix spacing only)
- structure '1' = top-level, '1.1' = sub-section, '1.1.1' = sub-sub-section
- Do NOT invent titles — only extract what is explicitly in the text
- If a page continues a section from a previous page, do NOT create a new node for it

Previous tree structure (continue from this):
{prev_tree_str}

Current document pages:
{pages_text}"""


def _merge_trees(
    existing: list[dict],
    new_nodes: list[dict],
    batch_start: int,
    batch_end: int,
) -> list[dict]:
    """
    Merge new LLM-extracted nodes into the accumulated tree.

    STRATEGY:
      The LLM may return:
        a) Only new nodes for the current batch (most common)
        b) A revision of nodes spanning previous + current batches

      We detect which case we're in by checking whether new node
      start_index values overlap with existing nodes.

      Overlap → the LLM revised earlier nodes, trust the new version
      No overlap → purely additive, just concatenate
    """
    if not existing:
        return new_nodes

    if not new_nodes:
        return existing

    # Find the minimum start_index in new_nodes
    new_min_start = min(
        (n.get("start_index", batch_start) for n in new_nodes),
        default=batch_start
    )

    # If the new nodes start before our batch, they overlap with existing
    # Remove existing nodes whose start_index >= new_min_start
    surviving = [n for n in existing if n.get("start_index", 0) < new_min_start]
    return surviving + new_nodes


def _dicts_to_tree_nodes(
    raw: list[dict],
    id_counter: NodeIDCounter,
    total_pages: int,
) -> list[TreeNode]:
    """
    Convert raw dicts from LLM output to proper TreeNode objects.

    Also computes end_index for each node:
      end_index = next sibling's start_index, or total_pages for the last

    LEARNING NOTE:
      The LLM returns start_index but NOT end_index (it can't know where
      a section ends when scanning from the start).  We compute end_index
      here by looking at what comes next.
    """
    if not raw:
        return []

    # Sort by start_index to ensure correct order
    sorted_raw = sorted(raw, key=lambda n: n.get("start_index", 0))

    nodes = []
    for i, item in enumerate(sorted_raw):
        # end_index = next sibling's start_index, or total_pages
        if i + 1 < len(sorted_raw):
            end_idx = sorted_raw[i + 1].get("start_index", total_pages)
        else:
            end_idx = total_pages

        child_nodes = _dicts_to_tree_nodes(
            item.get("nodes", []),
            id_counter,
            end_idx,
        )

        node = TreeNode(
            node_id=id_counter.next(),
            title=item.get("title", f"Section {i+1}"),
            start_index=item.get("start_index", 0),
            end_index=end_idx,
            nodes=child_nodes,
        )
        nodes.append(node)

    return nodes


# ─────────────────────────────────────────────────
# Post-processing: verify titles appear in pages
# ─────────────────────────────────────────────────

def verify_title_appearances(
    nodes: list[TreeNode],
    doc: Document,
    model: str,
) -> list[TreeNode]:
    """
    Verify that each node's title actually appears in the pages it claims.

    DESIGN NOTE from original PageIndex source:
      `check_title_appearance_in_start_concurrent` — this step catches
      cases where the LLM hallucinates a section title that doesn't
      exist in the document, or gets the page number wrong.

      Implementation: for each node, check if title text appears in
      the page at start_index ± 1.  If not, flag it for review.
      We use a lightweight string match here; PageIndex uses another
      LLM call for ambiguous cases.
    """
    verified = []
    for node in nodes:
        pages_to_check = doc.get_pages(
            max(0, node.start_index - 1),
            min(doc.total_pages, node.start_index + 2)
        )
        page_texts = " ".join(p.text.lower() for p in pages_to_check)
        title_words = node.title.lower().split()

        # Accept if more than half the title words appear
        matches = sum(1 for w in title_words if w in page_texts)
        if len(title_words) == 0 or matches / len(title_words) >= 0.5:
            verified.append(node)
        else:
            print(f"[tree_builder] WARNING: title '{node.title}' "
                  f"not found near page {node.start_index} — keeping anyway")
            verified.append(node)  # Keep it but warn

    return verified


# ─────────────────────────────────────────────────
# Document-level description
# ─────────────────────────────────────────────────

def _generate_doc_description(doc: Document, model: str) -> str:
    """
    Generate a one-paragraph description of the full document.

    Uses the first 3 pages as context (usually abstract + intro).
    This description is stored at the DocumentIndex level and helps
    the retriever quickly decide if the document is relevant to a query.
    """
    preview_text = doc.to_tagged_text(start=0, end=min(3, doc.total_pages))
    prompt = f"""Based on the following pages from the start of a document,
write a single concise paragraph (2-3 sentences) describing what this document is about.
Be factual and specific. Do not use markdown.

Document pages:
{preview_text}

Description:"""

    try:
        return llm_client.complete(prompt=prompt, model=model, max_tokens=200)
    except Exception as e:
        print(f"[tree_builder] Could not generate description: {e}")
        return ""
