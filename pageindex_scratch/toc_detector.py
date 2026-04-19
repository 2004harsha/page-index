"""
toc_detector.py — Stage 2: Table of Contents Detection

WHAT IT DOES:
  Checks whether the document already has a Table of Contents (ToC)
  in its first N pages.  If it does, we can extract node structure
  *much more cheaply* than the incremental LLM scan.

WHY IT MATTERS:
  This is a critical bifurcation in the pipeline:

    HAS TOC → parse page numbers from it → skip ~90% of LLM calls
    NO TOC  → must scan every page batch to discover structure

  Professional documents (SEC filings, research papers, manuals)
  almost always have a ToC.  Detecting it correctly is a major
  efficiency win.

ALGORITHM:
  1. Grab the first `toc_check_pages` pages (default 20)
  2. Ask the LLM: "Does this document have a Table of Contents?
     If yes, extract all entries with their page numbers."
  3. Return a structured dict: {"has_toc": bool, "entries": [...]}

LEARNING FOCUS:
  - Why we check only the first N pages (ToCs are always at the front)
  - The LLM prompt structure for reliable structured output
  - How to validate the extracted entries
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from .models import Document, TreeNode, NodeIDCounter
from . import llm_client


# ─────────────────────────────────────────────────
# Data returned by this stage
# ─────────────────────────────────────────────────

@dataclass
class TocDetectionResult:
    has_toc: bool
    entries: list[dict]         # [{"title": ..., "page": int, "level": int}]
    raw_toc_text: Optional[str] = None   # for debugging


# ─────────────────────────────────────────────────
# Main detection function
# ─────────────────────────────────────────────────

def detect_toc(
    doc: Document,
    toc_check_pages: int = 20,
    model: str = llm_client.DEFAULT_MODEL,
) -> TocDetectionResult:
    """
    Detect and extract the Table of Contents from a document.

    Args:
        doc: the parsed document
        toc_check_pages: how many pages to scan (default 20)
        model: which LLM to use

    Returns:
        TocDetectionResult with has_toc flag and parsed entries
    """
    check_pages = min(toc_check_pages, doc.total_pages)
    pages_text = doc.to_tagged_text(start=0, end=check_pages)

    print(f"[toc_detector] Scanning first {check_pages} pages for ToC...")

    prompt = _build_detection_prompt(pages_text)
    try:
        result = llm_client.complete_json(
            prompt=prompt,
            model=model,
            max_tokens=4096,
        )
    except Exception as e:
        print(f"[toc_detector] LLM call failed, assuming no ToC: {e}")
        return TocDetectionResult(has_toc=False, entries=[])

    return _parse_detection_result(result)


def _build_detection_prompt(pages_text: str) -> str:
    """
    PROMPT DESIGN NOTES:
      - We ask for `page_index_given_in_toc` to distinguish real ToC
        (with page numbers) from a simple chapter overview (no page numbers).
      - We ask for `level` to capture hierarchy (1 = chapter, 2 = section, etc.)
      - We instruct "respond ONLY with valid JSON" — no preamble.
      - We pass the page-tagged text so the LLM can cross-reference
        the ToC entries with the actual page content.
    """
    return f"""You are analyzing a document to determine if it has a Table of Contents.

Given the following pages from the start of a document, please:
1. Determine if there is a Table of Contents (ToC)
2. If yes, extract all ToC entries with their page numbers

Respond ONLY with valid JSON in this exact format:
{{
  "has_toc": true or false,
  "page_index_given_in_toc": "yes" or "no",
  "toc_entries": [
    {{
      "title": "Section title exactly as it appears",
      "page": <page number as integer, 0-indexed>,
      "level": <hierarchy level: 1=chapter, 2=section, 3=subsection>
    }}
  ],
  "toc_raw_text": "copy of the raw ToC text if found, else null"
}}

If there is no ToC, set has_toc to false and toc_entries to [].

IMPORTANT:
- page numbers should be 0-indexed (first page of document = 0)
- Only extract entries that have explicit page numbers in the ToC
- level 1 = top-level chapter/section
- level 2 = sub-section
- level 3 = sub-sub-section

Document pages:
{pages_text}"""


def _parse_detection_result(raw: dict) -> TocDetectionResult:
    has_toc = raw.get("has_toc", False)
    page_nums_given = raw.get("page_index_given_in_toc", "no") == "yes"
    entries = raw.get("toc_entries", [])

    # Only use entries if the ToC actually has page numbers
    # A ToC without page numbers can't help us build page ranges
    if not page_nums_given:
        entries = []
        has_toc = False

    # Validate and clean entries
    clean_entries = []
    for entry in entries:
        if "title" in entry and "page" in entry:
            clean_entries.append({
                "title": str(entry["title"]).strip(),
                "page": int(entry["page"]),
                "level": int(entry.get("level", 1)),
            })

    return TocDetectionResult(
        has_toc=has_toc and bool(clean_entries),
        entries=clean_entries,
        raw_toc_text=raw.get("toc_raw_text"),
    )


# ─────────────────────────────────────────────────
# Convert ToC entries → initial tree nodes
# ─────────────────────────────────────────────────

def toc_entries_to_tree(
    entries: list[dict],
    total_pages: int,
    id_counter: NodeIDCounter,
) -> list[TreeNode]:
    """
    Convert flat ToC entries (with levels) into a TreeNode hierarchy.

    ALGORITHM:
      We process entries in order.  A stack tracks "open" parent nodes
      at each level.  When we encounter a level-N entry, all currently
      open nodes at levels >= N are "closed" (their end_index is set).

      This mirrors how a human reader constructs a mental model from a ToC:
      "Chapter 3 started at page 15.  Chapter 4 starts at page 42.
       Therefore Chapter 3 ends at page 41."

    The page range for each node is:
      start_index = entry["page"]
      end_index   = next sibling's page (or total_pages if last)
    """
    if not entries:
        return []

    # Build flat list first, computing end_index
    flat: list[dict] = []
    for i, entry in enumerate(entries):
        start = entry["page"]
        # End is the start of the next entry at same/higher level, or doc end
        end = total_pages
        for j in range(i + 1, len(entries)):
            if entries[j]["level"] <= entry["level"]:
                end = entries[j]["page"]
                break
        flat.append({
            "title": entry["title"],
            "level": entry["level"],
            "start_index": start,
            "end_index": end,
            "node_id": id_counter.next(),
        })

    # Convert to TreeNode hierarchy using a level stack
    root_nodes: list[TreeNode] = []
    # stack[i] = the last open node at level i
    stack: dict[int, TreeNode] = {}

    for item in flat:
        level = item["level"]
        node = TreeNode(
            node_id=item["node_id"],
            title=item["title"],
            start_index=item["start_index"],
            end_index=item["end_index"],
        )

        if level == 1:
            root_nodes.append(node)
        else:
            # Find closest parent (nearest level < current)
            parent_level = level - 1
            while parent_level >= 1 and parent_level not in stack:
                parent_level -= 1
            if parent_level >= 1 and parent_level in stack:
                stack[parent_level].nodes.append(node)
            else:
                # Orphaned entry — attach to root level
                root_nodes.append(node)

        stack[level] = node

    return root_nodes
