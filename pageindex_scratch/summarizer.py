"""
summarizer.py — Stage 4: Node Summarization

WHAT IT DOES:
  Generates an LLM summary for each node in the tree and stores it
  in node.summary.

WHY IT MATTERS:
  During tree search (Stage 6), the retriever must decide which branch
  of the tree to follow WITHOUT reading the full page content.
  The summary is the key enabler — it's a compressed representation
  of what the node contains, readable in the LLM context window.

  Think of it like a book's back-of-chapter summary: you can skim it
  to decide "do I need to read this chapter?" without reading the
  whole chapter.

WHEN TO SUMMARIZE:
  - Always summarize leaf nodes (they have no children to describe them)
  - Summarize internal nodes if they're "wide" (many pages, few children)
  - Skip summarization if the node has a short enough page range that
    the retriever can just read the raw pages directly

CONCURRENCY:
  Summarization is the most LLM-call-intensive stage.  A 100-node
  tree = 100 LLM calls.  We run them concurrently with asyncio.

LEARNING FOCUS:
  - What information goes in a summary (content, not just title)
  - How summaries differ from titles
  - Async/concurrent LLM calls pattern
  - Token budget per summary
"""

from __future__ import annotations
import asyncio
from typing import Optional

from .models import DocumentIndex, TreeNode, Document
from . import llm_client


# ─────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────

MAX_SUMMARY_LENGTH = 200    # tokens per summary
CONCURRENCY_LIMIT = 5       # max parallel LLM calls


# ─────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────

def summarize_index(
    index: DocumentIndex,
    doc: Document,
    model: str = llm_client.DEFAULT_MODEL,
    concurrency: int = CONCURRENCY_LIMIT,
) -> DocumentIndex:
    """
    Add summaries to all nodes in the index.

    Uses asyncio for concurrent LLM calls.
    Modifies the index in-place and returns it.
    """
    all_nodes = _collect_all_nodes(index)
    nodes_needing_summary = [n for n in all_nodes if n.summary is None]

    print(f"[summarizer] Summarizing {len(nodes_needing_summary)} nodes "
          f"(concurrency={concurrency})...")

    asyncio.run(_summarize_batch(nodes_needing_summary, doc, model, concurrency))

    print("[summarizer] Done.")
    return index


# ─────────────────────────────────────────────────
# Async summarization
# ─────────────────────────────────────────────────

async def _summarize_batch(
    nodes: list[TreeNode],
    doc: Document,
    model: str,
    concurrency: int,
) -> None:
    """
    Summarize all nodes concurrently, respecting the concurrency limit.

    CONCURRENCY DESIGN:
      We use a semaphore to cap simultaneous LLM calls.
      Without this, 100 concurrent calls would hit rate limits
      and all fail.  With semaphore(5): we process 5 at a time,
      so a 100-node tree takes ~20 rounds instead of 100 sequential calls.
    """
    sem = asyncio.Semaphore(concurrency)

    async def summarize_one(node: TreeNode) -> None:
        async with sem:
            node.summary = await _async_summarize_node(node, doc, model)

    tasks = [summarize_one(node) for node in nodes]
    await asyncio.gather(*tasks, return_exceptions=True)


async def _async_summarize_node(
    node: TreeNode,
    doc: Document,
    model: str,
) -> Optional[str]:
    """
    Generate a summary for a single node asynchronously.
    """
    # Get the page content for this node
    page_text = doc.to_tagged_text(start=node.start_index, end=node.end_index)

    if not page_text.strip():
        return None

    prompt = _build_summary_prompt(node.title, page_text)

    loop = asyncio.get_event_loop()
    try:
        summary = await loop.run_in_executor(
            None,
            lambda: llm_client.complete(
                prompt=prompt,
                model=model,
                max_tokens=MAX_SUMMARY_LENGTH,
                temperature=0.0,
            )
        )
        return summary.strip()
    except Exception as e:
        print(f"[summarizer] Failed to summarize '{node.title}': {e}")
        return None


def _build_summary_prompt(title: str, page_text: str) -> str:
    """
    PROMPT DESIGN NOTES:
      - We tell the LLM the section title so it can focus on what's
        distinctive (not just re-state the title)
      - We ask for KEY FACTS (numbers, names, conclusions) — this is
        what makes the summary useful for retrieval reasoning
      - We say "2-3 sentences max" — keeps summaries short enough to
        fit many in the retriever's context window
      - We say "no markdown" — plain text is easier to embed in the
        compact tree representation

    WHY FACTUAL SUMMARIES BEAT GENERIC ONES:
      Generic: "This section discusses financial performance."
      Factual:  "Revenue grew 12% to $1.8 trillion; EBITDA margin
                 compressed from 22% to 19% due to supply chain costs."

      The second summary lets the retriever answer questions like
      "what was the EBITDA margin?" without even fetching the pages.
    """
    return f"""Summarize the following document section in 2-3 sentences.
Focus on key facts, numbers, names, and conclusions — not generic descriptions.
Do NOT use markdown. Write plain text only.

Section title: {title}

Section content:
{page_text[:3000]}

Summary:"""


# ─────────────────────────────────────────────────
# Synchronous fallback (for environments without async)
# ─────────────────────────────────────────────────

def summarize_index_sync(
    index: DocumentIndex,
    doc: Document,
    model: str = llm_client.DEFAULT_MODEL,
) -> DocumentIndex:
    """
    Synchronous version of summarize_index.
    Slower (sequential LLM calls) but simpler to debug.
    """
    all_nodes = _collect_all_nodes(index)
    total = len([n for n in all_nodes if n.summary is None])
    print(f"[summarizer] Summarizing {total} nodes (sequential)...")

    for i, node in enumerate(all_nodes):
        if node.summary is not None:
            continue
        page_text = doc.to_tagged_text(start=node.start_index, end=node.end_index)
        if not page_text.strip():
            continue

        prompt = _build_summary_prompt(node.title, page_text)
        try:
            node.summary = llm_client.complete(
                prompt=prompt, model=model, max_tokens=MAX_SUMMARY_LENGTH
            ).strip()
            print(f"[summarizer] {i+1}/{total} — {node.title[:50]}")
        except Exception as e:
            print(f"[summarizer] Failed: {node.title}: {e}")

    return index


# ─────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────

def _collect_all_nodes(index: DocumentIndex) -> list[TreeNode]:
    """Flatten the entire tree into a list of all nodes (DFS)."""
    result = []

    def dfs(node: TreeNode) -> None:
        result.append(node)
        for child in node.nodes:
            dfs(child)

    for root in index.root_nodes:
        dfs(root)

    return result
