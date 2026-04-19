"""
retriever.py — Stage 6: Agentic Tree Search Retrieval

THIS IS THE KEY DIFFERENTIATOR OF PAGEINDEX vs. VECTOR RAG.

WHAT IT DOES:
  Given a user query and a DocumentIndex (the tree), finds the
  most relevant page content by *reasoning over the tree structure*
  rather than by embedding similarity.

THE ALGORITHM (AlphaGo analogy):
  AlphaGo evaluates the game board state and picks the most
  promising next move.  PageIndex's retriever evaluates the
  tree structure and picks the most promising next node to read.

  Loop:
    1. Show the LLM: (query + compact tree + visited nodes)
    2. LLM selects the node_id most likely to contain the answer
    3. Fetch raw page content for that node
    4. LLM evaluates: "is this enough to answer the query?"
       YES → stop
       NO  → repeat from step 1 (max_iterations times)

WHAT MAKES THIS BETTER THAN TOP-K COSINE:
  - It reasons about RELEVANCE, not SIMILARITY
  - It can follow in-document cross-references ("see Appendix G")
  - It's traceable: you can see exactly which nodes were visited
  - It uses context from prior iterations ("already found X, now need Y")

LEARNING FOCUS:
  - The node selection prompt (how to make the LLM reason, not guess)
  - The sufficiency check (separate concern from selection)
  - Loop state management via RetrievalState
  - How context accumulates across iterations
"""

from __future__ import annotations
from typing import Optional

from .models import DocumentIndex, TreeNode, RetrievalState, Document
from . import llm_client


# ─────────────────────────────────────────────────
# Main retrieval entry point
# ─────────────────────────────────────────────────

def retrieve(
    query: str,
    index: DocumentIndex,
    doc: Document,
    model: str = llm_client.DEFAULT_MODEL,
    max_iterations: int = 5,
) -> RetrievalState:
    """
    Run agentic tree search to find pages relevant to the query.

    Returns a RetrievalState containing:
      - collected_context: list of relevant page texts
      - visited_node_ids: nodes that were read
      - is_sufficient: whether the retriever declared success

    The caller (Stage 8) uses state.full_context() to generate the answer.
    """
    state = RetrievalState(query=query, max_iterations=max_iterations)

    tree_repr = index.to_compact_repr()

    while state.budget_remaining() and not state.is_sufficient:
        state.iteration += 1
        print(f"[retriever] Iteration {state.iteration}/{max_iterations}")

        # Step 1: Select the next node to explore
        node_id = _select_node(
            query=query,
            tree_repr=tree_repr,
            state=state,
            model=model,
        )

        if node_id is None:
            print("[retriever] LLM could not select a node — stopping")
            break

        if node_id in state.visited_node_ids:
            print(f"[retriever] Node {node_id} already visited — stopping")
            break

        # Step 2: Fetch page content for the selected node
        node = index.find_node(node_id)
        if node is None:
            print(f"[retriever] Node {node_id} not found in index")
            break

        page_text = doc.to_tagged_text(start=node.start_index, end=node.end_index)
        print(f"[retriever] Fetched node [{node_id}] '{node.title}' "
              f"(pages {node.start_index}-{node.end_index})")

        # Step 3: Accumulate context
        context_entry = (
            f"=== Node: {node.title} (pages {node.start_index}-{node.end_index}) ===\n"
            f"{page_text}"
        )
        state.add_context(node_id, context_entry)

        # Step 4: Check if we have enough context to answer
        state.is_sufficient = _check_sufficiency(
            query=query,
            accumulated_context=state.full_context(),
            model=model,
        )

        if state.is_sufficient:
            print("[retriever] Context deemed sufficient — stopping")
        else:
            print("[retriever] Context not yet sufficient — continuing search")

    return state


# ─────────────────────────────────────────────────
# Step 1: Node selection
# ─────────────────────────────────────────────────

def _select_node(
    query: str,
    tree_repr: str,
    state: RetrievalState,
    model: str,
) -> Optional[str]:
    """
    Ask the LLM to pick the most relevant node_id to explore next.

    PROMPT DESIGN (most important part of the retriever):
      We give the LLM:
        - The query
        - The full tree (compact repr with summaries)
        - Already visited nodes (to avoid repetition)
        - Already collected context (to guide "what's still missing")

      We ask for ONLY the node_id — nothing else.
      This makes parsing trivial and forces the model to commit.

    WHY THIS WORKS:
      The LLM can read the compact tree (titles + summaries) and
      reason: "The query is about EBITDA margins.  I can see node 0006
      is 'Financial Analysis' with summary 'covers margin trends...'.
      That's where I should look."

      It's the same reasoning a human uses when scanning a book's ToC.
    """
    visited_str = ", ".join(state.visited_node_ids) if state.visited_node_ids else "none"
    context_summary = (
        state.full_context()[:500] + "..."
        if len(state.full_context()) > 500
        else state.full_context()
    )

    prompt = f"""You are helping answer this question: "{query}"

You have access to a hierarchical document index. Your task is to select the SINGLE
most relevant node to read next to find information that answers the question.

Document index:
{tree_repr}

Nodes already visited (do NOT select these): {visited_str}

Context already collected:
{context_summary if context_summary else "(none yet)"}

Instructions:
- Look at the node titles and summaries carefully
- Select the node most likely to contain information needed to answer the question
- If already collected context partially answers the question, look for nodes with the MISSING information
- Respond with ONLY the node_id (e.g. "0006") — nothing else, no explanation

node_id:"""

    try:
        response = llm_client.complete(
            prompt=prompt,
            model=model,
            max_tokens=20,
            temperature=0.0,
        )
        # Extract just the node_id (strip any accidental whitespace/quotes)
        node_id = response.strip().strip('"\'').strip()
        # Validate format: should be 4 digits
        if node_id.isdigit() or (len(node_id) == 4 and node_id.isdigit()):
            return node_id.zfill(4)
        # Handle if LLM returned a longer response
        import re
        match = re.search(r'\b(\d{4})\b', node_id)
        if match:
            return match.group(1)
        print(f"[retriever] Unexpected node_id format: '{node_id}'")
        return None
    except Exception as e:
        print(f"[retriever] Node selection failed: {e}")
        return None


# ─────────────────────────────────────────────────
# Step 4: Sufficiency check
# ─────────────────────────────────────────────────

def _check_sufficiency(
    query: str,
    accumulated_context: str,
    model: str,
) -> bool:
    """
    Ask the LLM: "Do you have enough information to answer the question?"

    DESIGN NOTE:
      This is a SEPARATE LLM call from node selection, by design.
      Mixing "select next node" with "am I done?" in one call leads to
      inconsistent behaviour — the model might commit to a node even
      when it already has the answer, or vice versa.

      Separation of concerns: one call → one decision.

    THRESHOLD:
      We use a simple yes/no question.  In production you might want
      a confidence score or ask the LLM to list what's still missing.
    """
    if not accumulated_context.strip():
        return False

    prompt = f"""Question: "{query}"

Retrieved context:
{accumulated_context[:3000]}

Based on the retrieved context above, can you fully and accurately answer the question?

Answer with ONLY "yes" or "no":"""

    try:
        response = llm_client.complete(
            prompt=prompt,
            model=model,
            max_tokens=10,
            temperature=0.0,
        ).strip().lower()
        return response.startswith("yes")
    except Exception:
        return False


# ─────────────────────────────────────────────────
# Utility: explain retrieval trace
# ─────────────────────────────────────────────────

def explain_retrieval(state: RetrievalState, index: DocumentIndex) -> str:
    """
    Return a human-readable explanation of what the retriever did.

    Useful for debugging and for building explainable RAG systems.
    This is a key advantage over vector search — we can show exactly
    which nodes were visited and in what order.
    """
    lines = [
        f"Query: {state.query}",
        f"Iterations: {state.iteration}",
        f"Sufficient: {state.is_sufficient}",
        "",
        "Nodes visited (in order):",
    ]
    for i, node_id in enumerate(state.visited_node_ids):
        node = index.find_node(node_id)
        if node:
            lines.append(
                f"  {i+1}. [{node_id}] {node.title} "
                f"(pp {node.start_index}-{node.end_index})"
            )
        else:
            lines.append(f"  {i+1}. [{node_id}] (not found)")

    return "\n".join(lines)
