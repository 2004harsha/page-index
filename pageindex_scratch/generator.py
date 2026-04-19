"""
generator.py — Stage 8: Answer Generation

WHAT IT DOES:
  Takes the accumulated context from Stage 6 (tree search) and
  generates a final, well-sourced answer to the user's query.

WHY IT'S A SEPARATE STAGE:
  Retrieval and generation are fundamentally different tasks:
    - Retrieval: find WHAT is relevant (structure-aware, navigational)
    - Generation: synthesize and express (language-aware, compositional)

  Keeping them separate allows:
    - Different temperature settings (retrieval=0.0, generation=0.2)
    - Different prompts tuned for each task
    - Easy replacement of the generator (e.g. swap in a fine-tuned model)
    - Offline evaluation of retrieval quality independent of generation

LEARNING FOCUS:
  - Source citation pattern (traceability back to page numbers)
  - Handling insufficient context gracefully
  - Why generation temperature can be slightly higher than retrieval
"""

from .models import RetrievalState, DocumentIndex
from . import llm_client


# ─────────────────────────────────────────────────
# Main generation function
# ─────────────────────────────────────────────────

def generate_answer(
    state: RetrievalState,
    index: DocumentIndex,
    model: str = llm_client.DEFAULT_MODEL,
    temperature: float = 0.1,
) -> str:
    """
    Generate an answer from retrieved context.

    Args:
        state: RetrievalState from Stage 6 (contains context + metadata)
        index: DocumentIndex (for source attribution)
        model: LLM to use
        temperature: slightly higher than 0 for more natural phrasing

    Returns:
        A string answer with source page references.
    """
    if not state.collected_context:
        return (
            "I could not find relevant information in the document "
            f"to answer: '{state.query}'\n\n"
            "The document index was searched but no matching sections were found."
        )

    # Build source attribution
    sources = _build_source_attribution(state, index)

    prompt = _build_generation_prompt(
        query=state.query,
        context=state.full_context(),
        sources=sources,
    )

    try:
        answer = llm_client.complete(
            prompt=prompt,
            model=model,
            max_tokens=1024,
            temperature=temperature,
        )
        return answer.strip()
    except Exception as e:
        return f"Error generating answer: {e}\n\nRaw context:\n{state.full_context()[:500]}"


def _build_generation_prompt(query: str, context: str, sources: str) -> str:
    """
    PROMPT DESIGN NOTES:
      - We provide the FULL context (all retrieved pages, not just summaries)
      - We instruct citing page/section references — traceability
      - We say "based ONLY on the provided context" — prevents hallucination
      - We ask for plain text — keeps it usable downstream
    """
    return f"""Answer the following question based ONLY on the provided document context.
Be specific and cite the relevant section/page numbers where applicable.
If the context does not fully answer the question, say what IS known and what is uncertain.

Question: {query}

Retrieved document sections:
{context}

Sources: {sources}

Answer:"""


def _build_source_attribution(state: RetrievalState, index: DocumentIndex) -> str:
    """
    Build a source string listing all nodes visited.

    Example output:
      "Section 3.1 AI & ML (pages 9-10), Section 3.2 Cloud (pages 11-12)"
    """
    parts = []
    for node_id in state.visited_node_ids:
        node = index.find_node(node_id)
        if node:
            parts.append(f"{node.title} (pages {node.start_index}-{node.end_index})")
    return ", ".join(parts) if parts else "unknown"
