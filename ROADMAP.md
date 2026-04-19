# PageIndex from Scratch — Project Roadmap

A learning implementation of VectifyAI's PageIndex system: **vectorless, reasoning-based RAG**.

> Reference: https://github.com/VectifyAI/PageIndex

---

## What you will learn

By building this project you'll deeply understand:

1. Why vector-based RAG fails on professional documents
2. How a hierarchical tree index differs from a chunk store
3. The incremental LLM merge algorithm (the heart of the system)
4. Agentic retrieval — the loop that replaces cosine search
5. How to design prompts for structured, deterministic output
6. How page boundaries flow through every stage of the pipeline

---

## Project structure

```
pageindex_project/
├── main.py                         ← CLI entry point
├── tests.py                        ← Tests (all stages, no API key needed)
├── ROADMAP.md                      ← This file
├── requirements.txt
└── pageindex_scratch/
    ├── __init__.py
    ├── models.py        ← Stage 0: Core data structures
    ├── ingestion.py     ← Stage 1: PDF → list[PageContent]
    ├── toc_detector.py  ← Stage 2: Detect & parse existing ToC
    ├── tree_builder.py  ← Stage 3: Build hierarchical TreeNode tree  ★ CORE
    ├── summarizer.py    ← Stage 4: LLM summaries per node (async)
    ├── llm_client.py    ← LLM wrapper (swap models here)
    ├── retriever.py     ← Stage 6: Agentic tree search             ★ CORE
    ├── generator.py     ← Stage 8: Answer from retrieved context
    └── pipeline.py      ← Orchestrator tying all stages together
```

---

## The complete pipeline

### Phase 1 — Index building (offline, run once per document)

| Stage | File | Input | Output | Key concept |
|-------|------|-------|--------|-------------|
| 1 | `ingestion.py` | PDF / TXT / MD | `Document` (list of `PageContent`) | Physical page boundaries |
| 2 | `toc_detector.py` | `Document` | `TocDetectionResult` | LLM scans first N pages |
| 3 | `tree_builder.py` | `Document` + ToC result | `DocumentIndex` (tree) | **Incremental merge algorithm** |
| 4 | `summarizer.py` | `DocumentIndex` + `Document` | `DocumentIndex` (+ summaries) | Concurrent LLM calls |
| 5 | `models.py` | `DocumentIndex` | JSON on disk | Stable serialization |

### Phase 2 — Retrieval (online, runs per query)

| Stage | File | Input | Output | Key concept |
|-------|------|-------|--------|-------------|
| 5 | `models.py` | JSON on disk | `DocumentIndex` | Load in <1ms |
| 6 | `retriever.py` | Query + `DocumentIndex` + `Document` | `RetrievalState` | **Agentic loop** |
| 8 | `generator.py` | `RetrievalState` + `DocumentIndex` | Answer string | Cited generation |

---

## Deep dive: the 5 design decisions to understand

### 1. Physical page tags `<physical_index_N>`

Every page is wrapped:
```
<physical_index_5>
...text of page 5...
</physical_index_5>
```
This is how the LLM can reliably report `start_index: 5` in the tree JSON. Without the tags, the model guesses page numbers and gets them wrong.

**Exercise**: Remove the tags and see how tree construction accuracy drops.

---

### 2. The incremental merge algorithm (tree_builder.py)

```python
accumulated_tree = []
for batch in page_batches:
    prompt = instructions + batch_pages + json(accumulated_tree)
    new_nodes = LLM(prompt)
    accumulated_tree = merge(accumulated_tree, new_nodes)
```

The critical insight: we pass `accumulated_tree` back to the LLM on every call. The LLM can see what structure it already committed to. This prevents:
- Inconsistent numbering across batches
- Duplicate nodes at section boundaries
- Lost context when a section spans multiple batches

**Analogy**: AlphaGo evaluates the full board state at each move. This algorithm evaluates the full accumulated tree at each batch.

**Exercise**: Try running tree_builder without the accumulated_tree context (pass `[]` every time). Observe the inconsistent output.

---

### 3. Node summaries enable reasoning without fetching pages

During retrieval, the LLM sees the compact tree:
```
[0010] 4. Financial Analysis (pp 12-16)
  [Revenue grew 8.4%; EBITDA fell from 22% to 19%.]
  [0011] 4.1 Revenue Trends (pp 12-13)
  [0012] 4.2 Cost Structure (pp 13-14)
  [0013] 4.3 Profitability (pp 14-16)
```

The retriever can already reason: "EBITDA question → node 0013 Profitability". It doesn't need to fetch pages 12-16 first.

In vector RAG, every query requires fetching raw chunks to compute similarity. PageIndex separates the "navigate" step (summaries, free) from the "read" step (pages, token-expensive).

**Exercise**: Run retrieval with summaries disabled (`node.summary = None` for all nodes). Measure how many iterations it takes vs. with summaries.

---

### 4. The agentic retrieval loop

```python
state = RetrievalState(query)
while not state.is_sufficient and state.budget_remaining():
    node_id = LLM_select_node(query, tree, already_visited)
    page_text = fetch_pages(node_id)
    state.add_context(node_id, page_text)
    state.is_sufficient = LLM_check_sufficiency(query, state.context)
```

Two LLM calls per iteration: one to select, one to evaluate. This loop is what enables:
- **Cross-references**: "see Appendix G" → LLM navigates to appendix
- **Multi-hop questions**: answer needs data from two separate sections
- **Self-correction**: if first node was wrong, loop continues

**Exercise**: Force `state.is_sufficient = True` after 1 iteration. Compare answer quality vs. the full loop.

---

### 5. Similarity ≠ Relevance (the core thesis)

Vector RAG computes: `cosine(embed(query), embed(chunk))`

PageIndex computes: `LLM_reason("Given this tree structure and this query, which section is relevant?")`

The difference:
- A question about "EBITDA compression" might have low cosine similarity with a section titled "4.3 Profitability" if that section uses different vocabulary
- But an LLM reading the tree knows "Profitability" = EBITDA

**Exercise**: Build a test where cosine similarity would fail (query uses different words than the section title) and show PageIndex still finds the right section.

---

## Running the project

### Step 1: Install dependencies
```bash
pip install openai pymupdf python-dotenv
```

### Step 2: Set API key
```bash
export OPENAI_API_KEY=sk-...
# or for Anthropic:
export OPENAI_API_KEY=your-anthropic-key
export OPENAI_BASE_URL=https://api.anthropic.com/v1
export PAGEINDEX_MODEL=claude-sonnet-4-5
```

### Step 3: Run tests (no API key needed)
```bash
python main.py test
```

### Step 4: Build an index
```bash
python main.py build path/to/report.pdf
```

### Step 5: Query
```bash
python main.py query path/to/report.pdf "What was the revenue growth?" --explain
```

### Step 6: Run full demo (needs API key)
```bash
python main.py demo
```

---

## Exercises for deep understanding

### Beginner
1. Add a `depth()` method to `TreeNode` that returns the max depth of its subtree
2. Add a `to_markdown_outline()` method that renders the tree as a `##`/`###` outline
3. Write a test for `_merge_trees` with a 3-way merge

### Intermediate
4. Implement `retrieve_top_k(query, index, k=3)` that returns the top-k most relevant node_ids WITHOUT an agentic loop (use LLM to rank all nodes at once). Compare quality to the agentic loop.
5. Add token counting to `tree_builder.py` so batches stay within `max_tokens_per_node`
6. Implement async TOC detection that processes multiple page ranges in parallel

### Advanced
7. Replace the LLM node-selector with a BFS vs. DFS vs. best-first comparison. Which traversal strategy finds the right node in fewest iterations?
8. Implement `merge_indexes(index_a, index_b)` — merge two DocumentIndexes (e.g. two chapters of a book) into one
9. Add a vector fallback: if the agentic loop fails to find a sufficient answer in N iterations, fall back to embedding the leaf node summaries and running cosine search on them

---

## Key files to read in the real PageIndex codebase

Once you've built this, read these in the original repo to see the production version:

- `pageindex/page_index.py` — the real incremental tree builder (compare with `tree_builder.py`)
- `pageindex/retrieve.py` — the real retriever (compare with `retriever.py`)
- `pageindex/utils.py` — LLM helpers and config loading
- `cookbook/pageindex_RAG_simple.ipynb` — minimal end-to-end notebook

---

## Why this beats vector RAG for professional documents

| | Vector RAG | PageIndex |
|--|--|--|
| Retrieval unit | Fixed-size chunks | Natural sections |
| Navigation | Cosine similarity | LLM reasoning |
| Cross-references | Missed | Followed |
| Context integrity | Often broken at chunk boundaries | Preserved |
| Explainability | "Top-k similar chunks" | Exact node path |
| Multi-hop queries | Requires re-ranking tricks | Native (loop) |
| Accuracy on FinanceBench | ~70-80% | 98.7% |
