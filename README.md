# PageIndex from Scratch

A learning implementation of VectifyAI's PageIndex system: a **vectorless, reasoning-based RAG** pipeline for long documents.

## What this project does

PageIndex builds a hierarchical tree over a document instead of splitting text into fixed-size chunks. The pipeline is designed to:

- preserve natural document structure
- keep page boundaries intact
- use LLM reasoning for navigation instead of embedding similarity
- support document indexing, querying, and tree inspection from the command line

## Project layout

```text
pageindex_project/
├── main.py              # CLI entry point
├── tests.py             # Tests that run without an API key
├── ROADMAP.md           # Learning roadmap and design notes
├── requirements.txt     # Python dependencies
├── .env                 # Local environment variables
├── .gitignore
└── pageindex_scratch/
    ├── __init__.py
    ├── generator.py
    ├── ingestion.py
    ├── llm_client.py
    ├── models.py
    ├── pipeline.py
    ├── retriever.py
    ├── summarizer.py
    ├── toc_detector.py
    └── tree_builder.py
```

## Requirements

- Python 3.10+ recommended
- An OpenAI-compatible API key for `build`, `query`, and `demo`
- Optional: a PDF reader backend via `pymupdf` or `pdfminer.six`

## Setup

### 1. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root or edit the provided template:

```env
OPENAI_API_KEY=your_openai_api_key_here
PAGEINDEX_MODEL=gpt-4o
```

Optional Anthropic-through-proxy configuration:

```env
OPENAI_BASE_URL=https://api.anthropic.com/v1
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Usage

All commands are run from the project root.

### Run tests

No API key is required.

```powershell
python main.py test
```

### Build an index

```powershell
python main.py build path\to\document.pdf
```

Useful flags:

- `--force` rebuilds even if an index already exists
- `--no-summarize` skips node summarization for a faster but lower-quality index
- `--index-dir` changes where indexes are stored

### Query a document

```powershell
python main.py query path\to\document.pdf "What was the revenue growth?" --explain
```

### Show the tree for a built document

```powershell
python main.py show path\to\document.pdf
```

### Run the synthetic demo

```powershell
python main.py demo
```

## How it works

The pipeline is split into stages:

1. Ingestion converts a PDF, TXT, or MD document into page-level content.
2. TOC detection scans the document for an existing structure.
3. Tree building constructs a hierarchical index.
4. Summarization adds compact node summaries.
5. Retrieval navigates the tree with LLM reasoning.
6. Generation answers the question from the retrieved context.

See [ROADMAP.md](ROADMAP.md) for a deeper explanation of the architecture and the learning exercises.

## Testing

The tests are designed to run without an API key by mocking LLM calls. They cover:

- document and tree data models
- ingestion helpers
- TOC detection parsing
- tree construction and merging
- summarization collection
- retrieval explanation helpers

If you want to run them directly:

```powershell
python tests.py
```

## Notes

- Keep `.env` out of version control; it contains secrets.
- The default cache directory is `./pageindex_cache`.
- Built indexes are stored separately from source documents.

## License

No license file is included yet. Add one if you plan to share or publish this project.
