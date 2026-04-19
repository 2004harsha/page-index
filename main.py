#!/usr/bin/env python3
"""
main.py — Entry point for the PageIndex from Scratch project.

Usage:
  # Build an index for a document
  python main.py build path/to/document.pdf

  # Query a document
  python main.py query path/to/document.pdf "What was the revenue growth?"

  # Demo with synthetic document (no PDF needed, but needs API key)
  python main.py demo

  # Run all tests (no API key needed)
  python main.py test

  # Show the tree for a built document
  python main.py show path/to/document.pdf

Environment variables:
  OPENAI_API_KEY   — required for build/query/demo
  PAGEINDEX_MODEL  — model to use (default: gpt-4o)
                     Anthropic: set OPENAI_BASE_URL + ANTHROPIC_API_KEY via proxy
"""

import sys
import os
import argparse
from pathlib import Path


def cmd_test(_args):
    """Run all unit tests (no API key needed)."""
    print("Running PageIndex pipeline tests...\n")
    import tests
    failed = tests.results.failed
    sys.exit(0 if failed == 0 else 1)


def cmd_build(args):
    """Build a PageIndex for a document."""
    from pageindex_scratch import PageIndexPipeline

    pipeline = PageIndexPipeline(
        model=os.getenv("PAGEINDEX_MODEL", "gpt-4o"),
        index_dir=args.index_dir,
        skip_summarization=args.no_summarize,
    )
    index = pipeline.build(args.doc_path, force_rebuild=args.force)
    print(f"\n✓ Index saved to {pipeline._index_path_for(args.doc_path)}")


def cmd_query(args):
    """Query a built document index."""
    from pageindex_scratch import PageIndexPipeline

    pipeline = PageIndexPipeline(
        model=os.getenv("PAGEINDEX_MODEL", "gpt-4o"),
        index_dir=args.index_dir,
    )
    answer = pipeline.query(args.doc_path, args.question, explain=args.explain)
    print("\n" + "═" * 60)
    print("ANSWER")
    print("═" * 60)
    print(answer)
    print("═" * 60)


def cmd_show(args):
    """Display the tree structure of a built index."""
    from pageindex_scratch.models import DocumentIndex
    from pathlib import Path

    index_dir = Path(args.index_dir)
    stem = Path(args.doc_path).stem
    index_path = index_dir / f"{stem}_pageindex.json"

    if not index_path.exists():
        print(f"No index found at {index_path}")
        print(f"Run: python main.py build {args.doc_path}")
        sys.exit(1)

    index = DocumentIndex.load(str(index_path))
    print(f"\nDocument: {index.doc_path}")
    if index.description:
        print(f"Description: {index.description}")
    print(f"\nTree structure:")
    print("─" * 60)
    print(index.to_compact_repr())


def cmd_demo(_args):
    """
    End-to-end demo using a synthetic document.
    Requires an API key.
    """
    from pageindex_scratch.ingestion import create_test_document
    from pageindex_scratch import PageIndexPipeline
    import tempfile, json

    print("╔═══════════════════════════════════════════════════════╗")
    print("║     PageIndex from Scratch — End-to-End Demo          ║")
    print("╚═══════════════════════════════════════════════════════╝\n")

    # Create a synthetic text document so we don't need a real PDF
    doc = create_test_document(num_pages=20)

    # Save it as a text file
    with tempfile.NamedTemporaryFile(
        suffix=".txt", delete=False, mode='w', encoding='utf-8'
    ) as f:
        f.write("\n\n\n".join(p.text for p in doc.pages))
        tmp_path = f.name

    try:
        pipeline = PageIndexPipeline(
            model=os.getenv("PAGEINDEX_MODEL", "gpt-4o"),
            index_dir=tempfile.mkdtemp(),
            skip_summarization=False,
        )

        print("Phase 1: Building index...")
        index = pipeline.build(tmp_path)

        questions = [
            "What was the revenue growth and total market size?",
            "How did the Asia Pacific market perform?",
            "What were the main risk factors?",
        ]

        print("\nPhase 2: Querying...\n")
        for q in questions:
            print(f"Q: {q}")
            answer = pipeline.query(tmp_path, q, explain=True)
            print(f"A: {answer}\n")
            print("─" * 60)

    finally:
        os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(
        description="PageIndex from Scratch — Vectorless reasoning-based RAG"
    )
    parser.add_argument("--index-dir", default="./pageindex_cache",
                        help="Directory to store/load indexes")

    sub = parser.add_subparsers(dest="command")

    # test
    sub.add_parser("test", help="Run unit tests (no API key needed)")

    # build
    p_build = sub.add_parser("build", help="Build index for a document")
    p_build.add_argument("doc_path", help="Path to PDF, TXT, or MD file")
    p_build.add_argument("--force", action="store_true",
                         help="Force rebuild even if index exists")
    p_build.add_argument("--no-summarize", action="store_true",
                         help="Skip node summarization (faster, lower quality retrieval)")

    # query
    p_query = sub.add_parser("query", help="Query a built document index")
    p_query.add_argument("doc_path", help="Path to the document")
    p_query.add_argument("question", help="Question to answer")
    p_query.add_argument("--explain", action="store_true",
                         help="Show retrieval trace in answer")

    # show
    p_show = sub.add_parser("show", help="Display index tree structure")
    p_show.add_argument("doc_path", help="Path to the document")

    # demo
    sub.add_parser("demo", help="End-to-end demo with synthetic document")

    args = parser.parse_args()

    dispatch = {
        "test":  cmd_test,
        "build": cmd_build,
        "query": cmd_query,
        "show":  cmd_show,
        "demo":  cmd_demo,
    }

    if args.command not in dispatch:
        parser.print_help()
        sys.exit(0)

    dispatch[args.command](args)


if __name__ == "__main__":
    main()
