"""
ingestion.py — Stage 1: Document Ingestion

WHAT IT DOES:
  Converts a raw PDF (or plain text file) into a Document object
  — an ordered list of PageContent objects, one per physical page.

WHY IT MATTERS:
  PageIndex is page-aware.  Unlike chunking-based RAG (which splits
  text into arbitrary token windows), PageIndex preserves physical
  page boundaries.  Every node in the tree stores `start_index` and
  `end_index` as *page numbers*, so retrieval maps back to exact pages.

  The physical_index tags are the key trick:
    <physical_index_5>
    ...page 5 text...
    </physical_index_5>
  When the LLM sees these tags during tree construction, it can tell
  the pipeline "this section starts at page 7 and ends at page 12"
  and those numbers are trustworthy.

LEARNING FOCUS:
  - How page boundaries are preserved
  - How tagged page strings are constructed
  - Why this differs from naive text splitting
"""

import re
from pathlib import Path
from typing import Optional

from .models import Document, PageContent


# ─────────────────────────────────────────────────
# Primary entry point
# ─────────────────────────────────────────────────

def ingest(path: str) -> Document:
    """
    Load a document from disk and return a Document.

    Supports:
      - .pdf  (via PyMuPDF if installed, else falls back to pdfminer)
      - .txt  (splits on double-newline as a page proxy)
      - .md   (splits on --- or level-1 headings as section boundaries)

    DESIGN NOTE:
      Real PageIndex uses a custom OCR pipeline that preserves the
      *visual* layout hierarchy (columns, headers, footnotes).
      Here we do a simpler extraction to keep the focus on the
      *indexing* logic rather than OCR engineering.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return _ingest_pdf(path)
    elif suffix == ".txt":
        return _ingest_txt(path)
    elif suffix == ".md":
        return _ingest_md(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


# ─────────────────────────────────────────────────
# PDF ingestion
# ─────────────────────────────────────────────────

def _ingest_pdf(path: str) -> Document:
    """
    Extract text per physical page from a PDF.

    Uses PyMuPDF (fitz) — fast and layout-aware.
    Falls back to pdfminer if fitz is unavailable.
    """
    try:
        import fitz  # PyMuPDF
        return _ingest_pdf_pymupdf(path)
    except ImportError:
        pass

    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextContainer
        return _ingest_pdf_pdfminer(path)
    except ImportError:
        raise ImportError(
            "PDF ingestion requires either PyMuPDF or pdfminer.six.\n"
            "  pip install pymupdf        (recommended)\n"
            "  pip install pdfminer.six   (fallback)"
        )


def _ingest_pdf_pymupdf(path: str) -> Document:
    import fitz
    doc_fitz = fitz.open(path)
    pages = []
    for page_num, page in enumerate(doc_fitz):
        text = page.get_text("text").strip()
        # Clean up common PDF artifacts
        text = _clean_pdf_text(text)
        pages.append(PageContent(page_num=page_num, text=text))
    doc_fitz.close()
    return Document(path=path, pages=pages)


def _ingest_pdf_pdfminer(path: str) -> Document:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer

    pages = []
    for page_num, page_layout in enumerate(extract_pages(path)):
        texts = []
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                texts.append(element.get_text())
        text = _clean_pdf_text("".join(texts))
        pages.append(PageContent(page_num=page_num, text=text))
    return Document(path=path, pages=pages)


# ─────────────────────────────────────────────────
# Plain text ingestion
# ─────────────────────────────────────────────────

def _ingest_txt(path: str) -> Document:
    """
    Treat a plain .txt file as a document where a blank line
    (two consecutive newlines) separates 'pages'.

    This is useful for testing the pipeline without a real PDF.
    """
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    # Split on double blank lines
    chunks = re.split(r"\n{3,}", raw.strip())
    pages = [
        PageContent(page_num=i, text=chunk.strip())
        for i, chunk in enumerate(chunks)
        if chunk.strip()
    ]
    return Document(path=path, pages=pages)


# ─────────────────────────────────────────────────
# Markdown ingestion
# ─────────────────────────────────────────────────

def _ingest_md(path: str) -> Document:
    """
    Split a markdown file into sections at # headings or --- rules.

    DESIGN NOTE:
      PageIndex recommends using their own OCR to convert PDF→MD
      to preserve heading hierarchy.  If you convert with a generic
      tool and use this mode, hierarchy may be lost.
      For this project, we trust the markdown structure.
    """
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    # Split at top-level headings or horizontal rules
    sections = re.split(r"(?m)^(?=#{1,2} )|^---\s*$", raw)
    pages = [
        PageContent(page_num=i, text=section.strip())
        for i, section in enumerate(sections)
        if section.strip()
    ]
    return Document(path=path, pages=pages)


# ─────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────

def _clean_pdf_text(text: str) -> str:
    """
    Remove common PDF extraction noise.

    LEARNING NOTE:
      PDF text extraction is imperfect.  Common issues:
        - Ligatures: "ﬁ" instead of "fi"
        - Hyphenated line breaks: "impor-\ntant"
        - Header/footer repetition every page
      We do minimal cleaning here; production systems do much more.
    """
    # Fix common ligatures
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff")
    text = text.replace("ﬃ", "ffi").replace("ﬄ", "ffl")

    # Remove hyphenation at line breaks (e.g. "impor-\ntant" → "important")
    text = re.sub(r"-\n(\w)", r"\1", text)

    # Collapse excessive whitespace but preserve paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def create_test_document(num_pages: int = 20) -> Document:
    """
    Create a synthetic Document for unit testing without needing a real PDF.

    The content mimics a typical technical report structure — useful for
    testing tree construction and retrieval without API calls.
    """
    sections = [
        ("Abstract", "This report examines market trends across three verticals."),
        ("1. Introduction", "The global economy in 2024 showed significant shifts."),
        ("1.1 Scope", "This analysis covers Q1–Q4 2024 across 40 countries."),
        ("1.2 Methodology", "Data was collected from 500 primary sources."),
        ("2. Market Overview", "Total addressable market reached $4.2 trillion."),
        ("2.1 North America", "Revenue grew 12% YoY to $1.8 trillion."),
        ("2.2 Europe", "EU markets contracted 3% amid regulatory headwinds."),
        ("2.3 Asia Pacific", "APAC surged 24% driven by semiconductor demand."),
        ("3. Technology Sector", "Tech remained the dominant growth driver."),
        ("3.1 AI & ML", "Enterprise AI spend tripled to $280 billion."),
        ("3.2 Cloud Infrastructure", "Cloud providers added 15 new regions."),
        ("3.3 Semiconductors", "Chip shortages persisted in automotive sector."),
        ("4. Financial Analysis", "Net margins compressed across most sectors."),
        ("4.1 Revenue Trends", "Aggregate revenue grew 8.4% to $12.7 trillion."),
        ("4.2 Cost Structure", "COGS increased 11% due to supply chain costs."),
        ("4.3 Profitability", "EBITDA margins fell from 22% to 19% on average."),
        ("5. Risk Factors", "Macro risks include rising rates and geopolitical tensions."),
        ("6. Outlook", "2025 growth projected at 6–9% across covered sectors."),
        ("Appendix A: Data Tables", "Full dataset available in attached spreadsheet."),
        ("Appendix B: Methodology Notes", "Statistical methods follow ISO 9001 guidelines."),
    ]

    pages = []
    for i in range(num_pages):
        if i < len(sections):
            title, body = sections[i]
            text = f"{title}\n\n{body}\n\nPage {i + 1} of {num_pages}."
        else:
            text = f"[Continuation of section content — page {i + 1}]"
        pages.append(PageContent(page_num=i, text=text))

    return Document(path="<synthetic_test_document>", pages=pages)
