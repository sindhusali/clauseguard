"""
ClauseGuard — Step 3 (updated)
Parse a contract PDF into clause-level chunks with page citations AND the
actual clause number parsed from the heading itself (e.g. "1. License
Grant" -> clause_number=1). Previously only the page number was tracked as
verified metadata, and the LLM was left to guess clause numbers from context
at answer time — which occasionally produced wrong numbers even when the
page citation was correct. This fixes that by capturing the real number
during parsing, once, so every later citation is grounded in real data.

Runs locally, no GPU needed.
"""

import re
import sys
import json
from dataclasses import dataclass, asdict
from pypdf import PdfReader


@dataclass
class Clause:
    chunk_id: str
    text: str
    page_number: int
    paragraph_index_on_page: int
    clause_number: int | None  # None if this chunk isn't a numbered clause (e.g. title/intro text)


MIN_CHUNK_CHARS = 40

# Matches a leading clause number like "1." "12." "3.1" at the start of a chunk.
CLAUSE_NUMBER_RE = re.compile(r"^\s*(\d+)(?:\.\d+)*\.\s")


def extract_pages(pdf_path):
    reader = PdfReader(pdf_path)
    return [page.extract_text() or "" for page in reader.pages]


def split_into_paragraphs(page_text):
    rough_paragraphs = re.split(r"\n\s*\n", page_text)
    paragraphs = []
    for para in rough_paragraphs:
        pieces = re.split(r"(?=\n?\s*\d+\.\d*\s)", para)
        paragraphs.extend(p.strip() for p in pieces if p.strip())
    return paragraphs


def extract_clause_number(paragraph_text: str) -> int | None:
    match = CLAUSE_NUMBER_RE.match(paragraph_text)
    if match:
        return int(match.group(1))
    return None


def chunk_contract(pdf_path):
    pages = extract_pages(pdf_path)
    clauses = []
    for page_number, page_text in enumerate(pages, start=1):
        paragraphs = split_into_paragraphs(page_text)
        for para_idx, para in enumerate(paragraphs, start=1):
            if len(para) < MIN_CHUNK_CHARS:
                continue
            clause_number = extract_clause_number(para)
            clauses.append(Clause(
                chunk_id=f"p{page_number}_c{para_idx}",
                text=para,
                page_number=page_number,
                paragraph_index_on_page=para_idx,
                clause_number=clause_number,
            ))
    return clauses


def main():
    if len(sys.argv) < 2:
        print("Usage: python pdf_chunking.py ../sample_contracts/contract.pdf")
        return

    pdf_path = sys.argv[1]
    clauses = chunk_contract(pdf_path)
    print(f"Extracted {len(clauses)} clause-level chunks from {pdf_path}\n")

    for clause in clauses[:5]:
        clause_label = f"Clause {clause.clause_number}" if clause.clause_number else "no clause number"
        print(f"[{clause.chunk_id}] (page {clause.page_number}, {clause_label}): {clause.text[:120]}...")

    out_path = pdf_path.rsplit(".", 1)[0] + "_chunks.json"
    with open(out_path, "w") as f:
        json.dump([asdict(c) for c in clauses], f, indent=2)
    print(f"\nSaved all chunks to {out_path}")


if __name__ == "__main__":
    main()
