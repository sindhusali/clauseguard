"""
ClauseGuard — Step 3
Parse a contract PDF into clause-level chunks with page citations.
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


MIN_CHUNK_CHARS = 40


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


def chunk_contract(pdf_path):
    pages = extract_pages(pdf_path)
    clauses = []
    for page_number, page_text in enumerate(pages, start=1):
        paragraphs = split_into_paragraphs(page_text)
        for para_idx, para in enumerate(paragraphs, start=1):
            if len(para) < MIN_CHUNK_CHARS:
                continue
            clauses.append(Clause(f"p{page_number}_c{para_idx}", para, page_number, para_idx))
    return clauses


def main():
    if len(sys.argv) < 2:
        print("Usage: python pdf_chunking.py ../sample_contracts/contract.pdf")
        return

    pdf_path = sys.argv[1]
    clauses = chunk_contract(pdf_path)
    print(f"Extracted {len(clauses)} clause-level chunks from {pdf_path}\n")

    for clause in clauses[:5]:
        print(f"[{clause.chunk_id}] (page {clause.page_number}): {clause.text[:120]}...")

    out_path = pdf_path.rsplit(".", 1)[0] + "_chunks.json"
    with open(out_path, "w") as f:
        json.dump([asdict(c) for c in clauses], f, indent=2)
    print(f"\nSaved all chunks to {out_path}")


if __name__ == "__main__":
    main()