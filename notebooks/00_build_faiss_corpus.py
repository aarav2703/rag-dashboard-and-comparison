#!/usr/bin/env python
"""Build the shared PDF corpus and FAISS vector index."""

from __future__ import annotations

import argparse
from pathlib import Path

from shared_rag_store import build_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build shared FAISS corpus")
    parser.add_argument("--input", required=True, type=str, help="Path to input PDF")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path = Path(args.input)

    print(f"Processing PDF: {pdf_path}")
    print("Parsing PDF...")
    print("Chunking text...")
    print("Embedding chunks...")
    print("Building FAISS index...")
    summary = build_store(pdf_path)
    print(f"Pages indexed: {summary['page_count']}")
    print(f"Chunks indexed: {summary['chunk_count']}")
    print(f"Embedding model: {summary['embedding_model']}")
    print(f"Embedding device: {summary['embedding_device']}")
    print("Exporting shared corpus artifacts...")
    print("Pipeline completed successfully")


if __name__ == "__main__":
    main()
