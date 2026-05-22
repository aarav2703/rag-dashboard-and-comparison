#!/usr/bin/env python
"""Standalone BM25 / lexical retrieval pipeline.

This script mirrors the naive pipeline structure but replaces embedding-based
ranking with exact lexical BM25 scoring and exports lexical diagnostics:
- keyword evidence highlights
- term rarity bars
- matched-token contribution breakdown
- missing-query-term warnings
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BACKEND_EXPORT_DIR = ROOT / "backend_or_exports"
FRONTEND_DATA_DIR = ROOT / "frontend" / "public" / "data"
DEFAULT_PDF_PATH = DATA_DIR / "MachineLearningNotes.pdf"
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BM25 lexical retrieval pipeline")
    parser.add_argument("--input", type=str, help="Path to input PDF file")
    return parser.parse_args()


def report_environment() -> None:
    print("Python", sys.version.split())
    print("Working directory ->", os.getcwd())


def extract_pages(pdf_path: Path) -> list[dict]:
    import fitz

    assert pdf_path.exists(), f"PDF not found: {pdf_path}"
    doc = fitz.open(str(pdf_path))
    pages = []
    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        text = page.get_text("text") or ""
        pages.append(
            {
                "page_number": page_index + 1,
                "text": text.strip(),
            }
        )
    return pages


def chunk_page_text(text: str, chunk_words: int = 650, overlap_words: int = 50) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start = max(0, end - overlap_words)
    return chunks


def build_chunks(pages: list[dict]) -> list[dict]:
    chunks = []
    for page in pages:
        page_number = page["page_number"]
        page_text = page["text"]
        page_chunks = chunk_page_text(page_text)
        for chunk_index, chunk_text in enumerate(page_chunks):
            chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "chunk_text": chunk_text,
                    "word_count": len(chunk_text.split()),
                    "preview": chunk_text[:220],
                }
            )
    return chunks


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def unique_preserve_order(tokens: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


def build_corpus_stats(chunks: list[dict]) -> dict:
    doc_tokens = []
    document_frequency = Counter()
    total_length = 0

    for chunk in chunks:
        tokens = tokenize(chunk["chunk_text"])
        doc_tokens.append(tokens)
        total_length += len(tokens)
        document_frequency.update(set(tokens))

    doc_count = max(1, len(doc_tokens))
    avgdl = total_length / doc_count

    idf = {
        term: math.log(1.0 + ((doc_count - freq + 0.5) / (freq + 0.5)))
        for term, freq in document_frequency.items()
    }

    return {
        "doc_tokens": doc_tokens,
        "document_frequency": document_frequency,
        "doc_count": doc_count,
        "avgdl": avgdl,
        "idf": idf,
    }


def bm25_score(doc_tokens: list[str], query_terms: list[str], stats: dict, k1: float = 1.5, b: float = 0.75) -> tuple[float, list[dict], list[str]]:
    tf = Counter(doc_tokens)
    doc_len = len(doc_tokens) or 1
    avgdl = stats["avgdl"] or 1.0
    document_frequency = stats["document_frequency"]
    idf_map = stats["idf"]

    score = 0.0
    contributions = []
    matched_terms = []

    for term in query_terms:
        term_tf = tf.get(term, 0)
        if term_tf <= 0:
            continue

        matched_terms.append(term)
        df = document_frequency.get(term, 0)
        idf = idf_map.get(term, math.log(1.0 + ((stats["doc_count"] - 0 + 0.5) / 0.5)))
        denominator = term_tf + k1 * (1.0 - b + b * (doc_len / avgdl))
        contribution = idf * ((term_tf * (k1 + 1.0)) / denominator)
        score += contribution
        contributions.append(
            {
                "term": term,
                "term_frequency": term_tf,
                "document_frequency": df,
                "idf": round(idf, 6),
                "score": round(contribution, 6),
            }
        )

    contributions.sort(key=lambda item: (-item["score"], item["term"]))
    return score, contributions, unique_preserve_order(matched_terms)


def find_highlight_spans(text: str, query_terms: list[str]) -> list[dict]:
    spans = []
    seen = set()
    for term in sorted(query_terms, key=len, reverse=True):
        if term in seen:
            continue
        seen.add(term)
        pattern = re.compile(rf"(?i)(?<!\w){re.escape(term)}(?!\w)")
        for match in pattern.finditer(text):
            spans.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "term": term,
                    "matched_text": text[match.start() : match.end()],
                }
            )
    spans.sort(key=lambda item: (item["start"], item["end"]))
    return spans


def build_query_term_stats(query_terms: list[str], stats: dict) -> list[dict]:
    document_frequency = stats["document_frequency"]
    idf_map = stats["idf"]
    max_idf = max(idf_map.values(), default=1.0) or 1.0

    rows = []
    for term in query_terms:
        df = int(document_frequency.get(term, 0))
        idf = float(idf_map.get(term, math.log(1.0 + ((stats["doc_count"] - 0 + 0.5) / 0.5))))
        rows.append(
            {
                "term": term,
                "document_frequency": df,
                "idf": round(idf, 6),
                "rarity": round(idf / max_idf, 6),
                "matched": df > 0,
            }
        )
    rows.sort(key=lambda item: (-item["idf"], item["term"]))
    return rows


def export_outputs(chunks: list[dict], query_out: dict, vis: dict) -> None:
    BACKEND_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)

    chunks_out = {
        "mode": "bm25",
        "pipeline": "bm25_lexical_rag",
        "chunk_count": len(chunks),
        "chunks": chunks,
    }

    for directory in (BACKEND_EXPORT_DIR, FRONTEND_DATA_DIR):
        with open(directory / "bm25_lexical_chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_out, f, indent=2, ensure_ascii=False)
        with open(directory / "bm25_lexical_query_result.json", "w", encoding="utf-8") as f:
            json.dump(query_out, f, indent=2, ensure_ascii=False)
        with open(directory / "bm25_lexical_vis.json", "w", encoding="utf-8") as f:
            json.dump(vis, f, indent=2, ensure_ascii=False)

    print("Exported -> bm25_lexical_chunks.json, bm25_lexical_query_result.json, bm25_lexical_vis.json")


def main() -> None:
    report_environment()
    args = parse_args()
    pdf_path = Path(args.input) if args.input else DEFAULT_PDF_PATH

    print(f"Processing PDF: {pdf_path}")
    print("Parsing PDF...")
    pages = extract_pages(pdf_path)
    print(f"Extracted {len(pages)} pages")

    print("Chunking text...")
    chunks = build_chunks(pages)
    print(f"Created {len(chunks)} chunks")

    print("Building lexical index...")
    stats = build_corpus_stats(chunks)
    print(f"Average document length -> {stats['avgdl']:.2f}")

    query = "What is the main contribution of the paper?"
    print("Ranking with BM25...")
    query_terms = unique_preserve_order(tokenize(query))
    missing_query_terms = [term for term in query_terms if stats["document_frequency"].get(term, 0) == 0]
    if missing_query_terms:
        print("Missing query terms:", ", ".join(missing_query_terms))

    scored_results = []
    for chunk in chunks:
        doc_tokens = tokenize(chunk["chunk_text"])
        score, term_contributions, matched_terms = bm25_score(doc_tokens, query_terms, stats)
        scored_results.append(
            {
                **chunk,
                "bm25_score": round(score, 6),
                "matched_terms": matched_terms,
                "term_contributions": term_contributions,
                "highlight_spans": find_highlight_spans(chunk["chunk_text"], matched_terms),
            }
        )

    scored_results.sort(key=lambda item: (-item["bm25_score"], item["page_number"], item["chunk_index"]))
    results = []
    for rank, item in enumerate(scored_results[:5], start=1):
        results.append(
            {
                "rank": rank,
                "chunk_id": item["chunk_id"],
                "page_number": item["page_number"],
                "bm25_score": item["bm25_score"],
                "chunk_text_preview": item["preview"],
                "full_chunk_text": item["chunk_text"],
                "matched_terms": item["matched_terms"],
                "term_contributions": item["term_contributions"],
                "highlight_spans": item["highlight_spans"],
                "term_hit_count": len(item["matched_terms"]),
            }
        )

    top_result_contributions = results[0]["term_contributions"] if results else []
    term_stats = build_query_term_stats(query_terms, stats)

    query_out = {
        "mode": "bm25",
        "pipeline": "bm25_lexical_rag",
        "query": query,
        "query_terms": query_terms,
        "missing_query_terms": missing_query_terms,
        "missing_query_warning": bool(missing_query_terms),
        "term_stats": term_stats,
        "top_result_term_contributions": top_result_contributions,
        "results": results,
    }

    vis = {
        "mode": "bm25",
        "pipeline": "bm25_lexical_rag",
        "query": query,
        "query_terms": query_terms,
        "missing_query_terms": missing_query_terms,
        "term_stats": term_stats,
        "top_result_term_contributions": top_result_contributions,
        "warning": (
            "Missing query terms: " + ", ".join(missing_query_terms)
            if missing_query_terms
            else "All query terms were observed in the corpus."
        ),
    }

    print("Top results:", len(results))
    print("Exporting results...")
    export_outputs(chunks, query_out, vis)
    print("Pipeline completed successfully")


if __name__ == "__main__":
    main()
