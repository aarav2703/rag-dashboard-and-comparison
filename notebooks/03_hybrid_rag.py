#!/usr/bin/env python
"""Standalone Hybrid RAG pipeline.

Combines semantic vector retrieval with BM25 lexical retrieval, then exports
merge diagnostics for the frontend:
- dual-source merge Sankey
- vector-only / BM25-only / both source badges
- overlap matrix
- rank-fusion table
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import uuid
from collections import Counter
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BACKEND_EXPORT_DIR = ROOT / "backend_or_exports"
FRONTEND_DATA_DIR = ROOT / "frontend" / "public" / "data"
DEFAULT_PDF_PATH = DATA_DIR / "MachineLearningNotes.pdf"
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
DEFAULT_QUERY = "What is the main contribution of the paper?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid vector + BM25 RAG pipeline")
    parser.add_argument("--input", type=str, help="Path to input PDF file")
    return parser.parse_args()


def report_environment() -> torch.device:
    print("Python", sys.version.split())
    print("Working directory ->", os.getcwd())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Embedding device ->", device)
    return device


def extract_pages(pdf_path: Path) -> list[dict]:
    import fitz

    assert pdf_path.exists(), f"PDF not found: {pdf_path}"
    doc = fitz.open(str(pdf_path))
    pages = []
    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        pages.append(
            {
                "page_number": page_index + 1,
                "text": (page.get_text("text") or "").strip(),
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
        for chunk_index, chunk_text in enumerate(chunk_page_text(page["text"])):
            chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "page_number": page["page_number"],
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


def build_bm25_stats(chunks: list[dict]) -> dict:
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


def bm25_score(doc_tokens: list[str], query_terms: list[str], stats: dict, k1: float = 1.5, b: float = 0.75) -> tuple[float, list[str]]:
    tf = Counter(doc_tokens)
    doc_len = len(doc_tokens) or 1
    avgdl = stats["avgdl"] or 1.0
    score = 0.0
    matched_terms = []

    for term in query_terms:
        term_tf = tf.get(term, 0)
        if term_tf <= 0:
            continue
        matched_terms.append(term)
        idf = stats["idf"].get(term, math.log(1.0 + ((stats["doc_count"] + 0.5) / 0.5)))
        denominator = term_tf + k1 * (1.0 - b + b * (doc_len / avgdl))
        score += idf * ((term_tf * (k1 + 1.0)) / denominator)

    return score, unique_preserve_order(matched_terms)


def load_model(device: torch.device) -> tuple[SentenceTransformer, str]:
    candidates = [
        ("jinaai/jina-embeddings-v4", True),
        ("jinaai/jina-embeddings-v3", True),
        ("sentence-transformers/all-MiniLM-L6-v2", False),
    ]
    errors = []
    for name, trust_remote_code in candidates:
        try:
            model = SentenceTransformer(name, device=str(device), trust_remote_code=trust_remote_code)
            return model, name
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    print("Model loading attempts failed:")
    for error in errors:
        print("-", error)
    raise RuntimeError("No embedding model could be loaded.")


def embed_texts(model: SentenceTransformer, texts: list[str], device: torch.device, batch_size: int = 64) -> torch.Tensor:
    embeddings = []
    for start in range(0, len(texts), batch_size):
        emb = model.encode(texts[start : start + batch_size], convert_to_tensor=True, normalize_embeddings=True)
        embeddings.append(emb)
    if not embeddings:
        return torch.empty((0, model.get_sentence_embedding_dimension()), device=device)
    return torch.cat(embeddings, dim=0).to(device)


def normalize_scores(rows: list[dict], key: str, target_key: str) -> None:
    values = [row[key] for row in rows]
    min_value = min(values, default=0.0)
    max_value = max(values, default=0.0)
    span = max_value - min_value
    for row in rows:
        row[target_key] = 0.0 if span == 0 else (row[key] - min_value) / span


def reciprocal_rank(rank: int | None, k: int = 60) -> float:
    if rank is None:
        return 0.0
    return 1.0 / (k + rank)


def source_label(vector_rank: int | None, bm25_rank: int | None) -> str:
    if vector_rank and bm25_rank:
        return "both"
    if vector_rank:
        return "vector-only"
    return "bm25-only"


def build_hybrid(chunks: list[dict], query: str, device: torch.device) -> tuple[dict, dict]:
    print("Building lexical index...")
    bm25_stats = build_bm25_stats(chunks)
    query_terms = unique_preserve_order(tokenize(query))

    print("Loading embedding model...")
    model, model_name = load_model(device)
    print("Loaded embedding model ->", model_name)

    print("Computing embeddings...")
    emb_t = embed_texts(model, [chunk["chunk_text"] for chunk in chunks], device)
    q_t = model.encode([query], convert_to_tensor=True, normalize_embeddings=True).to(device)
    vector_scores = torch.matmul(emb_t, q_t[0]).cpu().numpy()

    scored_rows = []
    for index, chunk in enumerate(chunks):
        bm25_raw, matched_terms = bm25_score(bm25_stats["doc_tokens"][index], query_terms, bm25_stats)
        scored_rows.append(
            {
                **chunk,
                "vector_score": float(vector_scores[index]),
                "bm25_score": float(bm25_raw),
                "matched_terms": matched_terms,
            }
        )

    normalize_scores(scored_rows, "vector_score", "vector_norm")
    normalize_scores(scored_rows, "bm25_score", "bm25_norm")

    vector_sorted = sorted(scored_rows, key=lambda row: row["vector_score"], reverse=True)
    bm25_sorted = sorted(scored_rows, key=lambda row: row["bm25_score"], reverse=True)
    vector_ranks = {row["chunk_id"]: rank for rank, row in enumerate(vector_sorted[:10], start=1)}
    bm25_ranks = {row["chunk_id"]: rank for rank, row in enumerate(bm25_sorted[:10], start=1)}

    candidate_ids = set(vector_ranks) | set(bm25_ranks)
    candidates = []
    for row in scored_rows:
        if row["chunk_id"] not in candidate_ids:
            continue
        vector_rank = vector_ranks.get(row["chunk_id"])
        bm25_rank = bm25_ranks.get(row["chunk_id"])
        fusion_score = (
            0.5 * row["vector_norm"]
            + 0.5 * row["bm25_norm"]
            + reciprocal_rank(vector_rank)
            + reciprocal_rank(bm25_rank)
        )
        candidates.append(
            {
                **row,
                "vector_rank": vector_rank,
                "bm25_rank": bm25_rank,
                "fusion_score": round(fusion_score, 6),
                "source": source_label(vector_rank, bm25_rank),
            }
        )

    candidates.sort(key=lambda row: (-row["fusion_score"], row["page_number"], row["chunk_index"]))
    results = []
    for rank, row in enumerate(candidates[:5], start=1):
        results.append(
            {
                "rank": rank,
                "chunk_id": row["chunk_id"],
                "page_number": row["page_number"],
                "hybrid_score": row["fusion_score"],
                "vector_score": round(row["vector_score"], 6),
                "bm25_score": round(row["bm25_score"], 6),
                "vector_rank": row["vector_rank"],
                "bm25_rank": row["bm25_rank"],
                "source": row["source"],
                "matched_terms": row["matched_terms"],
                "chunk_text_preview": row["preview"],
                "full_chunk_text": row["chunk_text"],
            }
        )

    source_counts = Counter(candidate["source"] for candidate in candidates)
    final_counts = Counter(result["source"] for result in results)
    overlap = {
        "vector_candidates": len(vector_ranks),
        "bm25_candidates": len(bm25_ranks),
        "both": source_counts.get("both", 0),
        "vector_only": source_counts.get("vector-only", 0),
        "bm25_only": source_counts.get("bm25-only", 0),
    }

    rank_fusion_table = [
        {
            "rank": index + 1,
            "chunk_id": row["chunk_id"],
            "page_number": row["page_number"],
            "source": row["source"],
            "vector_rank": row["vector_rank"],
            "bm25_rank": row["bm25_rank"],
            "vector_score": round(row["vector_score"], 6),
            "bm25_score": round(row["bm25_score"], 6),
            "fusion_score": row["fusion_score"],
            "preview": row["preview"],
        }
        for index, row in enumerate(candidates[:10])
    ]

    query_out = {
        "mode": "hybrid",
        "pipeline": "hybrid_rag",
        "query": query,
        "query_terms": query_terms,
        "results": results,
        "source_counts": dict(source_counts),
        "overlap": overlap,
        "rank_fusion_table": rank_fusion_table,
    }

    vis = {
        "mode": "hybrid",
        "pipeline": "hybrid_rag",
        "query": query,
        "merge_sankey": {
            "nodes": [
                {"name": "Vector candidates"},
                {"name": "BM25 candidates"},
                {"name": "Merged candidates"},
                {"name": "Final evidence"},
            ],
            "links": [
                {"source": 0, "target": 2, "value": max(1, len(vector_ranks))},
                {"source": 1, "target": 2, "value": max(1, len(bm25_ranks))},
                {"source": 2, "target": 3, "value": max(1, len(results))},
            ],
        },
        "overlap": overlap,
        "overlap_matrix": [
            {"source": "vector-only", "count": overlap["vector_only"]},
            {"source": "bm25-only", "count": overlap["bm25_only"]},
            {"source": "both", "count": overlap["both"]},
        ],
        "final_source_counts": dict(final_counts),
        "rank_fusion_table": rank_fusion_table,
    }
    return query_out, vis


def export_outputs(chunks: list[dict], query_out: dict, vis: dict) -> None:
    BACKEND_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)

    chunks_out = {
        "mode": "hybrid",
        "pipeline": "hybrid_rag",
        "chunk_count": len(chunks),
        "chunks": chunks,
    }

    for directory in (BACKEND_EXPORT_DIR, FRONTEND_DATA_DIR):
        with open(directory / "hybrid_rag_chunks.json", "w", encoding="utf-8") as file:
            json.dump(chunks_out, file, indent=2, ensure_ascii=False)
        with open(directory / "hybrid_rag_query_result.json", "w", encoding="utf-8") as file:
            json.dump(query_out, file, indent=2, ensure_ascii=False)
        with open(directory / "hybrid_rag_vis.json", "w", encoding="utf-8") as file:
            json.dump(vis, file, indent=2, ensure_ascii=False)

    print("Exported -> hybrid_rag_chunks.json, hybrid_rag_query_result.json, hybrid_rag_vis.json")


def main() -> None:
    device = report_environment()
    args = parse_args()
    pdf_path = Path(args.input) if args.input else DEFAULT_PDF_PATH

    print(f"Processing PDF: {pdf_path}")
    print("Parsing PDF...")
    pages = extract_pages(pdf_path)
    print(f"Extracted {len(pages)} pages")

    print("Chunking text...")
    chunks = build_chunks(pages)
    print(f"Created {len(chunks)} chunks")

    print("Retrieving vector and BM25 candidates...")
    query_out, vis = build_hybrid(chunks, DEFAULT_QUERY, device)

    print("Exporting results...")
    export_outputs(chunks, query_out, vis)
    print("Pipeline completed successfully")


if __name__ == "__main__":
    main()
