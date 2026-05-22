#!/usr/bin/env python
"""Standalone Naive Vector RAG pipeline.

This script mirrors the notebook end-to-end:
- PDF extraction
- chunking
- embedding
- retrieval
- 2D projection
- JSON export for backend and frontend
"""

from __future__ import annotations

import json
import re
import sys
import uuid
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BACKEND_EXPORT_DIR = ROOT / "backend_or_exports"
FRONTEND_DATA_DIR = ROOT / "frontend" / "public" / "data"
PDF_PATH = DATA_DIR / "MachineLearningNotes.pdf"


def parse_args():
    """Parse command-line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Naive Vector RAG Pipeline")
    parser.add_argument("--input", type=str, help="Path to input PDF file")
    return parser.parse_args()


def report_gpu() -> torch.device:
    print("Python", sys.version.split())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            print("CUDA device:", torch.cuda.get_device_name(0))
        except Exception:
            print("CUDA available")
    else:
        print("Using CPU")
    print("Embedding device ->", device)
    return device


def extract_pages(pdf_path: Path) -> list[dict]:
    import fitz

    assert pdf_path.exists(), f"PDF not found: {pdf_path}"
    doc = fitz.open(str(pdf_path))
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text().strip()
        pages.append({"page_number": i + 1, "text": text})
    pd.DataFrame(pages).head()
    return pages


def chunk_page_text(
    page_text: str, page_number: int, words_per_chunk: int = 650, overlap: int = 50
) -> list[dict]:
    tokens = re.findall(r"\S+", page_text)
    chunks = []
    if not tokens:
        return chunks

    step = words_per_chunk - overlap
    for i in range(0, max(1, len(tokens)), step):
        chunk_tokens = tokens[i : i + words_per_chunk]
        if not chunk_tokens:
            break
        text = " ".join(chunk_tokens).strip()
        chunk_id = str(uuid.uuid4())
        chunks.append(
            {
                "chunk_id": chunk_id,
                "page_number": page_number,
                "chunk_text": text,
                "word_count": len(chunk_tokens),
            }
        )
        if i + words_per_chunk >= len(tokens):
            break
    return chunks


def build_chunks(pages: list[dict]) -> list[dict]:
    chunks = []
    for page in pages:
        chunks.extend(chunk_page_text(page["text"], page["page_number"]))
    return chunks


def load_model(device: torch.device) -> tuple[SentenceTransformer, str]:
    candidate_models = [
        ("jinaai/jina-embeddings-v4", True),
        ("jinaai/jina-embeddings-v3", True),
        ("sentence-transformers/all-MiniLM-L6-v2", False),
    ]

    errors = []
    for name, use_remote_code in candidate_models:
        try:
            model = SentenceTransformer(
                name, device=str(device), trust_remote_code=use_remote_code
            )
            return model, name
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    print("Model loading attempts failed:")
    for err in errors:
        print("-", err)
    raise RuntimeError(
        "No embedding model could be loaded. Check internet/cache and package versions."
    )


def embed_texts(
    model: SentenceTransformer,
    texts: list[str],
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        emb = model.encode(batch, convert_to_tensor=True, normalize_embeddings=True)
        embeddings.append(emb)
    if len(embeddings) == 0:
        return torch.empty((0, model.get_sentence_embedding_dimension()), device=device)
    return torch.cat(embeddings, dim=0).to(device)


def retrieve_topk(
    model: SentenceTransformer,
    emb_t: torch.Tensor,
    chunks: list[dict],
    query: str,
    device: torch.device,
    topk: int = 5,
):
    q_t = model.encode([query], convert_to_tensor=True, normalize_embeddings=True).to(
        device
    )
    sims = torch.matmul(emb_t, q_t[0]).cpu().numpy()
    idxs = sims.argsort()[::-1][:topk]
    results = []
    for rank, idx in enumerate(idxs, 1):
        c = chunks[idx]
        results.append(
            {
                "rank": rank,
                "chunk_id": c["chunk_id"],
                "page_number": c["page_number"],
                "similarity_score": float(sims[idx]),
                "chunk_text_preview": c["chunk_text"][:300],
                "full_chunk_text": c["chunk_text"],
            }
        )
    return q_t, sims, results


def project_2d(emb_t: torch.Tensor, q_t: torch.Tensor):
    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=42)
        pts = reducer.fit_transform(emb_t.cpu().numpy())
    except Exception:
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2)
        pts = reducer.fit_transform(emb_t.cpu().numpy())

    try:
        q2 = reducer.transform(q_t.cpu().numpy())
    except Exception:
        q2 = q_t.cpu().numpy()

    return pts, q2


def export_outputs(
    chunks: list[dict],
    results: list[dict],
    query: str,
    q_t: torch.Tensor,
    vis_points: list[dict],
    query_point: dict,
):
    BACKEND_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)

    chunks_out = [
        {
            "chunk_id": c["chunk_id"],
            "page_number": c["page_number"],
            "chunk_text": c["chunk_text"],
            "word_count": c["word_count"],
            "preview": c["chunk_text"][:200],
        }
        for c in chunks
    ]

    query_out = {
        "query": query,
        "results": results,
        "query_embedding": q_t[0].cpu().numpy().tolist(),
    }

    vis = {"points": vis_points, "query_point": query_point}

    for base in (BACKEND_EXPORT_DIR, FRONTEND_DATA_DIR):
        with open(base / "naive_rag_chunks.json", "w", encoding="utf-8") as f:
            json.dump({"chunks": chunks_out}, f, ensure_ascii=False, indent=2)
        with open(base / "naive_rag_query_result.json", "w", encoding="utf-8") as f:
            json.dump(query_out, f, ensure_ascii=False, indent=2)
        with open(base / "naive_rag_vis.json", "w", encoding="utf-8") as f:
            json.dump(vis, f, ensure_ascii=False, indent=2)

    print("Wrote exports to ../backend_or_exports/ and ../frontend/public/data/")


def main():
    device = report_gpu()

    # Parse command-line arguments
    args = parse_args()
    pdf_path = Path(args.input) if args.input else PDF_PATH

    print(f"Processing PDF: {pdf_path}")
    print("Parsing PDF...")
    pages = extract_pages(pdf_path)
    print(f"Extracted {len(pages)} pages")

    print("Chunking text...")
    chunks = build_chunks(pages)
    print(f"Created {len(chunks)} chunks")

    print("Loading embedding model...")
    model, active_model_name = load_model(device)
    print("Loaded embedding model ->", active_model_name)

    print("Computing embeddings...")
    all_texts = [c["chunk_text"] for c in chunks]
    emb_t = embed_texts(model, all_texts, device)
    print("Embeddings computed; device ->", emb_t.device, "shape ->", emb_t.shape)

    print("Running retrieval...")
    query = "What is the main contribution of the paper?"
    q_t, sims, results = retrieve_topk(model, emb_t, chunks, query, device)
    print("Top results:", len(results))

    print("Projecting to 2D space...")
    pts, q2 = project_2d(emb_t, q_t)

    vis_points = []
    retrieved_ids = {r["chunk_id"] for r in results}
    for i, c in enumerate(chunks):
        vis_points.append(
            {
                "chunk_id": c["chunk_id"],
                "page_number": c["page_number"],
                "x": float(pts[i, 0]),
                "y": float(pts[i, 1]),
                "preview": c["chunk_text"][:200],
                "is_retrieved": c["chunk_id"] in retrieved_ids,
                "is_cited": False,
                "similarity_score": float(sims[i]),
            }
        )

    query_point = {
        "x": float(q2[0, 0]),
        "y": float(q2[0, 1]),
        "is_query": True,
        "query": query,
    }

    print("Exporting results...")
    export_outputs(chunks, results, query, q_t, vis_points, query_point)
    print("Pipeline completed successfully")


if __name__ == "__main__":
    main()
