#!/usr/bin/env python
"""Build the app's shared retrieval store from HotpotQA benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"
ARTIFACTS_DIR = ROOT / "benchmarking" / "artifacts"
STORE_DIR = ROOT / "backend_or_exports" / "current_corpus"
FRONTEND_DATA_DIR = ROOT / "frontend" / "public" / "data"


MODE_PREFIXES = (
    "naive_rag",
    "bm25_lexical",
    "hybrid_rag",
    "rerank_rag",
    "graph_rag",
    "vectorless_markdown",
    "agentic_rag",
    "multihop_rag",
)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def import_store_builders() -> dict[str, Any]:
    if str(NOTEBOOKS_DIR) not in sys.path:
        sys.path.insert(0, str(NOTEBOOKS_DIR))
    from shared_rag_store import (  # noqa: PLC0415
        build_faiss_index,
        embed_texts,
        get_device,
        load_embedding_model,
        project_embeddings,
    )

    return {
        "build_faiss_index": build_faiss_index,
        "embed_texts": embed_texts,
        "get_device": get_device,
        "load_embedding_model": load_embedding_model,
        "project_embeddings": project_embeddings,
    }


def build_faiss_index(embeddings: np.ndarray):
    import faiss

    dimension = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def project_embeddings_fallback(embeddings: np.ndarray) -> np.ndarray:
    if len(embeddings) == 0:
        return np.empty((0, 2), dtype="float32")
    if len(embeddings) == 1:
        return np.array([[0.0, 0.0]], dtype="float32")
    return np.zeros((len(embeddings), 2), dtype="float32")


def load_hotpot_artifacts(artifacts_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    chunks_payload = load_json(artifacts_dir / "hotpot_chunks.json")
    queries_payload = load_json(artifacts_dir / "hotpot_queries.json")
    relevance_payload = load_json(artifacts_dir / "hotpot_relevance.json")
    chunks = chunks_payload.get("chunks", [])
    if not chunks:
        raise RuntimeError(
            f"No chunks found in {artifacts_dir / 'hotpot_chunks.json'}. Run Stage 1 first."
        )
    return chunks, queries_payload, relevance_payload


def ensure_chunk_compatibility(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compatible = []
    for index, chunk in enumerate(chunks):
        chunk_text = str(chunk.get("chunk_text", "")).strip()
        if not chunk_text:
            continue
        compatible.append(
            {
                **chunk,
                "chunk_id": str(chunk.get("chunk_id") or f"hotpot::chunk::{index}"),
                "page_number": int(chunk.get("page_number") or 1),
                "chunk_index": index,
                "chunk_text": chunk_text,
                "word_count": int(chunk.get("word_count") or len(chunk_text.split())),
                "preview": str(chunk.get("preview") or chunk_text[:220]),
            }
        )
    if not compatible:
        raise RuntimeError("No non-empty HotpotQA chunks are available.")
    return compatible


def make_projection_payload(chunks: list[dict[str, Any]], projection: np.ndarray) -> dict[str, Any]:
    return {
        "points": [
            {
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk["page_number"],
                "x": float(projection[index, 0]),
                "y": float(projection[index, 1]),
                "preview": chunk["preview"],
                "is_retrieved": False,
                "is_cited": False,
                "similarity_score": 0.0,
                "source": "hotpotqa",
                "title": chunk.get("title"),
                "sentence_index": chunk.get("sentence_index"),
                "example_id": chunk.get("example_id"),
            }
            for index, chunk in enumerate(chunks)
        ],
        "query_point": None,
    }


def write_frontend_placeholders(chunks: list[dict[str, Any]], projection_payload: dict[str, Any]) -> None:
    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
    empty_query = {"query": "", "results": []}
    chunks_payload = {"mode": "shared", "chunk_count": len(chunks), "chunks": chunks}
    for prefix in MODE_PREFIXES:
        write_json(FRONTEND_DATA_DIR / f"{prefix}_chunks.json", chunks_payload)
        write_json(FRONTEND_DATA_DIR / f"{prefix}_query_result.json", empty_query)
        write_json(
            FRONTEND_DATA_DIR / f"{prefix}_vis.json",
            projection_payload if prefix == "naive_rag" else {"query": "", "results": []},
        )


def build_hotpot_store(
    artifacts_dir: Path = ARTIFACTS_DIR,
    batch_size: int = 64,
    skip_embeddings: bool = False,
) -> dict[str, Any]:
    import faiss

    raw_chunks, queries_payload, relevance_payload = load_hotpot_artifacts(artifacts_dir)
    chunks = ensure_chunk_compatibility(raw_chunks)

    STORE_DIR.mkdir(parents=True, exist_ok=True)
    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if skip_embeddings:
        embeddings = np.zeros((len(chunks), 384), dtype="float32")
        model_name = "skipped"
        device_label = "skipped"
        build_index = build_faiss_index
        project = project_embeddings_fallback
    else:
        builders = import_store_builders()
        device = builders["get_device"]()
        model, model_name = builders["load_embedding_model"](device)
        device_label = str(device)
        embeddings = builders["embed_texts"](
            model,
            [chunk["chunk_text"] for chunk in chunks],
            batch_size=batch_size,
        )
        build_index = builders["build_faiss_index"]
        project = builders["project_embeddings"]

    index = build_index(embeddings)
    projection = project(embeddings)
    projection_payload = make_projection_payload(chunks, projection)

    np.save(STORE_DIR / "embeddings.npy", embeddings)
    faiss.write_index(index, str(STORE_DIR / "faiss.index"))
    write_json(STORE_DIR / "chunks.json", {"chunks": chunks})
    write_json(STORE_DIR / "projection.json", projection_payload)

    chunk_metadata = load_json(artifacts_dir / "hotpot_chunks.json").get("metadata", {})
    metadata = {
        **chunk_metadata,
        "source": "hotpotqa",
        "source_artifacts_dir": str(artifacts_dir),
        "query_count": len(queries_payload.get("queries", [])),
        "relevance_query_count": len(relevance_payload.get("queries", {})),
        "chunk_count": len(chunks),
        "embedding_model": model_name,
        "embedding_device": device_label,
        "index_type": "faiss.IndexFlatIP",
        "store_builder": "benchmarking/build_hotpot_store.py",
        "pid": os.getpid(),
    }
    write_json(STORE_DIR / "metadata.json", metadata)
    write_frontend_placeholders(chunks, projection_payload)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the shared store from HotpotQA artifacts.")
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Write a structurally valid store with zero embeddings for plumbing tests only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = build_hotpot_store(
        artifacts_dir=args.artifacts_dir,
        batch_size=args.batch_size,
        skip_embeddings=args.skip_embeddings,
    )
    print(f"Wrote HotpotQA store: {STORE_DIR}")
    print(f"Chunks: {metadata['chunk_count']}")
    print(f"Queries: {metadata['query_count']}")
    print(f"Embedding model: {metadata['embedding_model']}")


if __name__ == "__main__":
    main()
