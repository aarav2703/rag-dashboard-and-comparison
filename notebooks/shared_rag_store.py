from __future__ import annotations

import json
import math
import re
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from graph_cache import build_graph_artifacts


ROOT = Path(__file__).resolve().parents[1]
STORE_DIR = ROOT / "backend_or_exports" / "current_corpus"
FRONTEND_DATA_DIR = ROOT / "frontend" / "public" / "data"
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
MODEL_CANDIDATES = [
    ("sentence-transformers/all-MiniLM-L6-v2", False),
]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_embedding_model(device: torch.device | None = None):
    from sentence_transformers import SentenceTransformer

    device = device or get_device()
    errors = []
    for model_name, trust_remote_code in MODEL_CANDIDATES:
        try:
            model = SentenceTransformer(
                model_name,
                device=str(device),
                trust_remote_code=trust_remote_code,
                local_files_only=True,
            )
            return model, model_name
        except Exception as exc:
            errors.append(f"{model_name} local: {type(exc).__name__}: {exc}")
            try:
                model = SentenceTransformer(
                    model_name,
                    device=str(device),
                    trust_remote_code=trust_remote_code,
                )
                return model, model_name
            except Exception as online_exc:
                errors.append(f"{model_name} online: {type(online_exc).__name__}: {online_exc}")

    raise RuntimeError("No embedding model could be loaded: " + " | ".join(errors))


def extract_pages(pdf_path: Path) -> list[dict]:
    import fitz

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


def embed_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 64,
) -> np.ndarray:
    if not texts:
        return np.empty((0, model.get_sentence_embedding_dimension()), dtype="float32")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.astype("float32")


def project_embeddings(embeddings: np.ndarray) -> np.ndarray:
    if len(embeddings) == 0:
        return np.empty((0, 2), dtype="float32")
    if len(embeddings) == 1:
        return np.array([[0.0, 0.0]], dtype="float32")

    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=42)
        return reducer.fit_transform(embeddings).astype("float32")
    except Exception:
        from sklearn.decomposition import PCA

        return PCA(n_components=2).fit_transform(embeddings).astype("float32")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def build_faiss_index(embeddings: np.ndarray):
    import faiss

    dimension = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def build_store(pdf_path: str | Path) -> dict:
    import faiss

    pdf_path = Path(pdf_path)
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)

    pages = extract_pages(pdf_path)
    chunks = build_chunks(pages)
    device = get_device()
    model, model_name = load_embedding_model(device)
    embeddings = embed_texts(model, [chunk["chunk_text"] for chunk in chunks])
    index = build_faiss_index(embeddings)
    projection = project_embeddings(embeddings)

    np.save(STORE_DIR / "embeddings.npy", embeddings)
    faiss.write_index(index, str(STORE_DIR / "faiss.index"))
    write_json(STORE_DIR / "chunks.json", {"chunks": chunks})
    write_json(
        STORE_DIR / "metadata.json",
        {
            "source_pdf": str(pdf_path),
            "page_count": len(pages),
            "chunk_count": len(chunks),
            "embedding_model": model_name,
            "embedding_device": str(device),
            "index_type": "faiss.IndexFlatIP",
        },
    )
    write_json(
        STORE_DIR / "projection.json",
        {
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
                }
                for index, chunk in enumerate(chunks)
            ],
            "query_point": None,
        },
    )
    graph_metadata = build_graph_artifacts(STORE_DIR, chunks, source="pdf")

    empty_query = {"query": "", "results": []}
    chunks_payload = {"mode": "shared", "chunk_count": len(chunks), "chunks": chunks}
    projection_payload = json.loads((STORE_DIR / "projection.json").read_text(encoding="utf-8"))
    for prefix in ("naive_rag", "hybrid_rag", "graph_rag", "agentic_rag", "crag_rag"):
        write_json(FRONTEND_DATA_DIR / f"{prefix}_chunks.json", chunks_payload)
        write_json(FRONTEND_DATA_DIR / f"{prefix}_query_result.json", empty_query)
        write_json(FRONTEND_DATA_DIR / f"{prefix}_vis.json", projection_payload if prefix == "naive_rag" else {"query": "", "results": []})

    return {
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "embedding_model": model_name,
        "embedding_device": str(device),
        "graph_node_count": graph_metadata["node_count"],
        "graph_edge_count": graph_metadata["edge_count"],
    }


def load_store() -> dict:
    import faiss

    chunks_path = STORE_DIR / "chunks.json"
    embeddings_path = STORE_DIR / "embeddings.npy"
    index_path = STORE_DIR / "faiss.index"
    metadata_path = STORE_DIR / "metadata.json"

    if not chunks_path.exists() or not embeddings_path.exists() or not index_path.exists():
        raise RuntimeError("No FAISS corpus is available. Upload a PDF and run a pipeline first.")

    with open(chunks_path, "r", encoding="utf-8") as file:
        chunks = json.load(file).get("chunks", [])
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return {
        "chunks": chunks,
        "embeddings": np.load(embeddings_path),
        "index": faiss.read_index(str(index_path)),
        "metadata": metadata,
    }


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

    doc_count = max(1, len(chunks))
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
    score = 0.0
    contributions = []
    matched_terms = []

    for term in query_terms:
        term_tf = tf.get(term, 0)
        if term_tf <= 0:
            continue
        matched_terms.append(term)
        df = stats["document_frequency"].get(term, 0)
        idf = stats["idf"].get(term, math.log(1.0 + ((stats["doc_count"] + 0.5) / 0.5)))
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
    max_idf = max(stats["idf"].values(), default=1.0) or 1.0
    rows = []
    for term in query_terms:
        df = int(stats["document_frequency"].get(term, 0))
        idf = float(stats["idf"].get(term, math.log(1.0 + ((stats["doc_count"] + 0.5) / 0.5))))
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
