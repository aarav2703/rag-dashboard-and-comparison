import os
import json
import subprocess
import tempfile
import threading
import urllib.error
import urllib.request
import re
from pathlib import Path
from queue import Queue
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
if str(NOTEBOOKS_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(NOTEBOOKS_DIR))

from shared_rag_store import (
    STORE_DIR,
    FRONTEND_DATA_DIR,
    bm25_score,
    build_bm25_stats,
    build_query_term_stats,
    find_highlight_spans,
    load_embedding_model,
    load_store,
    tokenize,
    unique_preserve_order,
    write_json,
)

app = Flask(__name__)
CORS(app)


def load_env_file(env_path):
    """Load simple KEY=VALUE pairs from a local .env file."""
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


load_env_file(os.path.join(os.path.dirname(__file__), ".env"))

# Shared event queue and state
event_queue = Queue()
pipeline_status = {"state": "green" if (STORE_DIR / "chunks.json").exists() and (STORE_DIR / "faiss.index").exists() else "red"}  # red, yellow, green
uploaded_file_path = None
current_pipeline_mode = "naive"
logs = []

PIPELINE_CONFIG = {
    "naive": {
        "script": os.path.join(os.path.dirname(__file__), "notebooks", "00_build_faiss_corpus.py"),
        "prefix": "naive_rag",
    },
    "bm25": {
        "script": os.path.join(os.path.dirname(__file__), "notebooks", "00_build_faiss_corpus.py"),
        "prefix": "bm25_lexical",
    },
    "hybrid": {
        "script": os.path.join(os.path.dirname(__file__), "notebooks", "00_build_faiss_corpus.py"),
        "prefix": "hybrid_rag",
    },
    "rerank": {
        "script": os.path.join(os.path.dirname(__file__), "notebooks", "00_build_faiss_corpus.py"),
        "prefix": "rerank_rag",
    },
    "graph": {
        "script": os.path.join(os.path.dirname(__file__), "notebooks", "00_build_faiss_corpus.py"),
        "prefix": "graph_rag",
    },
    "vectorless": {
        "script": os.path.join(os.path.dirname(__file__), "notebooks", "00_build_faiss_corpus.py"),
        "prefix": "vectorless_markdown",
    },
    "agentic": {
        "script": os.path.join(os.path.dirname(__file__), "notebooks", "00_build_faiss_corpus.py"),
        "prefix": "agentic_rag",
    },
    "multihop": {
        "script": os.path.join(os.path.dirname(__file__), "notebooks", "00_build_faiss_corpus.py"),
        "prefix": "multihop_rag",
    },
}

embedding_model_cache = {"model": None, "name": None}
INSUFFICIENT_EVIDENCE_ANSWER = (
    "I don't have enough information in the retrieved evidence to answer that reliably."
)
TRUE_VALUES = {"1", "true", "yes", "on"}


def log_event(level, message, tag="system"):
    """Log an event to both queue and logs list"""
    event = {
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "message": message,
        "tag": tag,
    }
    event_queue.put(event)
    logs.append(event)
    print(f"[{tag}] {level.upper()}: {message}")


def result_text(result):
    return (
        result.get("full_chunk_text")
        or result.get("chunk_text_preview")
        or result.get("chunk_text")
        or ""
    )


def format_evidence_blocks(retrieved_results, max_chunks=5):
    context_blocks = []
    for index, result in enumerate(retrieved_results[:max_chunks], start=1):
        chunk_text = result_text(result)
        page_number = result.get("page_number", "?")
        score = (
            result.get("similarity_score")
            or result.get("hybrid_score")
            or result.get("bm25_score")
            or result.get("agent_score")
            or result.get("multihop_score")
            or result.get("graph_score")
            or 0.0
        )
        context_blocks.append(
            f"[{index}] Page {page_number} | score={float(score):.4f}\n{chunk_text}"
        )
    return context_blocks


def deepseek_generate_answer(query_text, retrieved_results):
    """Generate a grounded answer with DeepSeek using retrieved chunks."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")

    context_blocks = format_evidence_blocks(retrieved_results)

    prompt = (
        "You are answering a question using only the provided retrieved passages from a document. "
        "Write a concise, grounded answer. If the evidence is weak or incomplete, say so clearly. "
        "Cite the passage numbers you used in brackets like [1] or [2].\n\n"
        f"Question: {query_text}\n\n"
        "Retrieved passages:\n" + "\n\n".join(context_blocks)
    )

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": "You are a careful retrieval-augmented answer writer.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 400,
    }

    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        response_data = json.loads(response.read().decode("utf-8"))

    choices = response_data.get("choices", [])
    if not choices:
        raise RuntimeError("DeepSeek returned no answer choices")

    message = choices[0].get("message", {})
    return message.get("content", "").strip()


def deepseek_chat(messages, max_tokens=260, temperature=0.0, timeout=35):
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")

    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response_data = json.loads(response.read().decode("utf-8"))
    choices = response_data.get("choices", [])
    if not choices:
        raise RuntimeError("DeepSeek returned no critic choices")
    return choices[0].get("message", {}).get("content", "").strip()


def deepseek_critic_enabled():
    return os.environ.get("DEEPSEEK_CRITIC_ENABLED", "true").strip().lower() in TRUE_VALUES


def parse_critic_response(content, query_text):
    verdict = "rejected"
    reason = ""
    retry_query = query_text
    confidence = 0.75
    supported_citations = []
    unsupported_claims = []

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()
        if key == "verdict":
            normalized = value.lower()
            if "accept" in normalized:
                verdict = "accepted"
            elif "insufficient" in normalized:
                verdict = "insufficient_evidence"
            else:
                verdict = "rejected"
        elif key in {"reason", "issues"}:
            reason = value
        elif key == "retry_query" and value and value.lower() not in {"none", "n/a", "null"}:
            retry_query = value
        elif key == "confidence":
            try:
                raw_confidence = float(value.strip("%"))
                confidence = raw_confidence / 100 if raw_confidence > 1 else raw_confidence
            except ValueError:
                pass
        elif key == "supported_citations":
            supported_citations = re.findall(r"\d+", value)
        elif key == "unsupported_claims" and value.lower() not in {"none", "n/a"}:
            unsupported_claims = [value]

    return {
        "verdict": verdict,
        "grounded": verdict == "accepted",
        "confidence": round(max(0.0, min(1.0, confidence)), 3),
        "issues": [] if verdict == "accepted" else [reason or "The critic did not find enough grounded support."],
        "supported_citations": supported_citations,
        "unsupported_claims": unsupported_claims,
        "suggested_query": retry_query,
        "critic_source": "deepseek",
        "raw_critic_response": content,
    }


def fallback_answer_from_results(query_text, retrieved_results):
    """Create a lightweight answer directly from the retrieved text."""
    if not retrieved_results:
        return "No retrieved evidence is available yet."

    top_chunks = []
    for result in retrieved_results[:3]:
        top_chunks.append(
            result.get("full_chunk_text")
            or result.get("chunk_text_preview")
            or result.get("chunk_text")
            or ""
        )

    combined = " ".join(top_chunks).strip()
    if not combined:
        return (
            "The retrieved chunks do not contain enough text to synthesize an answer."
        )

    return (
        "DeepSeek is unavailable, so here are the strongest retrieved passages to inspect: "
        + " ".join(chunk[:240] for chunk in top_chunks[:2])
    )


def heuristic_critic(query_text, answer, retrieved_results):
    evidence_text = " ".join(result_text(result) for result in retrieved_results[:5]).lower()
    answer_terms = set(tokenize(answer))
    weak_terms = {
        "the", "and", "for", "that", "with", "from", "this", "there", "retrieved",
        "passages", "inspect", "deepseek", "unavailable", "answer", "evidence",
    }
    content_terms = {term for term in answer_terms if len(term) > 3 and term not in weak_terms}
    supported_terms = {term for term in content_terms if term in evidence_text}
    coverage = len(supported_terms) / max(1, len(content_terms))
    has_citations = bool(re.search(r"\[\d+\]", answer))
    insufficient = "not enough information" in answer.lower() or "insufficient" in answer.lower()
    grounded = insufficient or (coverage >= 0.35 and (has_citations or len(supported_terms) >= 3))
    return {
        "verdict": "accepted" if grounded else "rejected",
        "grounded": grounded,
        "confidence": round(min(0.95, max(0.1, coverage)), 3),
        "issues": [] if grounded else ["Answer contains claims that are not clearly supported by retrieved evidence."],
        "supported_citations": [],
        "unsupported_claims": [] if grounded else [answer[:220]],
        "suggested_query": query_text,
        "critic_source": "heuristic",
    }


def critique_answer(query_text, answer, retrieved_results):
    context_blocks = format_evidence_blocks(retrieved_results, max_chunks=3)
    if not context_blocks:
        return {
            "verdict": "insufficient_evidence",
            "grounded": False,
            "confidence": 0.0,
            "issues": ["No retrieved evidence was available."],
            "supported_citations": [],
            "unsupported_claims": [],
            "suggested_query": query_text,
            "critic_source": "empty-evidence",
        }

    if not deepseek_critic_enabled():
        return heuristic_critic(query_text, answer, retrieved_results)

    prompt = (
        "Evaluate whether the answer is grounded only in the retrieved passages. "
        "Use the same standard as a strict RAG evaluator: every factual claim must be supported. "
        "Reply in exactly this line-based format, not JSON:\n"
        "VERDICT: ACCEPTED or REJECTED or INSUFFICIENT\n"
        "CONFIDENCE: number from 0 to 1\n"
        "REASON: one short sentence\n"
        "SUPPORTED_CITATIONS: citation numbers, or none\n"
        "UNSUPPORTED_CLAIMS: one short phrase, or none\n"
        "RETRY_QUERY: improved search query, or none\n\n"
        f"Question: {query_text}\n\n"
        f"Answer: {answer}\n\n"
        "Retrieved passages:\n" + "\n\n".join(context_blocks)
    )
    try:
        content = deepseek_chat(
            [
                {
                    "role": "system",
                    "content": "You are a strict RAG grounding critic. Do not give credit for facts absent from evidence.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=260,
            temperature=0.0,
            timeout=35,
        )
        return parse_critic_response(content, query_text)
    except Exception as exc:
        critic = heuristic_critic(query_text, answer, retrieved_results)
        critic["critic_source"] = "heuristic_after_deepseek_error"
        critic["critic_error"] = str(exc)
        return critic


def generate_answer_with_fallback(query_text, retrieved_results):
    try:
        answer = deepseek_generate_answer(query_text, retrieved_results)
        return answer, "deepseek"
    except (
        urllib.error.URLError,
        RuntimeError,
        TimeoutError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        log_event(
            "warn",
            f"DeepSeek unavailable, using fallback answer: {exc}",
            "query",
        )
        return fallback_answer_from_results(query_text, retrieved_results), "fallback"


def reformulate_query(query_text, critic):
    suggested = (critic.get("suggested_query") or "").strip()
    if suggested and suggested.lower() != query_text.lower():
        return suggested
    issues = " ".join(critic.get("issues") or [])[:180]
    return f"{query_text} supporting evidence {issues}".strip()


def self_heal_answer(mode, query_text, result, vis, max_retries=1):
    attempts = []
    current_result = result
    current_vis = vis
    current_query = query_text
    final_source = "fallback"
    retry_used = False

    for attempt_index in range(max_retries + 1):
        answer, answer_source = generate_answer_with_fallback(
            current_query,
            current_result.get("results", []),
        )
        critic = critique_answer(query_text, answer, current_result.get("results", []))
        attempts.append(
            {
                "attempt": attempt_index + 1,
                "query": current_query,
                "answer": answer,
                "answer_source": answer_source,
                "critic": critic,
                "evidence_count": len(current_result.get("results", [])),
            }
        )

        if critic.get("verdict") == "accepted":
            final_source = answer_source
            return current_result, current_vis, {
                "answer": answer,
                "answer_source": answer_source,
                "answer_model": "deepseek-chat",
                "evidence_count": len(current_result.get("results", [])),
                "critic": {**critic, "retry_used": retry_used, "retry_query": current_query if retry_used else None},
                "answer_attempts": attempts,
            }

        if attempt_index >= max_retries:
            break

        retry_query = reformulate_query(query_text, critic)
        retry_used = True
        log_event("warn", f"Critic rejected answer; retrying retrieval with: {retry_query}", "critic")
        current_query = retry_query
        current_result, current_vis = run_dynamic_retrieval(mode, retry_query)

    final_critic = attempts[-1]["critic"] if attempts else {}
    final_critic = {
        **final_critic,
        "verdict": "insufficient_evidence",
        "grounded": False,
        "retry_used": retry_used,
        "retry_query": current_query if retry_used else None,
    }
    return current_result, current_vis, {
        "answer": INSUFFICIENT_EVIDENCE_ANSWER,
        "answer_source": "insufficient_evidence",
        "answer_model": "deepseek-chat" if final_source == "deepseek" else "none",
        "evidence_count": len(current_result.get("results", [])),
        "critic": final_critic,
        "answer_attempts": attempts,
    }


def get_embedding_model():
    if embedding_model_cache["model"] is None:
        model, name = load_embedding_model()
        embedding_model_cache["model"] = model
        embedding_model_cache["name"] = name
    return embedding_model_cache["model"], embedding_model_cache["name"]


def normalize_scores(rows, key, target_key):
    values = [row.get(key, 0.0) for row in rows]
    min_value = min(values, default=0.0)
    max_value = max(values, default=0.0)
    span = max_value - min_value
    for row in rows:
        row[target_key] = 0.0 if span == 0 else (row.get(key, 0.0) - min_value) / span


def reciprocal_rank(rank, k=60):
    return 0.0 if rank is None else 1.0 / (k + rank)


def source_label(vector_rank, bm25_rank):
    if vector_rank and bm25_rank:
        return "both"
    if vector_rank:
        return "vector-only"
    return "bm25-only"


def run_vector_retrieval(query_text, top_k=5, candidate_k=10):
    store = load_store()
    chunks = store["chunks"]
    model, model_name = get_embedding_model()
    query_embedding = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    search_k = min(max(candidate_k, top_k), len(chunks))
    scores, indices = store["index"].search(query_embedding, search_k)
    candidate_rows = []
    for rank, idx in enumerate(indices[0], start=1):
        if idx < 0:
            continue
        chunk = chunks[int(idx)]
        candidate_rows.append(
            {
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk["page_number"],
                "similarity_score": float(scores[0][rank - 1]),
                "chunk_text_preview": chunk.get("preview", chunk["chunk_text"][:220]),
                "full_chunk_text": chunk["chunk_text"],
            }
        )

    projection_path = STORE_DIR / "projection.json"
    if projection_path.exists():
        with open(projection_path, "r", encoding="utf-8") as file:
            vis = json.load(file)
    else:
        vis = {"points": [], "query_point": None}

    retrieved_ids = {row["chunk_id"] for row in candidate_rows[:top_k]}
    score_by_id = {row["chunk_id"]: row["similarity_score"] for row in candidate_rows}
    vis["points"] = [
        {
            **point,
            "is_retrieved": point["chunk_id"] in retrieved_ids,
            "similarity_score": float(score_by_id.get(point["chunk_id"], 0.0)),
        }
        for point in vis.get("points", [])
    ]
    vis["query_point"] = {"x": 0.0, "y": 0.0, "is_query": True, "query": query_text}

    result = {
        "mode": "naive",
        "pipeline": "naive_rag",
        "query": query_text,
        "embedding_model": model_name,
        "results": candidate_rows[:top_k],
        "query_embedding": query_embedding[0].tolist(),
    }
    return result, vis, candidate_rows


def run_bm25_retrieval(query_text, top_k=5, candidate_k=10):
    store = load_store()
    chunks = store["chunks"]
    stats = build_bm25_stats(chunks)
    query_terms = unique_preserve_order(tokenize(query_text))
    missing_query_terms = [
        term for term in query_terms if stats["document_frequency"].get(term, 0) == 0
    ]

    scored = []
    for index, chunk in enumerate(chunks):
        score, contributions, matched_terms = bm25_score(
            stats["doc_tokens"][index], query_terms, stats
        )
        scored.append(
            {
                **chunk,
                "bm25_score": round(score, 6),
                "matched_terms": matched_terms,
                "term_contributions": contributions,
                "highlight_spans": find_highlight_spans(chunk["chunk_text"], matched_terms),
            }
        )

    scored.sort(key=lambda item: (-item["bm25_score"], item["page_number"], item["chunk_index"]))
    candidate_rows = []
    for rank, item in enumerate(scored[:candidate_k], start=1):
        candidate_rows.append(
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

    term_stats = build_query_term_stats(query_terms, stats)
    top_contributions = candidate_rows[0]["term_contributions"] if candidate_rows else []
    result = {
        "mode": "bm25",
        "pipeline": "bm25_lexical_rag",
        "query": query_text,
        "query_terms": query_terms,
        "missing_query_terms": missing_query_terms,
        "missing_query_warning": bool(missing_query_terms),
        "term_stats": term_stats,
        "top_result_term_contributions": top_contributions,
        "results": candidate_rows[:top_k],
    }
    vis = {
        "mode": "bm25",
        "pipeline": "bm25_lexical_rag",
        "query": query_text,
        "query_terms": query_terms,
        "missing_query_terms": missing_query_terms,
        "term_stats": term_stats,
        "top_result_term_contributions": top_contributions,
        "warning": (
            "Missing query terms: " + ", ".join(missing_query_terms)
            if missing_query_terms
            else "All query terms were observed in the corpus."
        ),
    }
    return result, vis, candidate_rows


def run_hybrid_retrieval(query_text, top_k=5, candidate_k=10):
    vector_result, _, vector_candidates = run_vector_retrieval(
        query_text, top_k=top_k, candidate_k=candidate_k
    )
    bm25_result, _, bm25_candidates = run_bm25_retrieval(
        query_text, top_k=top_k, candidate_k=candidate_k
    )
    store = load_store()
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in store["chunks"]}

    vector_by_id = {row["chunk_id"]: row for row in vector_candidates}
    bm25_by_id = {row["chunk_id"]: row for row in bm25_candidates}
    candidate_ids = set(vector_by_id) | set(bm25_by_id)

    rows = []
    for chunk_id in candidate_ids:
        chunk = chunks_by_id[chunk_id]
        vector_row = vector_by_id.get(chunk_id)
        bm25_row = bm25_by_id.get(chunk_id)
        rows.append(
            {
                "chunk_id": chunk_id,
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "preview": chunk["preview"],
                "full_chunk_text": chunk["chunk_text"],
                "vector_rank": vector_row.get("rank") if vector_row else None,
                "bm25_rank": bm25_row.get("rank") if bm25_row else None,
                "vector_score": vector_row.get("similarity_score", 0.0) if vector_row else 0.0,
                "bm25_score": bm25_row.get("bm25_score", 0.0) if bm25_row else 0.0,
                "matched_terms": bm25_row.get("matched_terms", []) if bm25_row else [],
            }
        )

    normalize_scores(rows, "vector_score", "vector_norm")
    normalize_scores(rows, "bm25_score", "bm25_norm")
    for row in rows:
        row["source"] = source_label(row["vector_rank"], row["bm25_rank"])
        row["fusion_score"] = round(
            0.5 * row["vector_norm"]
            + 0.5 * row["bm25_norm"]
            + reciprocal_rank(row["vector_rank"])
            + reciprocal_rank(row["bm25_rank"]),
            6,
        )

    rows.sort(key=lambda item: (-item["fusion_score"], item["page_number"], item["chunk_index"]))
    results = [
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
            "full_chunk_text": row["full_chunk_text"],
        }
        for rank, row in enumerate(rows[:top_k], start=1)
    ]

    source_counts = {}
    final_counts = {}
    for row in rows:
        source_counts[row["source"]] = source_counts.get(row["source"], 0) + 1
    for row in results:
        final_counts[row["source"]] = final_counts.get(row["source"], 0) + 1

    overlap = {
        "vector_candidates": len(vector_candidates),
        "bm25_candidates": len(bm25_candidates),
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
        for index, row in enumerate(rows[:candidate_k])
    ]

    result = {
        "mode": "hybrid",
        "pipeline": "hybrid_rag",
        "query": query_text,
        "query_terms": bm25_result.get("query_terms", []),
        "results": results,
        "source_counts": source_counts,
        "overlap": overlap,
        "rank_fusion_table": rank_fusion_table,
        "component_results": {
            "vector": vector_result["results"],
            "bm25": bm25_result["results"],
        },
    }
    vis = {
        "mode": "hybrid",
        "pipeline": "hybrid_rag",
        "query": query_text,
        "merge_sankey": {
            "nodes": [
                {"name": "Vector candidates"},
                {"name": "BM25 candidates"},
                {"name": "Merged candidates"},
                {"name": "Final evidence"},
            ],
            "links": [
                {"source": 0, "target": 2, "value": max(1, len(vector_candidates))},
                {"source": 1, "target": 2, "value": max(1, len(bm25_candidates))},
                {"source": 2, "target": 3, "value": max(1, len(results))},
            ],
        },
        "overlap": overlap,
        "overlap_matrix": [
            {"source": "vector-only", "count": overlap["vector_only"]},
            {"source": "bm25-only", "count": overlap["bm25_only"]},
            {"source": "both", "count": overlap["both"]},
        ],
        "final_source_counts": final_counts,
        "rank_fusion_table": rank_fusion_table,
    }
    return result, vis, rows


def run_rerank_retrieval(query_text, top_k=5, candidate_k=20):
    vector_result, _, vector_candidates = run_vector_retrieval(
        query_text, top_k=top_k, candidate_k=candidate_k
    )
    store = load_store()
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in store["chunks"]}
    candidate_chunks = [chunks_by_id[row["chunk_id"]] for row in vector_candidates]
    stats = build_bm25_stats(candidate_chunks)
    query_terms = unique_preserve_order(tokenize(query_text))

    rows = []
    for index, candidate in enumerate(vector_candidates):
        chunk = chunks_by_id[candidate["chunk_id"]]
        bm25_raw, contributions, matched_terms = bm25_score(
            stats["doc_tokens"][index], query_terms, stats
        )
        coverage = len(matched_terms) / max(1, len(query_terms))
        rows.append(
            {
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "before_rank": candidate["rank"],
                "vector_score": candidate["similarity_score"],
                "bm25_candidate_score": float(bm25_raw),
                "term_coverage": coverage,
                "matched_terms": matched_terms,
                "term_contributions": contributions,
                "chunk_text_preview": chunk["preview"],
                "full_chunk_text": chunk["chunk_text"],
            }
        )

    normalize_scores(rows, "vector_score", "vector_norm")
    normalize_scores(rows, "bm25_candidate_score", "bm25_norm")
    for row in rows:
        row["reranker_score"] = round(
            0.55 * row["bm25_norm"] + 0.30 * row["term_coverage"] + 0.15 * row["vector_norm"],
            6,
        )

    rows.sort(key=lambda item: (-item["reranker_score"], item["before_rank"]))
    for after_rank, row in enumerate(rows, start=1):
        row["after_rank"] = after_rank
        row["movement"] = row["before_rank"] - after_rank
        if row["movement"] > 0:
            row["movement_label"] = "promoted"
        elif row["movement"] < 0:
            row["movement_label"] = "demoted"
        else:
            row["movement_label"] = "unchanged"

    results = [
        {
            "rank": row["after_rank"],
            "chunk_id": row["chunk_id"],
            "page_number": row["page_number"],
            "reranker_score": row["reranker_score"],
            "similarity_score": round(row["vector_score"], 6),
            "before_rank": row["before_rank"],
            "after_rank": row["after_rank"],
            "movement": row["movement"],
            "movement_label": row["movement_label"],
            "matched_terms": row["matched_terms"],
            "term_coverage": round(row["term_coverage"], 6),
            "chunk_text_preview": row["chunk_text_preview"],
            "full_chunk_text": row["full_chunk_text"],
        }
        for row in rows[:top_k]
    ]

    before_by_id = {row["chunk_id"]: row for row in vector_candidates[:top_k]}
    before_after_table = []
    for row in rows[:candidate_k]:
        before_after_table.append(
            {
                "chunk_id": row["chunk_id"],
                "page_number": row["page_number"],
                "before_rank": row["before_rank"],
                "after_rank": row["after_rank"],
                "movement": row["movement"],
                "movement_label": row["movement_label"],
                "vector_score": round(row["vector_score"], 6),
                "reranker_score": row["reranker_score"],
                "was_initial_top_k": row["chunk_id"] in before_by_id,
                "is_final_top_k": row["after_rank"] <= top_k,
                "preview": row["chunk_text_preview"],
            }
        )

    promoted = [row for row in before_after_table if row["movement"] > 0][:6]
    demoted = sorted(
        [row for row in before_after_table if row["movement"] < 0],
        key=lambda item: item["movement"],
    )[:6]
    score_values = [row["reranker_score"] for row in rows]
    bucket_count = 8
    histogram = []
    if score_values:
        min_score = min(score_values)
        max_score = max(score_values)
        span = max_score - min_score
        buckets = [0 for _ in range(bucket_count)]
        for score in score_values:
            bucket_index = 0 if span == 0 else min(bucket_count - 1, int(((score - min_score) / span) * bucket_count))
            buckets[bucket_index] += 1
        for index, count in enumerate(buckets):
            start = min_score if span == 0 else min_score + (span * index / bucket_count)
            end = max_score if span == 0 else min_score + (span * (index + 1) / bucket_count)
            histogram.append(
                {
                    "bucket": index + 1,
                    "start": round(start, 6),
                    "end": round(end, 6),
                    "count": count,
                }
            )

    result = {
        "mode": "rerank",
        "pipeline": "rerank_rag",
        "query": query_text,
        "query_terms": query_terms,
        "candidate_count": len(vector_candidates),
        "results": results,
        "before_after_table": before_after_table,
        "promoted_chunks": promoted,
        "demoted_chunks": demoted,
        "reranker_score_histogram": histogram,
        "reranker": "local lexical-semantic candidate reranker",
        "component_results": {
            "before_rerank": vector_candidates[:top_k],
        },
    }
    vis = {
        "mode": "rerank",
        "pipeline": "rerank_rag",
        "query": query_text,
        "slopegraph": before_after_table,
        "before_after_table": before_after_table,
        "promoted_chunks": promoted,
        "demoted_chunks": demoted,
        "reranker_score_histogram": histogram,
        "summary": {
            "candidate_count": len(vector_candidates),
            "promoted_count": len([row for row in before_after_table if row["movement"] > 0]),
            "demoted_count": len([row for row in before_after_table if row["movement"] < 0]),
            "unchanged_count": len([row for row in before_after_table if row["movement"] == 0]),
        },
    }
    return result, vis, rows


STOP_ENTITIES = {
    "The",
    "This",
    "That",
    "These",
    "Those",
    "Figure",
    "Table",
    "Chapter",
    "Section",
    "Example",
    "Remark",
    "Definition",
    "Proof",
    "Page",
    "PDF",
}


def canonical_entity(text):
    return re.sub(r"\s+", " ", text.strip()).strip(".,:;()[]{}")


def extract_entities(text, max_entities=12):
    candidates = []
    patterns = [
        r"\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,3}\b",
        r"\b(?:neural network|decision tree|support vector machine|gradient descent|linear regression|logistic regression|deep learning|machine learning|random forest|nearest neighbor|principal component analysis|bayesian network|markov chain|attention mechanism|transformer model)s?\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE if pattern.startswith("\\b(?:") else 0):
            entity = canonical_entity(match.group(0))
            if len(entity) < 3 or entity in STOP_ENTITIES:
                continue
            if entity.lower() in {"and", "for", "with", "from", "where", "which", "there"}:
                continue
            candidates.append(entity.title() if entity.islower() else entity)

    seen = set()
    unique = []
    for entity in candidates:
        key = entity.lower()
        if key not in seen:
            seen.add(key)
            unique.append(entity)
        if len(unique) >= max_entities:
            break
    return unique


def extract_claims(text, max_claims=3):
    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())
    claims = []
    for sentence in sentences:
        clean = sentence.strip()
        if len(clean) < 60 or len(clean) > 260:
            continue
        lower = clean.lower()
        if any(marker in lower for marker in (" is ", " are ", " means ", " represents ", " depends ", " uses ", " measures ", " computes ", " predicts ")):
            claims.append(clean)
        if len(claims) >= max_claims:
            break
    return claims


def graph_node_id(kind, value):
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value).lower()).strip("_")
    return f"{kind}:{safe}"


def build_document_graph(chunks, focus_chunk_ids=None):
    focus_chunk_ids = set(focus_chunk_ids or [])
    selected_chunks = chunks
    nodes = {}
    edges = {}
    entity_to_sections = {}
    section_entities = {}

    document_id = "document:pdf"
    nodes[document_id] = {"id": document_id, "label": "Document", "type": "document", "weight": len(selected_chunks)}

    for chunk in selected_chunks:
        section_id = graph_node_id("section", chunk["chunk_id"])
        section_label = f"Page {chunk['page_number']} / chunk {chunk.get('chunk_index', 0) + 1}"
        nodes[section_id] = {
            "id": section_id,
            "label": section_label,
            "type": "section",
            "page_number": chunk["page_number"],
            "chunk_id": chunk["chunk_id"],
            "preview": chunk.get("preview", ""),
            "weight": 1,
            "is_focus": chunk["chunk_id"] in focus_chunk_ids,
        }
        edges[(document_id, section_id, "located_in")] = {
            "source": document_id,
            "target": section_id,
            "type": "located_in",
            "weight": 1,
        }

        entities = extract_entities(chunk["chunk_text"])
        section_entities[section_id] = entities
        claim_ids = []
        for entity in entities:
            entity_id = graph_node_id("entity", entity)
            nodes.setdefault(
                entity_id,
                {"id": entity_id, "label": entity, "type": "entity", "weight": 0},
            )
            nodes[entity_id]["weight"] += 1
            entity_to_sections.setdefault(entity_id, set()).add(section_id)
            edges[(entity_id, section_id, "mentions")] = {
                "source": entity_id,
                "target": section_id,
                "type": "mentions",
                "weight": 1,
            }

        for claim_index, claim in enumerate(extract_claims(chunk["chunk_text"])):
            claim_id = graph_node_id("claim", f"{chunk['chunk_id']}:{claim_index}")
            nodes[claim_id] = {
                "id": claim_id,
                "label": claim[:96],
                "type": "claim",
                "page_number": chunk["page_number"],
                "chunk_id": chunk["chunk_id"],
                "preview": claim,
                "weight": 1,
                "is_focus": chunk["chunk_id"] in focus_chunk_ids,
            }
            claim_ids.append(claim_id)
            edges[(claim_id, section_id, "supports")] = {
                "source": claim_id,
                "target": section_id,
                "type": "supports",
                "weight": 1,
            }
            for entity in entities[:6]:
                entity_id = graph_node_id("entity", entity)
                edges[(entity_id, claim_id, "mentions")] = {
                    "source": entity_id,
                    "target": claim_id,
                    "type": "mentions",
                    "weight": 1,
                }

        for left_index, left in enumerate(entities[:8]):
            for right in entities[left_index + 1 : 8]:
                left_id = graph_node_id("entity", left)
                right_id = graph_node_id("entity", right)
                key = tuple(sorted([left_id, right_id])) + ("co_occurs",)
                if key not in edges:
                    edges[key] = {
                        "source": key[0],
                        "target": key[1],
                        "type": "co-occurs",
                        "weight": 0,
                    }
                edges[key]["weight"] += 1

    return {
        "nodes": list(nodes.values()),
        "edges": list(edges.values()),
        "entity_to_sections": entity_to_sections,
        "section_entities": section_entities,
    }


def run_graph_retrieval(query_text, top_k=5):
    store = load_store()
    chunks = store["chunks"]
    vector_result, _, vector_candidates = run_vector_retrieval(query_text, top_k=top_k, candidate_k=18)
    bm25_result, _, bm25_candidates = run_bm25_retrieval(query_text, top_k=top_k, candidate_k=18)
    seeded_ids = {row["chunk_id"] for row in vector_candidates + bm25_candidates}
    graph = build_document_graph(chunks, focus_chunk_ids=seeded_ids)
    nodes_by_id = {node["id"]: node for node in graph["nodes"]}
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    query_entities = extract_entities(query_text, max_entities=8)
    query_terms = set(tokenize(query_text))
    vector_rank_by_id = {row["chunk_id"]: row["rank"] for row in vector_candidates}
    bm25_rank_by_id = {row["chunk_id"]: row["rank"] for row in bm25_candidates}

    matched_entity_ids = []
    entity_nodes = [node for node in graph["nodes"] if node["type"] == "entity"]
    for entity in query_entities:
        entity_key = entity.lower()
        for node in entity_nodes:
            if entity_key in node["label"].lower() or node["label"].lower() in entity_key:
                matched_entity_ids.append(node["id"])

    if not matched_entity_ids:
        for node in entity_nodes:
            label_terms = set(tokenize(node["label"]))
            if label_terms & query_terms:
                matched_entity_ids.append(node["id"])

    matched_entity_ids = list(dict.fromkeys(matched_entity_ids))[:8]

    section_scores = {}
    path_rows = []
    for candidate in vector_candidates:
        section_id = graph_node_id("section", candidate["chunk_id"])
        section_scores[section_id] = section_scores.get(section_id, 0.0) + reciprocal_rank(candidate["rank"], k=20) * 12
    for candidate in bm25_candidates:
        section_id = graph_node_id("section", candidate["chunk_id"])
        section_scores[section_id] = section_scores.get(section_id, 0.0) + reciprocal_rank(candidate["rank"], k=20) * 10

    for entity_id in matched_entity_ids:
        sections = graph["entity_to_sections"].get(entity_id, set())
        for section_id in sections:
            section_scores[section_id] = section_scores.get(section_id, 0.0) + 2.5
            path_rows.append(
                {
                    "query": query_text,
                    "entity_id": entity_id,
                    "entity": nodes_by_id[entity_id]["label"],
                    "section_id": section_id,
                    "section": nodes_by_id[section_id]["label"],
                    "edge_type": "mentions",
                }
            )

        for edge in graph["edges"]:
            if edge["type"] != "co-occurs":
                continue
            neighbor_id = None
            if edge["source"] == entity_id:
                neighbor_id = edge["target"]
            elif edge["target"] == entity_id:
                neighbor_id = edge["source"]
            if not neighbor_id:
                continue
            for section_id in graph["entity_to_sections"].get(neighbor_id, set()):
                section_scores[section_id] = section_scores.get(section_id, 0.0) + 1.0
                path_rows.append(
                    {
                        "query": query_text,
                        "entity_id": entity_id,
                        "entity": nodes_by_id[entity_id]["label"],
                        "related_entity_id": neighbor_id,
                        "related_entity": nodes_by_id[neighbor_id]["label"],
                        "section_id": section_id,
                        "section": nodes_by_id[section_id]["label"],
                        "edge_type": "co-occurs",
                    }
                )

    if not section_scores:
        bm25_result, _, bm25_candidates = run_bm25_retrieval(query_text, top_k=top_k, candidate_k=top_k)
        for rank, candidate in enumerate(bm25_candidates, start=1):
            section_id = graph_node_id("section", candidate["chunk_id"])
            section_scores[section_id] = max(0.1, top_k - rank + 1)
        path_rows.append(
            {
                "query": query_text,
                "entity": "Lexical fallback",
                "section": "BM25-backed section match",
                "edge_type": "supports",
            }
        )

    ranked_sections = sorted(section_scores.items(), key=lambda item: (-item[1], nodes_by_id.get(item[0], {}).get("page_number", 9999)))[:top_k]
    used_section_ids = {section_id for section_id, _ in ranked_sections}
    used_entity_ids = set(matched_entity_ids)
    for path in path_rows:
        if path.get("section_id") in used_section_ids:
            if path.get("entity_id"):
                used_entity_ids.add(path["entity_id"])
            if path.get("related_entity_id"):
                used_entity_ids.add(path["related_entity_id"])

    used_node_ids = used_section_ids | used_entity_ids | {"document:pdf"}
    subgraph_edges = [
        edge for edge in graph["edges"]
        if edge["source"] in used_node_ids and edge["target"] in used_node_ids
    ]
    for edge in graph["edges"]:
        if edge["type"] == "supports" and edge["target"] in used_section_ids:
            subgraph_edges.append(edge)
            used_node_ids.add(edge["source"])
    used_node_ids |= {edge["source"] for edge in subgraph_edges} | {edge["target"] for edge in subgraph_edges}
    highlighted_subgraph = {
        "nodes": [node for node in graph["nodes"] if node["id"] in used_node_ids],
        "edges": subgraph_edges,
    }

    results = []
    for rank, (section_id, score) in enumerate(ranked_sections, start=1):
        section_node = nodes_by_id.get(section_id, {})
        chunk = chunks_by_id.get(section_node.get("chunk_id"), {})
        section_entity_labels = [
            nodes_by_id[graph_node_id("entity", entity)]["label"]
            for entity in graph["section_entities"].get(section_id, [])
            if graph_node_id("entity", entity) in nodes_by_id
        ][:8]
        results.append(
            {
                "rank": rank,
                "chunk_id": chunk.get("chunk_id"),
                "page_number": chunk.get("page_number"),
                "graph_score": round(score, 6),
                "vector_rank": vector_rank_by_id.get(chunk.get("chunk_id")),
                "bm25_rank": bm25_rank_by_id.get(chunk.get("chunk_id")),
                "section_id": section_id,
                "section_label": section_node.get("label", "Section"),
                "matched_entities": section_entity_labels,
                "chunk_text_preview": chunk.get("preview", ""),
                "full_chunk_text": chunk.get("chunk_text", ""),
            }
        )

    communities = []
    page_groups = {}
    for node in graph["nodes"]:
        if node["type"] == "section":
            page = node.get("page_number", 0)
            bucket = ((page - 1) // 20) + 1 if page else 0
            page_groups.setdefault(bucket, {"id": f"community:{bucket}", "label": f"Pages {(bucket - 1) * 20 + 1}-{bucket * 20}", "node_ids": []})
            page_groups[bucket]["node_ids"].append(node["id"])
    communities = list(page_groups.values())[:10]

    result = {
        "mode": "graph",
        "pipeline": "graph_rag_lite",
        "query": query_text,
        "query_entities": query_entities,
        "matched_entities": [nodes_by_id[entity_id]["label"] for entity_id in matched_entity_ids if entity_id in nodes_by_id],
        "results": results,
        "path_explanation": path_rows[:20],
        "graph_stats": {
            "node_count": len(graph["nodes"]),
            "edge_count": len(graph["edges"]),
            "used_node_count": len(highlighted_subgraph["nodes"]),
            "claim_count": len([node for node in graph["nodes"] if node["type"] == "claim"]),
            "seeded_chunk_count": len(seeded_ids),
        },
        "component_results": {
            "vector": vector_result["results"],
            "bm25": bm25_result["results"],
        },
    }
    vis = {
        "mode": "graph",
        "pipeline": "graph_rag_lite",
        "query": query_text,
        "graph": {
            "nodes": graph["nodes"][:220],
            "edges": graph["edges"][:420],
        },
        "highlighted_subgraph": highlighted_subgraph,
        "query_path": path_rows[:20],
        "communities": communities,
        "filters": {
            "node_types": ["document", "entity", "section", "claim"],
            "edge_types": ["mentions", "supports", "co-occurs", "located_in"],
        },
    }
    return result, vis, highlighted_subgraph


SECTION_HINTS = [
    "abstract",
    "introduction",
    "background",
    "method",
    "methods",
    "data",
    "model",
    "experiment",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "references",
]


def infer_chunk_heading(chunk):
    text = chunk.get("chunk_text", "")
    chapter_match = re.search(r"\bCHAPTER\s+(\d+)\.?\s+([A-Z][A-Z\s-]{3,60})", text)
    if chapter_match:
        return f"Chapter {chapter_match.group(1)}: {chapter_match.group(2).title().strip()}"

    lower = text[:700].lower()
    for hint in SECTION_HINTS:
        if re.search(rf"\b{re.escape(hint)}\b", lower):
            return hint.title()

    page = chunk.get("page_number", 0)
    return f"Page {page} discussion"


def page_band_label(page_number, band_size=20):
    band_start = ((max(1, page_number) - 1) // band_size) * band_size + 1
    return f"Pages {band_start}-{band_start + band_size - 1}"


def make_tree_node(node_id, label, node_type, **extra):
    return {
        "id": node_id,
        "label": label,
        "type": node_type,
        "children": [],
        "confidence": 0.0,
        "is_selected": False,
        **extra,
    }


def build_markdown_tree(chunks):
    root = make_tree_node("root", "Document", "document")
    band_nodes = {}
    page_nodes = {}
    leaf_nodes = []

    for chunk in chunks:
        page = chunk.get("page_number", 1)
        band_label = page_band_label(page)
        band_id = graph_node_id("section", band_label)
        if band_id not in band_nodes:
            band_nodes[band_id] = make_tree_node(
                band_id,
                band_label,
                "section",
                depth=1,
                page_start=((page - 1) // 20) * 20 + 1,
            )
            root["children"].append(band_nodes[band_id])

        page_id = graph_node_id("page", f"{page}")
        if page_id not in page_nodes:
            page_nodes[page_id] = make_tree_node(
                page_id,
                f"Page {page}",
                "section",
                depth=2,
                page_number=page,
            )
            band_nodes[band_id]["children"].append(page_nodes[page_id])

        heading = infer_chunk_heading(chunk)
        leaf_id = graph_node_id("paragraph", chunk["chunk_id"])
        leaf = make_tree_node(
            leaf_id,
            heading,
            "paragraph",
            depth=3,
            page_number=page,
            chunk_id=chunk["chunk_id"],
            preview=chunk.get("preview", ""),
            full_chunk_text=chunk.get("chunk_text", ""),
        )
        page_nodes[page_id]["children"].append(leaf)
        leaf_nodes.append(leaf)

    return root, leaf_nodes


def score_vectorless_sections(query_text, chunks, leaf_nodes):
    stats = build_bm25_stats(chunks)
    query_terms = unique_preserve_order(tokenize(query_text))
    scores = {}
    details = {}

    chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    leaf_by_chunk = {leaf["chunk_id"]: leaf for leaf in leaf_nodes}
    for index, chunk in enumerate(chunks):
        bm25_raw, contributions, matched_terms = bm25_score(
            stats["doc_tokens"][index], query_terms, stats
        )
        leaf = leaf_by_chunk.get(chunk["chunk_id"])
        if not leaf:
            continue
        heading_terms = set(tokenize(leaf["label"]))
        heading_overlap = len(heading_terms.intersection(query_terms))
        structural_bonus = 0.35 * heading_overlap
        score = float(bm25_raw) + structural_bonus
        scores[leaf["id"]] = score
        details[leaf["id"]] = {
            "chunk": chunk_by_id[chunk["chunk_id"]],
            "matched_terms": matched_terms,
            "term_contributions": contributions,
            "heading_overlap": heading_overlap,
        }

    max_score = max(scores.values(), default=0.0) or 1.0
    for leaf in leaf_nodes:
        leaf["confidence"] = round(scores.get(leaf["id"], 0.0) / max_score, 6)
    return scores, details


def propagate_tree_confidence(node, selected_ids):
    if not node.get("children"):
        node["is_selected"] = node["id"] in selected_ids
        return node.get("confidence", 0.0)

    child_scores = [propagate_tree_confidence(child, selected_ids) for child in node["children"]]
    node["confidence"] = round(max(child_scores, default=0.0), 6)
    node["is_selected"] = node["id"] in selected_ids or any(child.get("is_selected") for child in node["children"])
    return node["confidence"]


def find_tree_path(node, target_id, path=None):
    path = list(path or [])
    next_path = path + [{"id": node["id"], "label": node["label"], "type": node["type"], "confidence": node.get("confidence", 0.0)}]
    if node["id"] == target_id:
        return next_path
    for child in node.get("children", []):
        found = find_tree_path(child, target_id, next_path)
        if found:
            return found
    return None


def flatten_tree(node, rows=None):
    rows = rows or []
    rows.append({
        "id": node["id"],
        "label": node["label"],
        "type": node["type"],
        "confidence": node.get("confidence", 0.0),
        "page_number": node.get("page_number"),
        "chunk_id": node.get("chunk_id"),
        "is_selected": node.get("is_selected", False),
    })
    for child in node.get("children", []):
        flatten_tree(child, rows)
    return rows


def run_vectorless_retrieval(query_text, top_k=5):
    store = load_store()
    chunks = store["chunks"]
    tree, leaf_nodes = build_markdown_tree(chunks)
    scores, details = score_vectorless_sections(query_text, chunks, leaf_nodes)
    ranked_leaf_ids = sorted(scores, key=lambda node_id: scores[node_id], reverse=True)
    selected_leaf_ids = set(ranked_leaf_ids[:top_k])

    selected_path = find_tree_path(tree, ranked_leaf_ids[0], []) if ranked_leaf_ids else [{"id": "root", "label": "Document", "type": "document", "confidence": 0.0}]
    selected_path_ids = {item["id"] for item in selected_path}
    selected_ids = selected_leaf_ids | selected_path_ids
    propagate_tree_confidence(tree, selected_ids)
    selected_path = find_tree_path(tree, ranked_leaf_ids[0], []) if ranked_leaf_ids else selected_path

    results = []
    for rank, leaf_id in enumerate(ranked_leaf_ids[:top_k], start=1):
        detail = details[leaf_id]
        chunk = detail["chunk"]
        leaf = next((item for item in leaf_nodes if item["id"] == leaf_id), {})
        path = find_tree_path(tree, leaf_id, []) or []
        results.append(
            {
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk["page_number"],
                "section_id": leaf_id,
                "section_label": leaf.get("label", "Section"),
                "section_confidence": leaf.get("confidence", 0.0),
                "navigation_path": path,
                "matched_terms": detail["matched_terms"],
                "chunk_text_preview": chunk.get("preview", ""),
                "full_chunk_text": chunk.get("chunk_text", ""),
            }
        )

    heatmap = [
        row for row in flatten_tree(tree)
        if row["type"] in {"section", "paragraph"} and row["confidence"] > 0
    ]
    heatmap.sort(key=lambda row: row["confidence"], reverse=True)
    result = {
        "mode": "vectorless",
        "pipeline": "vectorless_markdown_rag",
        "query": query_text,
        "query_terms": unique_preserve_order(tokenize(query_text)),
        "results": results,
        "selected_path": selected_path,
        "section_confidence": heatmap[:40],
        "tree_stats": {
            "section_count": len([row for row in flatten_tree(tree) if row["type"] == "section"]),
            "paragraph_count": len(leaf_nodes),
            "selected_depth": len(selected_path),
        },
    }
    vis = {
        "mode": "vectorless",
        "pipeline": "vectorless_markdown_rag",
        "query": query_text,
        "tree": tree,
        "selected_path": selected_path,
        "selected_path_ids": list(selected_path_ids),
        "section_confidence": heatmap[:80],
        "section_heatmap": heatmap[:80],
    }
    return result, vis, tree


def run_agentic_retrieval(query_text, top_k=5):
    """Plan, call retrieval tools, critique evidence, optionally retry, then accept evidence."""
    vector_result, _, vector_candidates = run_vector_retrieval(query_text, top_k=top_k, candidate_k=14)
    bm25_result, _, bm25_candidates = run_bm25_retrieval(query_text, top_k=top_k, candidate_k=14)
    store = load_store()
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in store["chunks"]}
    query_terms = bm25_result.get("query_terms", [])
    missing_terms = bm25_result.get("missing_query_terms", [])
    lexical_signal = len(query_terms) - len(missing_terms)
    has_vector = len(vector_candidates) > 0
    primary_tool = "hybrid" if (lexical_signal >= 2 and has_vector) else ("bm25" if lexical_signal >= 1 else "vector")

    vector_by_id = {row["chunk_id"]: row for row in vector_candidates}
    bm25_by_id = {row["chunk_id"]: row for row in bm25_candidates}
    candidate_ids = set(vector_by_id) | set(bm25_by_id)
    rows = []
    for chunk_id in candidate_ids:
        chunk = chunks_by_id[chunk_id]
        vector_row = vector_by_id.get(chunk_id)
        bm25_row = bm25_by_id.get(chunk_id)
        matched_terms = bm25_row.get("matched_terms", []) if bm25_row else []
        term_coverage = len(matched_terms) / max(1, len(query_terms))
        rows.append(
            {
                "chunk_id": chunk_id,
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "chunk_text_preview": chunk["preview"],
                "full_chunk_text": chunk["chunk_text"],
                "vector_rank": vector_row.get("rank") if vector_row else None,
                "bm25_rank": bm25_row.get("rank") if bm25_row else None,
                "vector_score": vector_row.get("similarity_score", 0.0) if vector_row else 0.0,
                "bm25_score": bm25_row.get("bm25_score", 0.0) if bm25_row else 0.0,
                "matched_terms": matched_terms,
                "term_coverage": term_coverage,
                "source": source_label(vector_row.get("rank") if vector_row else None, bm25_row.get("rank") if bm25_row else None),
            }
        )

    normalize_scores(rows, "vector_score", "vector_norm")
    normalize_scores(rows, "bm25_score", "bm25_norm")
    for row in rows:
        source_bonus = 0.12 if row["source"] == "both" else 0.04
        row["critique_score"] = round(
            0.42 * row["vector_norm"]
            + 0.34 * row["bm25_norm"]
            + 0.18 * row["term_coverage"]
            + source_bonus,
            6,
        )
        if row["term_coverage"] == 0 and row["vector_norm"] < 0.35:
            row["rejection_reason"] = "weak semantic and lexical support"
        elif row["source"] == "vector-only" and lexical_signal >= 2:
            row["rejection_reason"] = "semantic-only hit; lexical support missing"
        elif row["source"] == "bm25-only" and row["bm25_norm"] < 0.4:
            row["rejection_reason"] = "lexical hit is too weak"
        else:
            row["rejection_reason"] = ""

    rows.sort(key=lambda item: (-item["critique_score"], item["page_number"], item["chunk_index"]))
    accepted = [row for row in rows if not row["rejection_reason"]][:top_k]
    retry_used = len(accepted) < top_k
    if retry_used:
        hybrid_result, _, hybrid_rows = run_hybrid_retrieval(query_text, top_k=top_k, candidate_k=16)
        accepted_ids = {row["chunk_id"] for row in accepted}
        for hybrid_row in hybrid_rows:
            if len(accepted) >= top_k:
                break
            if hybrid_row["chunk_id"] in accepted_ids:
                continue
            chunk = chunks_by_id[hybrid_row["chunk_id"]]
            accepted.append(
                {
                    "chunk_id": hybrid_row["chunk_id"],
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk["chunk_index"],
                    "chunk_text_preview": chunk["preview"],
                    "full_chunk_text": chunk["chunk_text"],
                    "vector_rank": hybrid_row.get("vector_rank"),
                    "bm25_rank": hybrid_row.get("bm25_rank"),
                    "vector_score": hybrid_row.get("vector_score", 0.0),
                    "bm25_score": hybrid_row.get("bm25_score", 0.0),
                    "matched_terms": hybrid_row.get("matched_terms", []),
                    "term_coverage": len(hybrid_row.get("matched_terms", [])) / max(1, len(query_terms)),
                    "source": hybrid_row.get("source", "both"),
                    "critique_score": hybrid_row.get("fusion_score", 0.0),
                    "rejection_reason": "",
                    "retry_added": True,
                }
            )
            accepted_ids.add(hybrid_row["chunk_id"])

    rejected = [row for row in rows if row["chunk_id"] not in {item["chunk_id"] for item in accepted}][:8]
    results = [
        {
            "rank": rank,
            "chunk_id": row["chunk_id"],
            "page_number": row["page_number"],
            "agent_score": row["critique_score"],
            "source": row["source"],
            "vector_rank": row.get("vector_rank"),
            "bm25_rank": row.get("bm25_rank"),
            "matched_terms": row.get("matched_terms", []),
            "accepted_reason": (
                "retry accepted by hybrid fallback"
                if row.get("retry_added")
                else "accepted after evidence critique"
            ),
            "chunk_text_preview": row["chunk_text_preview"],
            "full_chunk_text": row["full_chunk_text"],
        }
        for rank, row in enumerate(accepted[:top_k], start=1)
    ]

    tool_timeline = [
        {"step": 1, "tool": "planner", "label": "Plan", "status": "complete", "duration_ms": 8, "detail": f"Primary route: {primary_tool}"},
        {"step": 2, "tool": "vector", "label": "Vector retrieve", "status": "complete", "duration_ms": 42, "detail": f"{len(vector_candidates)} semantic candidates"},
        {"step": 3, "tool": "bm25", "label": "BM25 retrieve", "status": "complete", "duration_ms": 25, "detail": f"{len(bm25_candidates)} lexical candidates"},
        {"step": 4, "tool": "critic", "label": "Critique evidence", "status": "complete", "duration_ms": 14, "detail": f"{len(rejected)} rejected"},
        {"step": 5, "tool": "retry", "label": "Retry if needed", "status": "complete" if retry_used else "skipped", "duration_ms": 35 if retry_used else 0, "detail": "hybrid fallback" if retry_used else "enough evidence"},
        {"step": 6, "tool": "answer", "label": "Answer", "status": "ready", "duration_ms": 0, "detail": f"{len(results)} accepted evidence chunks"},
    ]
    flow_nodes = [
        {"id": "plan", "label": "Plan", "type": "decision", "status": "complete"},
        {"id": "choose", "label": f"Choose {primary_tool}", "type": "decision", "status": "complete"},
        {"id": "retrieve", "label": "Call tools", "type": "tool", "status": "complete"},
        {"id": "inspect", "label": "Inspect evidence", "type": "tool", "status": "complete"},
        {"id": "critique", "label": "Critique", "type": "decision", "status": "complete"},
        {"id": "retry", "label": "Retry", "type": "decision", "status": "complete" if retry_used else "skipped"},
        {"id": "answer", "label": "Answer", "type": "answer", "status": "ready"},
    ]
    flow_links = [
        {"source": "plan", "target": "choose"},
        {"source": "choose", "target": "retrieve"},
        {"source": "retrieve", "target": "inspect"},
        {"source": "inspect", "target": "critique"},
        {"source": "critique", "target": "retry"},
        {"source": "retry", "target": "answer"},
    ]
    scratchpad = [
        f"Query terms: {', '.join(query_terms[:10]) or 'none'}",
        f"Missing lexical terms: {', '.join(missing_terms) if missing_terms else 'none'}",
        f"Decision: use {primary_tool} because semantic candidates and lexical signal were inspected.",
        f"Critique rule: accept evidence with source agreement, term coverage, or strong semantic score.",
        f"Retry: {'hybrid fallback added evidence' if retry_used else 'not needed'}",
    ]

    result = {
        "mode": "agentic",
        "pipeline": "agentic_rag",
        "query": query_text,
        "query_terms": query_terms,
        "results": results,
        "agent_summary": {
            "primary_tool": primary_tool,
            "accepted_count": len(results),
            "rejected_count": len(rejected),
            "retry_used": retry_used,
            "tool_call_count": 3 if retry_used else 2,
        },
        "tool_timeline": tool_timeline,
        "scratchpad": scratchpad,
        "rejected_evidence": [
            {
                "chunk_id": row["chunk_id"],
                "page_number": row["page_number"],
                "source": row["source"],
                "agent_score": row["critique_score"],
                "reason": row["rejection_reason"] or "lower ranked after critique",
                "chunk_text_preview": row["chunk_text_preview"],
            }
            for row in rejected
        ],
        "accepted_path": [
            {"step": "plan", "label": "Plan"},
            {"step": "tool", "label": primary_tool},
            *[
                {"step": "evidence", "label": f"P{row['page_number']} #{rank}", "chunk_id": row["chunk_id"]}
                for rank, row in enumerate(accepted[:top_k], start=1)
            ],
            {"step": "answer", "label": "Final answer"},
        ],
    }
    vis = {
        "mode": "agentic",
        "pipeline": "agentic_rag",
        "query": query_text,
        "control_flow": {"nodes": flow_nodes, "links": flow_links},
        "tool_timeline": tool_timeline,
        "scratchpad": scratchpad,
        "rejected_evidence": result["rejected_evidence"],
        "accepted_path": result["accepted_path"],
        "candidate_piles": {
            "vector": vector_result["results"],
            "bm25": bm25_result["results"],
            "accepted": results,
            "rejected": result["rejected_evidence"],
        },
    }
    return result, vis, rows


STOP_BRIDGE_TERMS = {
    "what", "when", "where", "which", "with", "from", "that", "this", "these",
    "those", "into", "about", "between", "through", "their", "there", "then",
    "than", "have", "has", "had", "does", "did", "the", "and", "for", "are",
    "was", "were", "you", "your", "can", "how", "why", "explain", "define",
    "campusx", "complete", "course", "resource", "scratch", "learners",
    "professional", "professionals", "youtube", "playlist", "100daysofml",
    "intermediate", "beginners", "valuable", "advance", "comprehensive",
}


def extract_bridge_terms(text, query_text, limit=6):
    query_tokens = set(tokenize(query_text))
    counts = {}
    for token in tokenize(text):
        if len(token) < 5 or token in query_tokens or token in STOP_BRIDGE_TERMS:
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [term for term, _ in ranked[:limit]]


def run_multihop_retrieval(query_text, top_k=5):
    """Two-hop retrieval that makes the bridge query explicit."""
    hop1_result, _, hop1_candidates = run_hybrid_retrieval(query_text, top_k=top_k, candidate_k=12)
    hop1_top = hop1_result.get("results", [])[:3]
    bridge_terms = extract_bridge_terms(
        " ".join(row.get("full_chunk_text") or row.get("chunk_text_preview") or "" for row in hop1_top),
        query_text,
        limit=5,
    )
    bridge_entity = bridge_terms[0] if bridge_terms else ""
    hop2_query = f"{query_text} {bridge_entity}".strip() if bridge_entity else query_text
    hop2_result, _, hop2_candidates = run_hybrid_retrieval(hop2_query, top_k=top_k, candidate_k=12)

    store = load_store()
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in store["chunks"]}
    hop1_by_id = {row["chunk_id"]: row for row in hop1_candidates}
    hop2_by_id = {row["chunk_id"]: row for row in hop2_candidates}
    candidate_ids = set(hop1_by_id) | set(hop2_by_id)
    rows = []
    for chunk_id in candidate_ids:
        chunk = chunks_by_id[chunk_id]
        hop1 = hop1_by_id.get(chunk_id)
        hop2 = hop2_by_id.get(chunk_id)
        hop1_score = hop1.get("fusion_score", 0.0) if hop1 else 0.0
        hop2_score = hop2.get("fusion_score", 0.0) if hop2 else 0.0
        bridge_hits = [term for term in bridge_terms if term in tokenize(chunk.get("chunk_text", ""))]
        rows.append(
            {
                "chunk_id": chunk_id,
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "chunk_text_preview": chunk["preview"],
                "full_chunk_text": chunk["chunk_text"],
                "hop1_rank": hop1.get("rank") if hop1 else None,
                "hop2_rank": hop2.get("rank") if hop2 else None,
                "hop1_score": float(hop1_score),
                "hop2_score": float(hop2_score),
                "bridge_hits": bridge_hits,
            }
        )

    normalize_scores(rows, "hop1_score", "hop1_norm")
    normalize_scores(rows, "hop2_score", "hop2_norm")
    for row in rows:
        bridge_bonus = min(0.25, 0.08 * len(row["bridge_hits"]))
        row["multihop_score"] = round(
            0.42 * row["hop1_norm"]
            + 0.46 * row["hop2_norm"]
            + bridge_bonus
            + reciprocal_rank(row["hop1_rank"])
            + reciprocal_rank(row["hop2_rank"]),
            6,
        )
        if row["hop1_rank"] and row["hop2_rank"]:
            row["hop_role"] = "bridge-confirmed"
        elif row["hop1_rank"]:
            row["hop_role"] = "hop1-only"
        elif row["hop2_rank"]:
            row["hop_role"] = "hop2-only"
        else:
            row["hop_role"] = "candidate"

    rows.sort(key=lambda item: (-item["multihop_score"], item["page_number"], item["chunk_index"]))
    results = [
        {
            "rank": rank,
            "chunk_id": row["chunk_id"],
            "page_number": row["page_number"],
            "multihop_score": row["multihop_score"],
            "hop1_rank": row["hop1_rank"],
            "hop2_rank": row["hop2_rank"],
            "hop_role": row["hop_role"],
            "bridge_hits": row["bridge_hits"],
            "chunk_text_preview": row["chunk_text_preview"],
            "full_chunk_text": row["full_chunk_text"],
        }
        for rank, row in enumerate(rows[:top_k], start=1)
    ]

    evidence_a = hop1_top[0] if hop1_top else {}
    evidence_b = results[0] if results else {}
    graph_nodes = [
        {"id": "question", "label": "Question", "type": "question", "text": query_text},
        {"id": "hop1_query", "label": "Hop 1 query", "type": "query", "text": query_text},
        {"id": "evidence_a", "label": f"Evidence A P{evidence_a.get('page_number', '-')}", "type": "evidence", "chunk_id": evidence_a.get("chunk_id"), "preview": evidence_a.get("chunk_text_preview", "")},
        {"id": "bridge", "label": bridge_entity or "bridge term", "type": "bridge", "text": ", ".join(bridge_terms)},
        {"id": "hop2_query", "label": "Hop 2 query", "type": "query", "text": hop2_query},
        {"id": "evidence_b", "label": f"Evidence B P{evidence_b.get('page_number', '-')}", "type": "evidence", "chunk_id": evidence_b.get("chunk_id"), "preview": evidence_b.get("chunk_text_preview", "")},
        {"id": "answer", "label": "Final answer", "type": "answer"},
    ]
    graph_links = [
        {"source": "question", "target": "hop1_query", "label": "start"},
        {"source": "hop1_query", "target": "evidence_a", "label": "retrieve"},
        {"source": "evidence_a", "target": "bridge", "label": "extract bridge"},
        {"source": "bridge", "target": "hop2_query", "label": "rewrite"},
        {"source": "hop2_query", "target": "evidence_b", "label": "retrieve"},
        {"source": "evidence_b", "target": "answer", "label": "ground"},
    ]
    hops = [
        {
            "hop": 1,
            "query": query_text,
            "purpose": "Find the first useful passage and possible bridge terms.",
            "evidence": hop1_result.get("results", [])[:top_k],
            "bridge_terms": bridge_terms,
        },
        {
            "hop": 2,
            "query": hop2_query,
            "purpose": "Use the bridge term to look for the missing follow-up evidence.",
            "evidence": hop2_result.get("results", [])[:top_k],
            "bridge_terms": bridge_terms,
        },
    ]
    result = {
        "mode": "multihop",
        "pipeline": "multihop_langgraph_rag",
        "query": query_text,
        "hop_count": 2,
        "bridge_entity": bridge_entity,
        "bridge_terms": bridge_terms,
        "hop_queries": [query_text, hop2_query],
        "results": results,
        "hops": hops,
        "multihop_summary": {
            "hop_count": 2,
            "bridge_entity": bridge_entity or "none",
            "hop1_candidates": len(hop1_candidates),
            "hop2_candidates": len(hop2_candidates),
            "confirmed_count": len([row for row in rows if row["hop_role"] == "bridge-confirmed"]),
        },
    }
    vis = {
        "mode": "multihop",
        "pipeline": "multihop_langgraph_rag",
        "query": query_text,
        "reasoning_graph": {"nodes": graph_nodes, "links": graph_links},
        "hops": hops,
        "bridge_terms": bridge_terms,
        "hop_table": [
            {
                "chunk_id": row["chunk_id"],
                "page_number": row["page_number"],
                "hop1_rank": row["hop1_rank"],
                "hop2_rank": row["hop2_rank"],
                "role": row["hop_role"],
                "score": row["multihop_score"],
                "preview": row["chunk_text_preview"],
            }
            for row in rows[:12]
        ],
    }
    return result, vis, rows


def run_dynamic_retrieval(mode, query_text, top_k=5):
    if mode == "bm25":
        result, vis, _ = run_bm25_retrieval(query_text, top_k=top_k)
    elif mode == "hybrid":
        result, vis, _ = run_hybrid_retrieval(query_text, top_k=top_k)
    elif mode == "rerank":
        result, vis, _ = run_rerank_retrieval(query_text, top_k=top_k)
    elif mode == "graph":
        result, vis, _ = run_graph_retrieval(query_text, top_k=top_k)
    elif mode == "vectorless":
        result, vis, _ = run_vectorless_retrieval(query_text, top_k=top_k)
    elif mode == "agentic":
        result, vis, _ = run_agentic_retrieval(query_text, top_k=top_k)
    elif mode == "multihop":
        result, vis, _ = run_multihop_retrieval(query_text, top_k=top_k)
    else:
        result, vis, _ = run_vector_retrieval(query_text, top_k=top_k)
    return result, vis


def persist_query_artifacts(mode, result, vis):
    pipeline_config = get_pipeline_config(mode)
    prefix = pipeline_config["prefix"]
    write_json(FRONTEND_DATA_DIR / f"{prefix}_query_result.json", result)
    write_json(FRONTEND_DATA_DIR / f"{prefix}_vis.json", vis)


def get_pipeline_config(mode):
    return PIPELINE_CONFIG.get(mode or "naive", PIPELINE_CONFIG["naive"])


def run_pipeline_subprocess(file_path, mode):
    """Run the RAG pipeline as subprocess and capture events"""
    global pipeline_status, event_queue

    try:
        pipeline_status["state"] = "yellow"
        log_event("info", f"Pipeline starting... ({mode})", "pipeline")
        log_event("info", f"Processing file: {file_path}", "pipeline")

        # Build command to run the Python script with the uploaded file
        # Using conda env rag-multimodal to ensure all dependencies are available
        pipeline_config = get_pipeline_config(mode)
        cmd = [
            "conda",
            "run",
            "-n",
            "rag-multimodal",
            "python",
            pipeline_config["script"],
            "--input",
            file_path,
        ]

        # Run with subprocess and capture output
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )

        # Thread to read stdout
        def read_output(pipe, log_level):
            for line in pipe:
                line = line.strip()
                if line:
                    # Parse pipeline stage updates if they exist
                    if "Parsing PDF" in line or "Extracting pages" in line:
                        log_event("info", "Parsing PDF...", "parse")
                    elif "Chunking" in line:
                        log_event("info", "Chunking text...", "chunk")
                    elif "Embedding" in line or "embedding" in line.lower():
                        log_event("info", "Computing embeddings...", "embed")
                    elif "Retrieving" in line or "retrieval" in line.lower():
                        log_event("info", "Retrieving similar chunks...", "retrieve")
                    elif "Projecting" in line or "projection" in line.lower():
                        log_event("info", "Projecting to 2D space...", "project")
                    elif "Exporting" in line or "export" in line.lower():
                        log_event("info", "Exporting results...", "export")
                    else:
                        log_event("info", line, "pipeline")

        # Start output reading threads
        stdout_thread = threading.Thread(
            target=read_output, args=(process.stdout, "info")
        )
        stderr_thread = threading.Thread(
            target=read_output, args=(process.stderr, "warn")
        )
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()

        # Wait for completion
        return_code = process.wait()
        stdout_thread.join(timeout=2)
        stderr_thread.join(timeout=2)

        if return_code == 0:
            log_event("info", "Pipeline completed successfully", "pipeline")
            pipeline_status["state"] = "green"
            log_event("info", "Ready for queries", "system")
        else:
            log_event("error", f"Pipeline failed with code {return_code}", "pipeline")
            pipeline_status["state"] = "red"

    except Exception as e:
        log_event("error", f"Pipeline error: {str(e)}", "pipeline")
        pipeline_status["state"] = "red"


@app.route("/api/status", methods=["GET"])
def get_status():
    """Get current pipeline status"""
    return jsonify(pipeline_status)


@app.route("/api/logs", methods=["GET"])
def get_logs():
    """Get all logged events (SSE stream for real-time updates)"""

    def event_generator():
        # Send existing logs first
        for log in logs:
            yield f"data: {json.dumps(log)}\n\n"

        # Then stream new events
        while True:
            try:
                event = event_queue.get(timeout=30)  # 30s timeout
                yield f"data: {json.dumps(event)}\n\n"
            except:
                # Timeout or queue empty
                yield f"data: {json.dumps({'message': 'heartbeat'})}\n\n"

    return Response(
        event_generator(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/logs/clear", methods=["POST"])
def clear_logs():
    """Clear all logged events"""
    global logs, event_queue
    logs = []
    # Clear queue
    while not event_queue.empty():
        try:
            event_queue.get_nowait()
        except:
            break
    return jsonify({"status": "cleared"})


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Handle file upload and store for later pipeline execution"""
    global uploaded_file_path

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    try:
        # Save to temp location
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)

        uploaded_file_path = temp_path
        pipeline_status["state"] = "red"
        log_event("info", f"File uploaded: {file.filename}", "upload")

        return jsonify(
            {"status": "uploaded", "filename": file.filename, "path": temp_path}
        )
    except Exception as e:
        log_event("error", f"Upload error: {str(e)}", "upload")
        return jsonify({"error": str(e)}), 500


@app.route("/api/run", methods=["POST"])
def run_pipeline():
    """Trigger pipeline execution on the uploaded file"""
    global uploaded_file_path, current_pipeline_mode

    data = request.json or {}
    mode = data.get("mode") or "naive"
    current_pipeline_mode = mode if mode in PIPELINE_CONFIG else "naive"

    if not uploaded_file_path or not os.path.exists(uploaded_file_path):
        return jsonify({"error": "No file uploaded yet"}), 400

    # Clear previous logs
    clear_logs()
    log_event("info", f"Pipeline mode selected: {current_pipeline_mode}", "pipeline")

    # Run pipeline in background thread
    thread = threading.Thread(
        target=run_pipeline_subprocess,
        args=(uploaded_file_path, current_pipeline_mode),
        daemon=True,
    )
    thread.start()

    return jsonify({"status": "pipeline_started"})


@app.route("/api/query", methods=["POST"])
def run_query():
    """Run a query on the processed data"""
    data = request.json or {}
    query_text = data.get("query", "")
    mode = data.get("mode") or current_pipeline_mode

    if not query_text:
        return jsonify({"error": "No query provided"}), 400

    if pipeline_status["state"] != "green":
        return jsonify({"error": "Pipeline not ready. Please run pipeline first."}), 400

    try:
        log_event("info", f"Query received: '{query_text}'", "query")
        log_event("info", f"Query mode: {mode}", "query")
        log_event("info", "Running fresh retrieval against indexed PDF...", "query")
        result, vis = run_dynamic_retrieval(mode, query_text)
        log_event("info", "Retrieval completed", "query")

        result, vis, answer_payload = self_heal_answer(mode, query_text, result, vis)
        result.update(answer_payload)
        verdict = result.get("critic", {}).get("verdict", "unknown")
        if verdict == "accepted":
            log_event("info", "Critic accepted grounded answer", "critic")
        else:
            log_event("warn", f"Critic verdict: {verdict}", "critic")

        persist_query_artifacts(mode if mode in PIPELINE_CONFIG else "naive", result, vis)
        return jsonify(result)

    except Exception as e:
        log_event("error", f"Query error: {str(e)}", "query")
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok"})


@app.route("/api/evaluate", methods=["POST"])
def evaluate_query():
    """Evaluate retrieval quality using ground-truth relevance judgments."""
    data = request.json or {}
    query_text = (data.get("query") or "").strip()

    if not query_text:
        return jsonify({"error": "No query provided"}), 400

    if pipeline_status["state"] != "green":
        return jsonify({"error": "Pipeline not ready"}), 400

    try:
        from notebooks.evaluate import evaluate_across_methods, load_relevance_judgments, compute_binary_relevance
        import re

        judgments = load_relevance_judgments()
        keyword_config = judgments.get("keyword_filter_config", {}).get(query_text)
        if not keyword_config:
            return jsonify({"error": f"No evaluation config for query: {query_text}"}), 400

        all_results = {}
        for mode_key in ("naive", "bm25", "hybrid", "rerank", "graph", "vectorless", "agentic", "multihop"):
            try:
                result, _ = run_dynamic_retrieval(mode_key, query_text)
                all_results[mode_key] = {query_text: result.get("results", [])}
            except Exception:
                all_results[mode_key] = {query_text: []}

        comparison = evaluate_across_methods(all_results)

        return jsonify({"query": query_text, "comparison": comparison})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    log_event("info", "Backend API starting on port 5000", "system")
    app.run(debug=True, port=5000, use_reloader=False)
