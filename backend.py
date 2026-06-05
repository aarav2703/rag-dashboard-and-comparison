import os
import json
import subprocess
import tempfile
import threading
import urllib.error
import urllib.request
import re
import hashlib
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
from graph_cache import load_graph_artifacts

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
    "hybrid": {
        "script": os.path.join(os.path.dirname(__file__), "notebooks", "00_build_faiss_corpus.py"),
        "prefix": "hybrid_rag",
    },
    "graph": {
        "script": os.path.join(os.path.dirname(__file__), "notebooks", "00_build_faiss_corpus.py"),
        "prefix": "graph_rag",
    },
    "agentic": {
        "script": os.path.join(os.path.dirname(__file__), "notebooks", "00_build_faiss_corpus.py"),
        "prefix": "agentic_rag",
    },
    "crag": {
        "script": os.path.join(os.path.dirname(__file__), "notebooks", "00_build_faiss_corpus.py"),
        "prefix": "crag_rag",
    },
}

embedding_model_cache = {"model": None, "name": None}
INSUFFICIENT_EVIDENCE_ANSWER = (
    "I don't have enough information in the retrieved evidence to answer that reliably."
)
TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


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
    if os.environ.get("RAG_ALLOW_EXTERNAL_TOOLS", "true").strip().lower() in FALSE_VALUES:
        raise RuntimeError("External tools are disabled")
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


def external_tools_enabled():
    return os.environ.get("RAG_ALLOW_EXTERNAL_TOOLS", "true").strip().lower() not in FALSE_VALUES


def run_tavily_search_tool(query, max_results=5):
    """Run Tavily search and normalize web results into evidence-like rows."""
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not external_tools_enabled():
        return {
            "enabled": False,
            "provider": "tavily",
            "error": "External tools are disabled",
            "results": [],
        }
    if not api_key:
        return {
            "enabled": False,
            "provider": "tavily",
            "error": "TAVILY_API_KEY is not configured",
            "results": [],
        }

    payload = {
        "query": query,
        "max_results": max(1, min(int(max_results or 5), 8)),
        "search_depth": "advanced",
        "include_answer": False,
        "include_raw_content": "markdown",
    }
    request = urllib.request.Request(
        "https://api.tavily.com/search",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=35) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {
            "enabled": True,
            "provider": "tavily",
            "error": str(exc),
            "results": [],
        }

    rows = []
    for rank, item in enumerate(response_data.get("results", [])[: payload["max_results"]], start=1):
        title = str(item.get("title") or item.get("url") or f"Web result {rank}").strip()
        snippet = str(item.get("content") or item.get("snippet") or "").strip()
        raw_content = str(item.get("raw_content") or "").strip()
        text = raw_content or snippet
        rows.append(
            {
                "rank": rank,
                "chunk_id": f"web::{rank}::{hashlib.sha1(str(item.get('url') or title).encode('utf-8')).hexdigest()[:12]}",
                "page_number": "web",
                "source": "web_search",
                "provider": "tavily",
                "url": item.get("url", ""),
                "title": title,
                "snippet": snippet,
                "chunk_text_preview": (snippet or text)[:260],
                "full_chunk_text": text[:2200],
                "web_score": float(item.get("score") or 0.0),
            }
        )
    return {
        "enabled": True,
        "provider": "tavily",
        "query": query,
        "results": rows,
        "raw_result_count": len(response_data.get("results", [])),
    }


def parse_json_payload(content, fallback):
    text = (content or "").strip()
    if not text:
        return fallback
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return fallback
    return fallback


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


def grade_retrieved_evidence(query_text, rows):
    query_terms = unique_preserve_order(tokenize(query_text))
    graded = []
    for row in rows:
        matched_terms = row.get("matched_terms", [])
        coverage = len(matched_terms) / max(1, len(query_terms))
        score = float(row.get("reranker_score") or row.get("hybrid_score") or row.get("similarity_score") or 0.0)
        if coverage >= 0.45 or score >= 0.72:
            verdict = "correct"
            action = "answer"
            reason = "high ranked evidence with enough lexical or semantic support"
        elif coverage >= 0.18 or score >= 0.38:
            verdict = "ambiguous"
            action = "rewrite"
            reason = "partial support; query rewrite may improve evidence"
        else:
            verdict = "incorrect"
            action = "fallback"
            reason = "weak support for the query"
        graded.append(
            {
                "chunk_id": row.get("chunk_id"),
                "page_number": row.get("page_number"),
                "grade": verdict,
                "action": action,
                "reason": reason,
                "term_coverage": round(coverage, 6),
                "score": round(score, 6),
                "preview": row.get("chunk_text_preview") or row.get("preview", ""),
            }
        )
    return graded


def llm_grade_retrieved_evidence(query_text, rows):
    if not rows:
        return []
    evidence = [
        {
            "chunk_id": row.get("chunk_id"),
            "page_number": row.get("page_number"),
            "score": row.get("reranker_score") or row.get("hybrid_score") or row.get("similarity_score"),
            "text": (row.get("full_chunk_text") or row.get("chunk_text_preview") or "")[:900],
        }
        for row in rows[:8]
    ]
    prompt = (
        "Grade retrieved evidence for this question before answer generation. "
        "Return JSON only with key grades, an array. Each item must include chunk_id, grade "
        "(correct, ambiguous, or incorrect), grade_reason, confidence, suggested_query, fallback_action.\n\n"
        f"Question: {query_text}\n\n"
        f"Evidence: {json.dumps(evidence, ensure_ascii=False)}"
    )
    try:
        content = deepseek_chat(
            [
                {"role": "system", "content": "You are a strict CRAG retrieval evaluator. Return only JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=900,
            temperature=0.0,
            timeout=45,
        )
        payload = parse_json_payload(content, {"grades": []})
        grades = payload.get("grades", [])
    except Exception:
        return []
    by_id = {row.get("chunk_id"): row for row in rows}
    cleaned = []
    for item in grades:
        if not isinstance(item, dict) or item.get("chunk_id") not in by_id:
            continue
        grade = str(item.get("grade", "ambiguous")).lower()
        if grade not in {"correct", "ambiguous", "incorrect"}:
            grade = "ambiguous"
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.6))))
        except Exception:
            confidence = 0.6
        source = by_id[item["chunk_id"]]
        cleaned.append(
            {
                "chunk_id": item["chunk_id"],
                "page_number": source.get("page_number"),
                "grade": grade,
                "action": "answer" if grade == "correct" else "rewrite" if grade == "ambiguous" else "fallback",
                "reason": item.get("grade_reason") or item.get("reason") or "LLM retrieval grade",
                "grade_reason": item.get("grade_reason") or item.get("reason") or "LLM retrieval grade",
                "confidence": round(confidence, 3),
                "suggested_query": item.get("suggested_query") or query_text,
                "fallback_action": item.get("fallback_action") or ("answer" if grade == "correct" else "rewrite" if grade == "ambiguous" else "web_search_fallback"),
                "score": source.get("reranker_score", 0.0),
                "preview": source.get("chunk_text_preview", ""),
                "grader_source": "deepseek",
            }
        )
    return cleaned


def llm_grade_retrieval_set(query_text, rows):
    if not rows:
        return {}
    evidence = [
        {
            "chunk_id": row.get("chunk_id"),
            "page_number": row.get("page_number"),
            "score": row.get("reranker_score") or row.get("hybrid_score") or row.get("similarity_score"),
            "text": (row.get("full_chunk_text") or row.get("chunk_text_preview") or "")[:900],
        }
        for row in rows[:8]
    ]
    prompt = (
        "Grade the whole retrieved evidence set for CRAG. "
        "Return JSON only with retrieval_verdict (correct, ambiguous, incorrect), "
        "missing_evidence_summary, recommended_action (answer, rewrite, web_fallback), "
        "confidence, and suggested_query.\n\n"
        f"Question: {query_text}\n\n"
        f"Evidence: {json.dumps(evidence, ensure_ascii=False)}"
    )
    try:
        content = deepseek_chat(
            [
                {"role": "system", "content": "You are a strict CRAG retrieval-set evaluator. Return only JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.0,
            timeout=35,
        )
        payload = parse_json_payload(content, {})
    except Exception:
        return {}
    verdict = str(payload.get("retrieval_verdict", "ambiguous")).lower()
    if verdict not in {"correct", "ambiguous", "incorrect"}:
        verdict = "ambiguous"
    action = str(payload.get("recommended_action", "")).lower()
    if action not in {"answer", "rewrite", "web_fallback"}:
        action = "answer" if verdict == "correct" else "rewrite" if verdict == "ambiguous" else "web_fallback"
    try:
        confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.6))))
    except Exception:
        confidence = 0.6
    return {
        "retrieval_verdict": verdict,
        "missing_evidence_summary": payload.get("missing_evidence_summary", ""),
        "recommended_action": action,
        "confidence": round(confidence, 3),
        "suggested_query": payload.get("suggested_query") or query_text,
        "retrieval_grade_source": "deepseek",
    }


def run_crag_retrieval(query_text, top_k=5):
    rerank_result, rerank_vis, rerank_rows = run_rerank_retrieval(query_text, top_k=top_k, candidate_k=20)
    graded = llm_grade_retrieved_evidence(query_text, rerank_result.get("results", [])) or grade_retrieved_evidence(query_text, rerank_result.get("results", []))
    set_grade = llm_grade_retrieval_set(query_text, rerank_result.get("results", []))
    grade_counts = {}
    for row in graded:
        grade_counts[row["grade"]] = grade_counts.get(row["grade"], 0) + 1

    if set_grade:
        branch = set_grade["retrieval_verdict"]
    elif grade_counts.get("correct", 0) >= max(1, top_k // 2):
        branch = "correct"
    elif grade_counts.get("ambiguous", 0):
        branch = "ambiguous"
    else:
        branch = "incorrect"

    fallback_source = "none"
    web_results = []
    correction_attempts = []
    if branch == "correct":
        branch_action = "answer"
        correction_query = query_text
        fallback_results = []
    elif branch == "ambiguous":
        branch_action = "rewrite"
        correction_query = (
            set_grade.get("suggested_query")
            if set_grade and set_grade.get("suggested_query") != query_text
            else None
        ) or next((row.get("suggested_query") for row in graded if row.get("suggested_query") and row.get("suggested_query") != query_text), None) or f"{query_text} supporting evidence details".strip()
        correction_attempts.append({"action": "rewrite", "query": correction_query, "source": "local"})
        fallback_result, _, fallback_rows = run_hybrid_retrieval(correction_query, top_k=top_k, candidate_k=12)
        fallback_results = fallback_result.get("results", [])
        fallback_source = "local"
        existing_ids = {row["chunk_id"] for row in rerank_result.get("results", [])}
        for fallback_row in fallback_rows:
            if len(rerank_result["results"]) >= top_k:
                break
            if fallback_row["chunk_id"] in existing_ids:
                continue
            chunk_preview = fallback_row.get("preview") or fallback_row.get("chunk_text_preview", "")
            rerank_result["results"].append(
                {
                    "rank": len(rerank_result["results"]) + 1,
                    "chunk_id": fallback_row["chunk_id"],
                    "page_number": fallback_row["page_number"],
                    "reranker_score": fallback_row.get("fusion_score", 0.0),
                    "movement": 0,
                    "movement_label": "fallback",
                    "matched_terms": fallback_row.get("matched_terms", []),
                    "chunk_text_preview": chunk_preview,
                    "full_chunk_text": fallback_row.get("full_chunk_text", ""),
                }
            )
            existing_ids.add(fallback_row["chunk_id"])
    else:
        branch_action = "fallback"
        correction_query = (
            set_grade.get("suggested_query")
            if set_grade and set_grade.get("suggested_query") != query_text
            else f"{query_text} overview context".strip()
        )
        web_payload = run_tavily_search_tool(correction_query, max_results=top_k)
        web_results = web_payload.get("results", [])
        if web_results:
            fallback_source = "web"
            fallback_results = web_results
            rerank_result["results"] = web_results[:top_k]
            correction_attempts.append({"action": "web_fallback", "query": correction_query, "source": "tavily", "result_count": len(web_results)})
        else:
            fallback_source = "local"
            fallback_result, _, _ = run_hybrid_retrieval(correction_query, top_k=top_k, candidate_k=12)
            fallback_results = fallback_result.get("results", [])
            rerank_result["results"] = fallback_results[:top_k]
            correction_attempts.append({"action": "local_fallback", "query": correction_query, "source": "hybrid", "result_count": len(fallback_results), "warning": web_payload.get("error")})

    result = {
        **rerank_result,
        "mode": "crag",
        "pipeline": "corrective_rag_with_rerank",
        "crag_summary": {
            "branch": branch,
            "action": branch_action,
            "grade_counts": grade_counts,
            "correction_query": correction_query,
            "fallback_count": len(fallback_results),
            "fallback_source": fallback_source,
            "grader_source": graded[0].get("grader_source", "heuristic") if graded else "none",
            "average_grade_confidence": round(sum(row.get("confidence", 0.0) for row in graded) / max(1, len(graded)), 3),
            "retrieval_verdict": set_grade.get("retrieval_verdict", branch) if set_grade else branch,
            "retrieval_grade_source": set_grade.get("retrieval_grade_source", "heuristic") if set_grade else "heuristic",
            "missing_evidence_summary": set_grade.get("missing_evidence_summary", "") if set_grade else "",
            "recommended_action": set_grade.get("recommended_action", branch_action) if set_grade else branch_action,
        },
        "retrieval_verdict": set_grade.get("retrieval_verdict", branch) if set_grade else branch,
        "missing_evidence_summary": set_grade.get("missing_evidence_summary", "") if set_grade else "",
        "recommended_action": set_grade.get("recommended_action", branch_action) if set_grade else branch_action,
        "retrieval_grade_source": set_grade.get("retrieval_grade_source", "heuristic") if set_grade else "heuristic",
        "evidence_grades": graded,
        "correction_query": correction_query,
        "fallback_source": fallback_source,
        "fallback_results": fallback_results,
        "web_results": web_results,
        "correction_attempts": correction_attempts,
    }
    vis = {
        **rerank_vis,
        "mode": "crag",
        "pipeline": "corrective_rag_with_rerank",
        "evidence_grades": graded,
        "web_results": web_results,
        "correction_attempts": correction_attempts,
        "crag_flow": {
            "branch": branch,
            "action": branch_action,
            "correction_query": correction_query,
            "fallback_source": fallback_source,
            "nodes": [
                {"id": "query", "label": "Query", "type": "query"},
                {"id": "retrieve", "label": "Retrieve", "type": "tool"},
                {"id": "rerank", "label": "Rerank", "type": "tool"},
                {"id": "grade", "label": "Grade", "type": "decision"},
                {"id": branch_action, "label": branch_action.title(), "type": "action"},
                {"id": "answer", "label": "Answer", "type": "answer"},
            ],
            "links": [
                {"source": "query", "target": "retrieve"},
                {"source": "retrieve", "target": "rerank"},
                {"source": "rerank", "target": "grade"},
                {"source": "grade", "target": branch_action},
                {"source": branch_action, "target": "answer"},
            ],
        },
    }
    return result, vis, rerank_rows


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


def normalize_entity_label(value):
    value = re.sub(r"^\[|\]$", "", str(value or "")).strip().lower()
    value = re.sub(r"[^a-z0-9\s]+", " ", value)
    return " ".join(value.split())


def extract_chunk_title(text):
    match = re.match(r"\s*\[([^\]]{2,120})\]", str(text or ""))
    return match.group(1).strip() if match else ""


def add_graph_score(score_breakdown, section_id, key, value, detail=None):
    if not section_id:
        return
    row = score_breakdown.setdefault(section_id, {})
    row[key] = round(row.get(key, 0.0) + float(value), 6)
    if detail:
        row.setdefault("details", []).append(detail)


def llm_extract_relationships(chunks, max_chunks=18):
    selected = chunks[:max_chunks]
    if not selected:
        return []
    blocks = []
    for chunk in selected:
        blocks.append(
            f"chunk_id: {chunk['chunk_id']}\n"
            f"page: {chunk.get('page_number')}\n"
            f"text: {chunk.get('chunk_text', '')[:900]}"
        )
    prompt = (
        "Extract a compact knowledge graph from these document chunks. "
        "Return JSON only with key relationships, an array of objects. "
        "Each object must have source_entity, relationship, target_entity, evidence_chunk_id, confidence.\n\n"
        + "\n\n---\n\n".join(blocks)
    )
    try:
        content = deepseek_chat(
            [
                {"role": "system", "content": "You extract precise document relationship triples. Return only JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=900,
            temperature=0.0,
            timeout=45,
        )
        payload = parse_json_payload(content, {"relationships": []})
        relationships = payload.get("relationships", payload if isinstance(payload, list) else [])
    except Exception:
        return []

    chunk_ids = {chunk["chunk_id"] for chunk in chunks}
    cleaned = []
    for row in relationships:
        if not isinstance(row, dict):
            continue
        source = canonical_entity(str(row.get("source_entity", "")))
        target = canonical_entity(str(row.get("target_entity", "")))
        relation = canonical_entity(str(row.get("relationship", ""))).lower().replace(" ", "_")
        evidence_id = str(row.get("evidence_chunk_id", ""))
        if not source or not target or not relation or evidence_id not in chunk_ids:
            continue
        try:
            confidence = max(0.0, min(1.0, float(row.get("confidence", 0.7))))
        except Exception:
            confidence = 0.7
        cleaned.append(
            {
                "source_entity": source,
                "relationship": relation,
                "target_entity": target,
                "evidence_chunk_id": evidence_id,
                "confidence": round(confidence, 3),
            }
        )
    return cleaned[:80]


def fallback_relationships(chunks, max_rows=60):
    rows = []
    for chunk in chunks:
        entities = extract_entities(chunk.get("chunk_text", ""), max_entities=8)
        for left_index, left in enumerate(entities[:5]):
            for right in entities[left_index + 1 : 5]:
                rows.append(
                    {
                        "source_entity": left,
                        "relationship": "co_occurs_with",
                        "target_entity": right,
                        "evidence_chunk_id": chunk["chunk_id"],
                        "confidence": 0.45,
                    }
                )
                if len(rows) >= max_rows:
                    return rows
    return rows


def build_relationship_communities(graph, max_communities=8):
    entity_edges = [
        edge for edge in graph.get("edges", [])
        if str(edge.get("source", "")).startswith("entity:") and str(edge.get("target", "")).startswith("entity:")
    ]
    adjacency = {}
    for edge in entity_edges:
        adjacency.setdefault(edge["source"], set()).add(edge["target"])
        adjacency.setdefault(edge["target"], set()).add(edge["source"])
    seen = set()
    communities = []
    nodes_by_id = {node["id"]: node for node in graph.get("nodes", [])}
    for node_id in adjacency:
        if node_id in seen:
            continue
        stack = [node_id]
        component = []
        seen.add(node_id)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        labels = [nodes_by_id.get(item, {}).get("label", item) for item in component]
        communities.append(
            {
                "id": f"community:{len(communities) + 1}",
                "label": ", ".join(labels[:3]) or f"Community {len(communities) + 1}",
                "node_ids": component,
                "entity_labels": labels,
            }
        )
    return sorted(communities, key=lambda item: -len(item["node_ids"]))[:max_communities]


def summarize_graph_communities(communities, relationships, max_items=6):
    if not communities:
        return []
    rel_text = "\n".join(
        f"- {row['source_entity']} {row['relationship']} {row['target_entity']}"
        for row in relationships[:40]
    )
    prompt = (
        "Summarize these graph communities using the relationship triples. "
        "Return JSON only with key community_summaries, an array of objects with community_id, title, summary.\n\n"
        f"Communities: {json.dumps(communities[:max_items], ensure_ascii=False)}\n\n"
        f"Relationships:\n{rel_text}"
    )
    try:
        content = deepseek_chat(
            [
                {"role": "system", "content": "You summarize knowledge graph communities concisely. Return only JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=700,
            temperature=0.0,
            timeout=45,
        )
        payload = parse_json_payload(content, {"community_summaries": []})
        summaries = payload.get("community_summaries", [])
    except Exception:
        summaries = []
    by_id = {item.get("community_id"): item for item in summaries if isinstance(item, dict)}
    rows = []
    for community in communities[:max_items]:
        summary = by_id.get(community["id"], {})
        rows.append(
            {
                "community_id": community["id"],
                "title": summary.get("title") or community["label"],
                "summary": summary.get("summary") or f"Entities: {', '.join(community.get('entity_labels', [])[:6])}",
                "node_ids": community["node_ids"],
            }
        )
    return rows


def build_document_graph(chunks, focus_chunk_ids=None):
    focus_chunk_ids = set(focus_chunk_ids or [])
    selected_chunks = chunks
    nodes = {}
    edges = {}
    entity_to_sections = {}
    section_entities = {}
    title_to_sections = {}

    document_id = "document:pdf"
    nodes[document_id] = {"id": document_id, "label": "Document", "type": "document", "weight": len(selected_chunks)}

    relationships = llm_extract_relationships(selected_chunks) or fallback_relationships(selected_chunks)

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

        chunk_title = extract_chunk_title(chunk["chunk_text"])
        entities = extract_entities(chunk["chunk_text"])
        if chunk_title:
            entities = unique_preserve_order([chunk_title, *entities])
            title_to_sections.setdefault(normalize_entity_label(chunk_title), set()).add(section_id)
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

    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in selected_chunks}
    for rel in relationships:
        source = rel["source_entity"]
        target = rel["target_entity"]
        relation = rel["relationship"] or "related_to"
        evidence_chunk_id = rel["evidence_chunk_id"]
        source_id = graph_node_id("entity", source)
        target_id = graph_node_id("entity", target)
        for entity_id, label in ((source_id, source), (target_id, target)):
            nodes.setdefault(
                entity_id,
                {"id": entity_id, "label": label, "type": "entity", "weight": 0},
            )
            nodes[entity_id]["weight"] += 1
        edge_key = (source_id, target_id, relation)
        edges[edge_key] = {
            "source": source_id,
            "target": target_id,
            "type": relation,
            "weight": max(1, int(round(rel.get("confidence", 0.7) * 3))),
            "evidence_chunk_id": evidence_chunk_id,
            "confidence": rel.get("confidence", 0.7),
        }
        section_id = graph_node_id("section", evidence_chunk_id)
        if evidence_chunk_id in chunks_by_id:
            entity_to_sections.setdefault(source_id, set()).add(section_id)
            entity_to_sections.setdefault(target_id, set()).add(section_id)
            edges[(source_id, section_id, "evidence_for")] = {
                "source": source_id,
                "target": section_id,
                "type": "evidence_for",
                "weight": 1,
            }
            edges[(target_id, section_id, "evidence_for")] = {
                "source": target_id,
                "target": section_id,
                "type": "evidence_for",
                "weight": 1,
            }

    return {
        "nodes": list(nodes.values()),
        "edges": list(edges.values()),
        "entity_to_sections": entity_to_sections,
        "section_entities": section_entities,
        "relationships": relationships,
    }


def run_graph_retrieval(query_text, top_k=5):
    store = load_store()
    chunks = store["chunks"]
    vector_result, _, vector_candidates = run_vector_retrieval(query_text, top_k=top_k, candidate_k=18)
    bm25_result, _, bm25_candidates = run_bm25_retrieval(query_text, top_k=top_k, candidate_k=18)
    seeded_ids = {row["chunk_id"] for row in vector_candidates + bm25_candidates}
    graph = load_graph_artifacts(STORE_DIR, chunks)
    nodes_by_id = graph.get("_nodes_by_id") or {node["id"]: node for node in graph["nodes"]}
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    query_entities = extract_entities(query_text, max_entities=8)
    query_terms = set(tokenize(query_text))
    vector_rank_by_id = {row["chunk_id"]: row["rank"] for row in vector_candidates}
    bm25_rank_by_id = {row["chunk_id"]: row["rank"] for row in bm25_candidates}

    matched_entity_ids = []
    entity_nodes = graph.get("_entity_nodes") or [node for node in graph["nodes"] if node["type"] == "entity"]
    normalized_query_entities = {normalize_entity_label(entity) for entity in query_entities}
    normalized_query_text = normalize_entity_label(query_text)
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
    score_breakdown = {}
    path_rows = []
    relationship_adjacency = graph.get("_relationship_adjacency", {})
    for candidate in vector_candidates:
        section_id = graph_node_id("section", candidate["chunk_id"])
        seed_score = reciprocal_rank(candidate["rank"], k=20) * 5
        section_scores[section_id] = section_scores.get(section_id, 0.0) + seed_score
        add_graph_score(score_breakdown, section_id, "vector_seed", seed_score, f"vector rank {candidate['rank']}")
    for candidate in bm25_candidates:
        section_id = graph_node_id("section", candidate["chunk_id"])
        seed_score = reciprocal_rank(candidate["rank"], k=20) * 4
        section_scores[section_id] = section_scores.get(section_id, 0.0) + seed_score
        add_graph_score(score_breakdown, section_id, "bm25_seed", seed_score, f"bm25 rank {candidate['rank']}")

    for entity_id in matched_entity_ids:
        sections = graph["entity_to_sections"].get(entity_id, set())
        for section_id in sections:
            section_scores[section_id] = section_scores.get(section_id, 0.0) + 6.0
            add_graph_score(score_breakdown, section_id, "direct_entity", 6.0, nodes_by_id[entity_id]["label"])
            path_rows.append(
                {
                    "query": query_text,
                    "entity_id": entity_id,
                    "entity": nodes_by_id[entity_id]["label"],
                    "section_id": section_id,
                    "section": nodes_by_id[section_id]["label"],
                    "edge_type": "mentions",
                    "path_depth": 0,
                    "score": 6.0,
                }
            )

        for edge in relationship_adjacency.get(entity_id, [])[:24]:
            neighbor_id = None
            if edge["source"] == entity_id:
                neighbor_id = edge["target"]
            elif edge["target"] == entity_id:
                neighbor_id = edge["source"]
            if not neighbor_id:
                continue
            for section_id in graph["entity_to_sections"].get(neighbor_id, set()):
                path_score = 2.5 + float(edge.get("confidence", 0.4)) * 2.0 + min(1.5, edge.get("weight", 1) * 0.2)
                section_scores[section_id] = section_scores.get(section_id, 0.0) + path_score
                add_graph_score(score_breakdown, section_id, "one_hop_relationship", path_score, f"{nodes_by_id[entity_id]['label']} -> {nodes_by_id.get(neighbor_id, {}).get('label', neighbor_id)}")
                path_rows.append(
                    {
                        "query": query_text,
                        "entity_id": entity_id,
                        "entity": nodes_by_id[entity_id]["label"],
                        "related_entity_id": neighbor_id,
                        "related_entity": nodes_by_id[neighbor_id]["label"],
                        "section_id": section_id,
                        "section": nodes_by_id[section_id]["label"],
                        "edge_type": edge["type"],
                        "path_depth": 1,
                        "score": round(path_score, 6),
                    }
                )
                for second_edge in relationship_adjacency.get(neighbor_id, [])[:16]:
                    second_neighbor_id = None
                    if second_edge["source"] == neighbor_id and second_edge["target"] != entity_id:
                        second_neighbor_id = second_edge["target"]
                    elif second_edge["target"] == neighbor_id and second_edge["source"] != entity_id:
                        second_neighbor_id = second_edge["source"]
                    if not second_neighbor_id:
                        continue
                    second_score = 1.1 + float(second_edge.get("confidence", 0.35))
                    for second_section_id in graph["entity_to_sections"].get(second_neighbor_id, set()):
                        section_scores[second_section_id] = section_scores.get(second_section_id, 0.0) + second_score
                        add_graph_score(score_breakdown, second_section_id, "two_hop_relationship", second_score, f"{nodes_by_id[entity_id]['label']} -> {nodes_by_id.get(neighbor_id, {}).get('label', neighbor_id)} -> {nodes_by_id.get(second_neighbor_id, {}).get('label', second_neighbor_id)}")
                        path_rows.append(
                            {
                                "query": query_text,
                                "entity_id": entity_id,
                                "entity": nodes_by_id[entity_id]["label"],
                                "related_entity_id": second_neighbor_id,
                                "related_entity": nodes_by_id[second_neighbor_id]["label"],
                                "section_id": second_section_id,
                                "section": nodes_by_id[second_section_id]["label"],
                                "edge_type": second_edge["type"],
                                "path_depth": 2,
                                "score": round(second_score, 6),
                            }
                        )

    title_matches = []
    for section_id, entities in graph["section_entities"].items():
        section_title = normalize_entity_label(entities[0] if entities else "")
        if not section_title:
            continue
        exact_entity_match = section_title in normalized_query_entities
        query_contains_title = section_title and section_title in normalized_query_text
        title_contains_query_entity = any(entity and entity in section_title for entity in normalized_query_entities)
        if exact_entity_match or query_contains_title or title_contains_query_entity:
            title_score = 8.0 if exact_entity_match or query_contains_title else 4.0
            section_scores[section_id] = section_scores.get(section_id, 0.0) + title_score
            add_graph_score(score_breakdown, section_id, "title_entity_match", title_score, entities[0])
            title_matches.append(
                {
                    "entity": entities[0],
                    "section_id": section_id,
                    "section": nodes_by_id.get(section_id, {}).get("label", section_id),
                    "score": title_score,
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
    subgraph_edges_by_key = {}
    edge_adjacency = graph.get("_edge_adjacency", {})
    for node_id in list(used_node_ids):
        for edge in edge_adjacency.get(node_id, []):
            if edge["source"] in used_node_ids and edge["target"] in used_node_ids:
                subgraph_edges_by_key[(edge["source"], edge["target"], edge.get("type"))] = edge
    for section_id in used_section_ids:
        for edge in graph.get("_support_edges_by_target", {}).get(section_id, []):
            subgraph_edges_by_key[(edge["source"], edge["target"], edge.get("type"))] = edge
            used_node_ids.add(edge["source"])
    subgraph_edges = list(subgraph_edges_by_key.values())
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
                "score_breakdown": score_breakdown.get(section_id, {}),
                "chunk_text_preview": chunk.get("preview", ""),
                "full_chunk_text": chunk.get("chunk_text", ""),
            }
        )

    communities = graph.get("communities", [])
    community_summaries = graph.get("community_summaries", [])
    community_hits = []
    for community in communities:
        labels = [normalize_entity_label(label) for label in community.get("entity_labels", [])]
        overlap = [label for label in labels if label in normalized_query_entities or label in normalized_query_text]
        related_sections = set()
        for node_id in community.get("node_ids", []):
            related_sections |= set(graph["entity_to_sections"].get(node_id, set()))
        hit_score = len(overlap) * 3 + len(related_sections & used_section_ids)
        if hit_score:
            community_hits.append(
                {
                    "community_id": community["id"],
                    "label": community["label"],
                    "score": hit_score,
                    "matched_entities": overlap[:8],
                    "used_section_count": len(related_sections & used_section_ids),
                }
            )
    community_hits.sort(key=lambda item: -item["score"])
    subgraph_retrieval_trace = [
        {
            "rank": rank,
            "section_id": section_id,
            "score": round(score, 6),
            "matched_path_count": len([path for path in path_rows if path.get("section_id") == section_id]),
            "used_in_answer_subgraph": section_id in used_section_ids,
            "score_breakdown": score_breakdown.get(section_id, {}),
        }
        for rank, (section_id, score) in enumerate(ranked_sections, start=1)
    ]

    result = {
        "mode": "graph",
        "pipeline": "graph_rag_lite",
        "query": query_text,
        "query_entities": query_entities,
        "matched_entities": [nodes_by_id[entity_id]["label"] for entity_id in matched_entity_ids if entity_id in nodes_by_id],
        "entity_matches": [
            {"entity_id": entity_id, "entity": nodes_by_id[entity_id]["label"]}
            for entity_id in matched_entity_ids
            if entity_id in nodes_by_id
        ] + title_matches[:12],
        "results": results,
        "path_explanation": path_rows[:20],
        "relationship_paths": sorted(path_rows, key=lambda item: -item.get("score", 0.0))[:40],
        "relationships": graph.get("relationships", [])[:80],
        "communities": communities,
        "community_summaries": community_summaries,
        "community_hits": community_hits[:10],
        "subgraph_retrieval_trace": subgraph_retrieval_trace,
        "graph_score_breakdown": {
            section_id: score_breakdown.get(section_id, {})
            for section_id, _score in ranked_sections
        },
        "graph_stats": {
            "node_count": len(graph["nodes"]),
            "edge_count": len(graph["edges"]),
            "used_node_count": len(highlighted_subgraph["nodes"]),
            "claim_count": len([node for node in graph["nodes"] if node["type"] == "claim"]),
            "seeded_chunk_count": len(seeded_ids),
            "graph_cache": graph.get("graph_metadata", {}),
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
        "community_summaries": community_summaries,
        "community_hits": community_hits[:10],
        "relationships": graph.get("relationships", [])[:80],
        "relationship_paths": sorted(path_rows, key=lambda item: -item.get("score", 0.0))[:40],
        "subgraph_retrieval_trace": subgraph_retrieval_trace,
        "graph_score_breakdown": {
            section_id: score_breakdown.get(section_id, {})
            for section_id, _score in ranked_sections
        },
        "filters": {
            "node_types": ["document", "entity", "section", "claim"],
            "edge_types": ["mentions", "supports", "co-occurs", "located_in", "evidence_for", "relationship"],
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


def llm_plan_agentic_steps(query_text, state):
    prompt = (
        "Plan a RAG tool loop. Return JSON only with key decisions, an array of 2-5 objects. "
        "Each object must include decision, tool, reason_summary, confidence, next_action. "
        "Do not include hidden chain-of-thought.\n\n"
        f"Question: {query_text}\n"
        f"State: {json.dumps(state, ensure_ascii=False)}"
    )
    try:
        content = deepseek_chat(
            [
                {"role": "system", "content": "You are a concise RAG planner. Return structured decisions only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=600,
            temperature=0.0,
            timeout=35,
        )
        payload = parse_json_payload(content, {"decisions": []})
        decisions = payload.get("decisions", [])
    except Exception:
        decisions = []
    cleaned = []
    available = set(state.get("available_tools", []))
    for item in decisions:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool", "hybrid")).lower()
        if tool not in available:
            tool = "hybrid"
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.65))))
        except Exception:
            confidence = 0.65
        cleaned.append(
            {
                "step": len(cleaned) + 1,
                "decision": str(item.get("decision") or f"Use {tool}"),
                "tool": tool,
                "query": str(item.get("query") or query_text),
                "reason_summary": str(item.get("reason_summary") or "Selected from available retrieval signals."),
                "confidence": round(confidence, 3),
                "result_count": 0,
                "next_action": str(item.get("next_action") or "retrieve"),
                "planner_source": "deepseek",
            }
        )
    if cleaned:
        return cleaned[:5]
    fallback_tool = "hybrid" if state.get("query_terms") else "vector"
    return [
        {
            "step": 1,
            "decision": f"Use {fallback_tool} retrieval",
            "tool": fallback_tool,
            "query": query_text,
            "reason_summary": "Fallback planner selected the strongest available local route.",
            "confidence": 0.62,
            "result_count": 0,
            "next_action": "retrieve",
            "planner_source": "heuristic",
        },
        {
            "step": 2,
            "decision": "Use second-hop retrieval if bridge terms appear",
            "tool": "second_hop",
            "query": query_text,
            "reason_summary": "Multi-hop questions may need bridge evidence.",
            "confidence": 0.58,
            "result_count": 0,
            "next_action": "extract_bridge",
            "planner_source": "heuristic",
        },
    ]


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
    planner_decisions = llm_plan_agentic_steps(
        query_text,
        {
            "query_terms": query_terms,
            "missing_terms": missing_terms,
            "vector_candidates": len(vector_candidates),
            "bm25_candidates": len(bm25_candidates),
            "available_tools": ["vector", "hybrid", "graph", "rerank", "second_hop", "web_search", "answer"],
        },
    )
    if planner_decisions:
        primary_tool = planner_decisions[0].get("tool") or primary_tool

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
    planner_confidence = min([row.get("confidence", 0.65) for row in planner_decisions], default=0.65)
    retry_used = len(accepted) < top_k or planner_confidence < 0.45
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
    multihop_result, multihop_vis, _ = run_multihop_retrieval(query_text, top_k=top_k)
    bridge_terms = multihop_result.get("bridge_terms", [])
    hop_queries = multihop_result.get("hop_queries", [query_text])
    for hop_row in multihop_result.get("results", []):
        if len(accepted) >= top_k:
            break
        if hop_row["chunk_id"] in {item["chunk_id"] for item in accepted}:
            continue
        accepted.append(
            {
                "chunk_id": hop_row["chunk_id"],
                "page_number": hop_row["page_number"],
                "chunk_index": chunks_by_id[hop_row["chunk_id"]]["chunk_index"],
                "chunk_text_preview": hop_row["chunk_text_preview"],
                "full_chunk_text": hop_row["full_chunk_text"],
                "vector_rank": None,
                "bm25_rank": None,
                "matched_terms": [],
                "source": hop_row.get("hop_role", "multi-hop"),
                "critique_score": hop_row.get("multihop_score", 0.0),
                "rejection_reason": "",
                "multi_hop_added": True,
            }
        )
    tool_execution_trace = []
    executed_tools = {"vector", "bm25", "second_hop"}
    no_new_rounds = 0
    for decision in planner_decisions[:3]:
        tool = decision.get("tool", "hybrid")
        tool_query = decision.get("query") or query_text
        if tool == "answer":
            decision["result_count"] = len(accepted)
            tool_execution_trace.append({**decision, "status": "stopped"})
            break
        if len(accepted) >= top_k and decision.get("confidence", 0.0) >= 0.72:
            decision["result_count"] = len(accepted)
            tool_execution_trace.append({**decision, "status": "answer_ready"})
            break
        if tool in executed_tools and tool not in {"web_search"}:
            decision["result_count"] = 0
            tool_execution_trace.append({**decision, "status": "skipped_duplicate"})
            continue

        before_count = len(accepted)
        tool_rows = []
        status = "complete"
        if tool == "hybrid":
            _tool_result, _tool_vis, hybrid_rows = run_hybrid_retrieval(tool_query, top_k=top_k, candidate_k=16)
            tool_rows = [
                {
                    "chunk_id": row["chunk_id"],
                    "page_number": row["page_number"],
                    "chunk_index": chunks_by_id[row["chunk_id"]]["chunk_index"],
                    "chunk_text_preview": row.get("preview") or row.get("chunk_text_preview", ""),
                    "full_chunk_text": row.get("full_chunk_text", ""),
                    "vector_rank": row.get("vector_rank"),
                    "bm25_rank": row.get("bm25_rank"),
                    "matched_terms": row.get("matched_terms", []),
                    "source": row.get("source", "hybrid"),
                    "critique_score": row.get("fusion_score", 0.0),
                    "rejection_reason": "",
                    "tool_added": "hybrid",
                }
                for row in hybrid_rows[:top_k]
            ]
        elif tool == "rerank":
            rerank_result, _rerank_vis, _rerank_rows = run_rerank_retrieval(tool_query, top_k=top_k, candidate_k=18)
            tool_rows = [
                {
                    "chunk_id": row["chunk_id"],
                    "page_number": row["page_number"],
                    "chunk_index": chunks_by_id[row["chunk_id"]]["chunk_index"],
                    "chunk_text_preview": row.get("chunk_text_preview", ""),
                    "full_chunk_text": row.get("full_chunk_text", ""),
                    "vector_rank": row.get("before_rank"),
                    "bm25_rank": None,
                    "matched_terms": row.get("matched_terms", []),
                    "source": "rerank",
                    "critique_score": row.get("reranker_score", 0.0),
                    "rejection_reason": "",
                    "tool_added": "rerank",
                }
                for row in rerank_result.get("results", [])
            ]
        elif tool == "graph":
            graph_result, _graph_vis, _graph_rows = run_graph_retrieval(tool_query, top_k=top_k)
            tool_rows = [
                {
                    "chunk_id": row["chunk_id"],
                    "page_number": row["page_number"],
                    "chunk_index": chunks_by_id.get(row["chunk_id"], {}).get("chunk_index", 0),
                    "chunk_text_preview": row.get("chunk_text_preview", ""),
                    "full_chunk_text": row.get("full_chunk_text", ""),
                    "vector_rank": row.get("vector_rank"),
                    "bm25_rank": row.get("bm25_rank"),
                    "matched_terms": row.get("matched_entities", []),
                    "source": "graph",
                    "critique_score": row.get("graph_score", 0.0),
                    "rejection_reason": "",
                    "tool_added": "graph",
                }
                for row in graph_result.get("results", [])
                if row.get("chunk_id") in chunks_by_id
            ]
        elif tool == "web_search":
            web_payload = run_tavily_search_tool(tool_query, max_results=5)
            status = "complete" if web_payload.get("results") else "unavailable"
            tool_rows = [
                {
                    "chunk_id": row["chunk_id"],
                    "page_number": row["page_number"],
                    "chunk_index": 0,
                    "chunk_text_preview": row.get("chunk_text_preview", ""),
                    "full_chunk_text": row.get("full_chunk_text", ""),
                    "vector_rank": None,
                    "bm25_rank": None,
                    "matched_terms": [],
                    "source": "web_search",
                    "critique_score": row.get("web_score", 0.0),
                    "rejection_reason": "",
                    "tool_added": "web_search",
                    "url": row.get("url", ""),
                    "title": row.get("title", ""),
                    "snippet": row.get("snippet", ""),
                    "provider": row.get("provider", "tavily"),
                }
                for row in web_payload.get("results", [])
            ]
            if web_payload.get("error"):
                decision["error"] = web_payload["error"]
        executed_tools.add(tool)
        accepted_ids = {row["chunk_id"] for row in accepted}
        for tool_row in tool_rows:
            if len(accepted) >= top_k:
                break
            if tool_row["chunk_id"] in accepted_ids:
                continue
            accepted.append(tool_row)
            accepted_ids.add(tool_row["chunk_id"])
        added_count = len(accepted) - before_count
        decision["result_count"] = added_count
        tool_execution_trace.append({**decision, "status": status})
        no_new_rounds = no_new_rounds + 1 if added_count == 0 else 0
        if no_new_rounds >= 2:
            break
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
                "accepted from multi-hop bridge retrieval"
                if row.get("multi_hop_added")
                else
                f"accepted from {row.get('tool_added')} tool"
                if row.get("tool_added")
                else
                "retry accepted by hybrid fallback"
                if row.get("retry_added")
                else "accepted after evidence critique"
            ),
            "chunk_text_preview": row["chunk_text_preview"],
            "full_chunk_text": row["full_chunk_text"],
            "url": row.get("url"),
            "title": row.get("title"),
            "snippet": row.get("snippet"),
        }
        for rank, row in enumerate(accepted[:top_k], start=1)
    ]

    tool_timeline = [
        {"step": 1, "tool": "planner", "label": "Plan", "status": "complete", "duration_ms": 8, "detail": f"Primary route: {primary_tool}"},
        {"step": 2, "tool": "vector", "label": "Vector retrieve", "status": "complete", "duration_ms": 42, "detail": f"{len(vector_candidates)} semantic candidates"},
        {"step": 3, "tool": "bm25", "label": "BM25 retrieve", "status": "complete", "duration_ms": 25, "detail": f"{len(bm25_candidates)} lexical candidates"},
        {"step": 4, "tool": "bridge", "label": "Extract bridge", "status": "complete", "duration_ms": 18, "detail": ", ".join(bridge_terms[:4]) or "no bridge terms"},
        {"step": 5, "tool": "hop2", "label": "Second-hop retrieve", "status": "complete", "duration_ms": 46, "detail": hop_queries[-1] if hop_queries else query_text},
        {"step": 6, "tool": "critic", "label": "Decide evidence", "status": "complete", "duration_ms": 14, "detail": f"{len(rejected)} rejected"},
        *[
            {
                "step": 7 + index,
                "tool": item.get("tool"),
                "label": item.get("tool", "tool").replace("_", " ").title(),
                "status": item.get("status", "complete"),
                "duration_ms": 0,
                "detail": f"{item.get('result_count', 0)} new results | {item.get('query', query_text)}",
            }
            for index, item in enumerate(tool_execution_trace)
        ],
        {"step": 7 + len(tool_execution_trace), "tool": "retry", "label": "Retry if needed", "status": "complete" if retry_used else "skipped", "duration_ms": 35 if retry_used else 0, "detail": "hybrid fallback" if retry_used else "enough evidence"},
        {"step": 8 + len(tool_execution_trace), "tool": "answer", "label": "Answer", "status": "ready", "duration_ms": 0, "detail": f"{len(results)} accepted evidence chunks"},
    ]
    flow_nodes = [
        {"id": "plan", "label": "Plan", "type": "decision", "status": "complete"},
        {"id": "choose", "label": f"Choose {primary_tool}", "type": "decision", "status": "complete"},
        {"id": "retrieve", "label": "Call tools", "type": "tool", "status": "complete"},
        {"id": "graph", "label": "Graph", "type": "tool", "status": "complete" if any(item.get("tool") == "graph" for item in tool_execution_trace) else "skipped"},
        {"id": "rerank", "label": "Rerank", "type": "tool", "status": "complete" if any(item.get("tool") == "rerank" for item in tool_execution_trace) else "skipped"},
        {"id": "web", "label": "Web", "type": "tool", "status": "complete" if any(item.get("tool") == "web_search" for item in tool_execution_trace) else "skipped"},
        {"id": "bridge", "label": "Bridge", "type": "tool", "status": "complete"},
        {"id": "hop2", "label": "Hop 2", "type": "tool", "status": "complete"},
        {"id": "critique", "label": "Decide", "type": "decision", "status": "complete"},
        {"id": "retry", "label": "Retry", "type": "decision", "status": "complete" if retry_used else "skipped"},
        {"id": "answer", "label": "Answer", "type": "answer", "status": "ready"},
    ]
    flow_links = [
        {"source": "plan", "target": "choose"},
        {"source": "choose", "target": "retrieve"},
        {"source": "retrieve", "target": "graph"},
        {"source": "retrieve", "target": "rerank"},
        {"source": "retrieve", "target": "web"},
        {"source": "retrieve", "target": "bridge"},
        {"source": "bridge", "target": "hop2"},
        {"source": "hop2", "target": "critique"},
        {"source": "critique", "target": "retry"},
        {"source": "retry", "target": "answer"},
    ]
    scratchpad = [
        f"Query terms: {', '.join(query_terms[:10]) or 'none'}",
        f"Missing lexical terms: {', '.join(missing_terms) if missing_terms else 'none'}",
        f"Decision: use {primary_tool} because semantic candidates and lexical signal were inspected.",
        f"Bridge terms: {', '.join(bridge_terms[:6]) if bridge_terms else 'none found'}",
        f"Second-hop query: {hop_queries[-1] if hop_queries else query_text}",
        f"Decision rule: accept evidence with source agreement, bridge confirmation, term coverage, or strong semantic score.",
        f"Tool loop: {len(tool_execution_trace)} planned tool executions",
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
            "tool_call_count": len(tool_timeline),
            "bridge_terms": bridge_terms,
            "hop_count": multihop_result.get("hop_count", 2),
            "planner_confidence": round(planner_confidence, 3),
        },
        "planner_decisions": planner_decisions,
        "tool_execution_trace": tool_execution_trace,
        "bridge_terms": bridge_terms,
        "hop_queries": hop_queries,
        "hops": multihop_result.get("hops", []),
        "reasoning_graph": multihop_vis.get("reasoning_graph", {}),
        "hop_table": multihop_vis.get("hop_table", []),
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
        "planner_decisions": planner_decisions,
        "tool_execution_trace": tool_execution_trace,
        "reasoning_graph": multihop_vis.get("reasoning_graph", {}),
        "hops": multihop_result.get("hops", []),
        "bridge_terms": bridge_terms,
        "hop_table": multihop_vis.get("hop_table", []),
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
    if mode == "hybrid":
        result, vis, _ = run_hybrid_retrieval(query_text, top_k=top_k)
    elif mode == "graph":
        result, vis, _ = run_graph_retrieval(query_text, top_k=top_k)
    elif mode == "agentic":
        result, vis, _ = run_agentic_retrieval(query_text, top_k=top_k)
    elif mode == "crag":
        result, vis, _ = run_crag_retrieval(query_text, top_k=top_k)
    else:
        result, vis, _ = run_vector_retrieval(query_text, top_k=top_k)
    return result, vis


def persist_query_artifacts(mode, result, vis):
    pipeline_config = get_pipeline_config(mode)
    prefix = pipeline_config["prefix"]
    write_json(FRONTEND_DATA_DIR / f"{prefix}_query_result.json", result)
    write_json(FRONTEND_DATA_DIR / f"{prefix}_vis.json", vis)


def attach_standard_answer(query_text, result):
    answer, answer_source = generate_answer_with_fallback(
        query_text,
        result.get("results", []),
    )
    return {
        "answer": answer,
        "answer_source": answer_source,
        "answer_model": "deepseek-chat" if answer_source == "deepseek" else "fallback",
        "evidence_count": len(result.get("results", [])),
    }


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

        if mode == "crag":
            result, vis, answer_payload = self_heal_answer(mode, query_text, result, vis)
            result.update(answer_payload)
            verdict = result.get("critic", {}).get("verdict", "unknown")
            if verdict == "accepted":
                log_event("info", "CRAG critic accepted grounded answer", "critic")
            else:
                log_event("warn", f"CRAG critic verdict: {verdict}", "critic")
        else:
            result.update(attach_standard_answer(query_text, result))

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
        for mode_key in ("naive", "hybrid", "graph", "agentic", "crag"):
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
