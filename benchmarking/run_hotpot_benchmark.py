#!/usr/bin/env python
"""Run all retrieval modes against HotpotQA benchmark queries."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "benchmarking" / "artifacts"
RESULTS_DIR = ROOT / "benchmarking" / "results"
STORE_METADATA_PATH = ROOT / "backend_or_exports" / "current_corpus" / "metadata.json"
DEFAULT_MODES = (
    "naive",
    "hybrid",
    "graph",
    "agentic",
    "crag",
)
SCORE_KEYS = (
    "similarity_score",
    "hybrid_score",
    "graph_score",
    "agent_score",
    "crag_score",
    "reranker_score",
    "bm25_score",
    "fusion_score",
    "web_score",
)
PREVIEW_LIMIT = 180


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def parse_modes(raw_modes: str) -> list[str]:
    if raw_modes.strip().lower() == "all":
        return list(DEFAULT_MODES)
    modes = [mode.strip() for mode in raw_modes.split(",") if mode.strip()]
    unknown = sorted(set(modes) - set(DEFAULT_MODES))
    if unknown:
        raise ValueError(f"Unknown modes: {', '.join(unknown)}")
    return modes


def load_queries(artifacts_dir: Path, limit: int | None) -> list[dict[str, Any]]:
    queries_path = artifacts_dir / "hotpot_queries.json"
    payload = load_json(queries_path)
    queries = payload.get("queries", [])
    if limit is not None:
        queries = queries[:limit]
    if not queries:
        raise RuntimeError(f"No queries found in {queries_path}. Run Stage 1 first.")
    return queries


def validate_hotpot_store(artifacts_dir: Path) -> dict[str, Any]:
    if not STORE_METADATA_PATH.exists():
        raise RuntimeError(
            f"Missing Hotpot store metadata at {STORE_METADATA_PATH}. "
            "Run `python benchmarking\\build_hotpot_store.py` before benchmarking."
        )
    store_metadata = load_json(STORE_METADATA_PATH)
    chunk_metadata = load_json(artifacts_dir / "hotpot_chunks.json").get("metadata", {})
    if store_metadata.get("source") != "hotpotqa":
        raise RuntimeError(
            "The current app corpus is not a HotpotQA store. "
            "Run `python benchmarking\\build_hotpot_store.py` before benchmarking."
        )
    expected_chunks = chunk_metadata.get("chunk_count")
    if expected_chunks and store_metadata.get("chunk_count") != expected_chunks:
        raise RuntimeError(
            "The current HotpotQA store does not match the artifact chunk count "
            f"({store_metadata.get('chunk_count')} store vs {expected_chunks} artifacts). "
            "Re-run `python benchmarking\\build_hotpot_store.py`."
        )
    return store_metadata


def import_retrieval():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from backend import generate_answer_with_fallback, run_dynamic_retrieval  # noqa: PLC0415

    return run_dynamic_retrieval, generate_answer_with_fallback


def parse_judge_scores(content: str) -> dict[str, float]:
    scores = {}
    for line in str(content or "").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        if key in {"faithfulness", "answer_relevancy"}:
            try:
                scores[key] = max(0.0, min(1.0, float(value.strip())))
            except ValueError:
                continue
    return scores


def judge_answer_quality(question: str, answer: str, retrieved_results: list[dict[str, Any]]) -> dict[str, Any]:
    from backend import deepseek_chat, format_evidence_blocks  # noqa: PLC0415

    if not answer:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "llm_metric_source": "empty-answer"}
    context = "\n\n".join(format_evidence_blocks(retrieved_results, max_chunks=5))
    prompt = (
        "Score this RAG answer from 0 to 1.\n"
        "FAITHFULNESS means every factual claim is supported by the retrieved context.\n"
        "ANSWER_RELEVANCY means the answer directly addresses the question.\n"
        "Reply exactly as:\n"
        "FAITHFULNESS: <number>\n"
        "ANSWER_RELEVANCY: <number>\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        f"Retrieved context:\n{context}"
    )
    try:
        content = deepseek_chat(
            [
                {"role": "system", "content": "You are a strict RAG evaluator. Return only numeric scores."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=80,
            temperature=0.0,
            timeout=35,
        )
        scores = parse_judge_scores(content)
        return {
            "faithfulness": scores.get("faithfulness", 0.0),
            "answer_relevancy": scores.get("answer_relevancy", 0.0),
            "llm_metric_source": "deepseek",
        }
    except Exception as exc:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "llm_metric_source": "error", "llm_metric_error": str(exc)}


def primary_score(row: dict[str, Any]) -> float | None:
    for key in SCORE_KEYS:
        value = row.get(key)
        if isinstance(value, (int, float)):
            return round(float(value), 6)
    return None


def compact_chunk(row: dict[str, Any]) -> dict[str, Any]:
    preview = str(
        row.get("preview")
        or row.get("chunk_text_preview")
        or row.get("chunk_text")
        or row.get("full_chunk_text")
        or ""
    ).strip()
    return {
        "chunk_id": row.get("chunk_id"),
        "rank": row.get("rank"),
        "score": primary_score(row),
        "page_number": row.get("page_number"),
        "title": row.get("title") or row.get("section_label"),
        "sentence_index": row.get("sentence_index"),
        "source": row.get("source"),
        "preview": preview[:PREVIEW_LIMIT],
    }


def compact_result(result: dict[str, Any], top_k: int | None) -> dict[str, Any]:
    results = result.get("results", [])
    if top_k is not None:
        results = results[:top_k]
    return {
        "mode": result.get("mode"),
        "pipeline": result.get("pipeline"),
        "query": result.get("query"),
        "results": [compact_chunk(row) for row in results],
        "total_returned": len(results),
    }


def run_benchmark(
    artifacts_dir: Path,
    output_dir: Path,
    modes: list[str],
    limit: int | None,
    top_k: int | None,
    fail_fast: bool,
    include_answers: bool,
    include_llm_metrics: bool,
    skip_store_check: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    queries = load_queries(artifacts_dir, limit)
    store_metadata = {} if skip_store_check else validate_hotpot_store(artifacts_dir)
    run_dynamic_retrieval, generate_answer_with_fallback = import_retrieval()
    previous_external_setting = os.environ.get("RAG_ALLOW_EXTERNAL_TOOLS")
    if not include_answers and not include_llm_metrics:
        os.environ["RAG_ALLOW_EXTERNAL_TOOLS"] = "false"

    try:
        raw_results: dict[str, Any] = {"queries": {}, "errors": []}
        started = time.perf_counter()

        for query_number, query in enumerate(queries, start=1):
            query_id = str(query["query_id"])
            question = str(query["question"])
            raw_results["queries"][query_id] = {
                "query_id": query_id,
                "question": question,
                "answer": query.get("answer"),
                "type": query.get("type"),
                "level": query.get("level"),
                "modes": {},
            }

            print(f"[{query_number}/{len(queries)}] {query_id}: {question}")
            for mode in modes:
                mode_started = time.perf_counter()
                try:
                    result, _vis = run_dynamic_retrieval(mode, question, top_k=top_k or 5)
                    full_results = result.get("results", [])
                    mode_payload = compact_result(result, top_k)
                    if include_answers:
                        answer_context = full_results[:top_k] if top_k is not None else full_results
                        answer, answer_source = generate_answer_with_fallback(question, answer_context)
                        mode_payload["answer"] = answer
                        mode_payload["answer_source"] = answer_source
                        if include_llm_metrics:
                            mode_payload.update(judge_answer_quality(question, answer, answer_context))
                    elapsed_ms = round((time.perf_counter() - mode_started) * 1000, 2)
                    mode_payload["elapsed_ms"] = elapsed_ms
                    raw_results["queries"][query_id]["modes"][mode] = mode_payload
                    print(f"  {mode}: {mode_payload['total_returned']} results in {elapsed_ms} ms")
                except Exception as exc:
                    elapsed_ms = round((time.perf_counter() - mode_started) * 1000, 2)
                    error = {
                        "query_id": query_id,
                        "mode": mode,
                        "error": str(exc),
                        "elapsed_ms": elapsed_ms,
                    }
                    raw_results["errors"].append(error)
                    raw_results["queries"][query_id]["modes"][mode] = {
                        "error": str(exc),
                        "elapsed_ms": elapsed_ms,
                        "results": [],
                        "total_returned": 0,
                    }
                    print(f"  {mode}: ERROR after {elapsed_ms} ms: {exc}")
                    if fail_fast:
                        raise

        elapsed_seconds = round(time.perf_counter() - started, 3)
        metadata = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "artifacts_dir": str(artifacts_dir),
            "output_dir": str(output_dir),
            "query_count": len(queries),
            "modes": modes,
            "top_k": top_k,
            "store_metadata": store_metadata,
            "include_answers": include_answers,
            "include_llm_metrics": include_llm_metrics,
            "result_format": "compact",
            "result_fields": ["chunk_id", "rank", "score", "page_number", "title", "sentence_index", "source", "preview"],
            "preview_limit": PREVIEW_LIMIT,
            "external_tools_enabled": os.environ.get("RAG_ALLOW_EXTERNAL_TOOLS", "true").strip().lower() != "false",
            "elapsed_seconds": elapsed_seconds,
            "error_count": len(raw_results["errors"]),
        }
        raw_results["metadata"] = metadata
        return raw_results, metadata
    finally:
        if previous_external_setting is None:
            os.environ.pop("RAG_ALLOW_EXTERNAL_TOOLS", None)
        else:
            os.environ["RAG_ALLOW_EXTERNAL_TOOLS"] = previous_external_setting


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HotpotQA retrieval benchmark.")
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--modes", default="all", help="Comma-separated modes or `all`.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--skip-store-check", action="store_true", help="Allow running without verifying that the active corpus is the HotpotQA store.")
    parser.add_argument("--include-answers", action="store_true", help="Generate answers for EM/token-F1 answer metrics.")
    parser.add_argument("--include-llm-metrics", action="store_true", help="Generate LLM judge faithfulness/relevancy metrics; implies --include-answers.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = parse_modes(args.modes)
    include_answers = args.include_answers or args.include_llm_metrics
    raw_results, metadata = run_benchmark(
        artifacts_dir=args.artifacts_dir,
        output_dir=args.output_dir,
        modes=modes,
        limit=args.limit,
        top_k=args.top_k,
        fail_fast=args.fail_fast,
        include_answers=include_answers,
        include_llm_metrics=args.include_llm_metrics,
        skip_store_check=args.skip_store_check,
    )
    write_json(args.output_dir / "hotpot_raw_results.json", raw_results)
    write_json(args.output_dir / "hotpot_run_metadata.json", metadata)
    print(f"Wrote raw results: {args.output_dir / 'hotpot_raw_results.json'}")
    print(f"Wrote run metadata: {args.output_dir / 'hotpot_run_metadata.json'}")


if __name__ == "__main__":
    main()
