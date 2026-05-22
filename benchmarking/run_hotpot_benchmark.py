#!/usr/bin/env python
"""Run all retrieval modes against HotpotQA benchmark queries."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "benchmarking" / "artifacts"
RESULTS_DIR = ROOT / "benchmarking" / "results"
DEFAULT_MODES = (
    "naive",
    "bm25",
    "hybrid",
    "rerank",
    "graph",
    "vectorless",
    "agentic",
    "multihop",
)


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


def import_retrieval():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from backend import run_dynamic_retrieval  # noqa: PLC0415

    return run_dynamic_retrieval


def compact_result(result: dict[str, Any], top_k: int | None) -> dict[str, Any]:
    results = result.get("results", [])
    if top_k is not None:
        results = results[:top_k]
    return {
        "mode": result.get("mode"),
        "pipeline": result.get("pipeline"),
        "query": result.get("query"),
        "results": results,
        "total_returned": len(results),
    }


def run_benchmark(
    artifacts_dir: Path,
    output_dir: Path,
    modes: list[str],
    limit: int | None,
    top_k: int | None,
    fail_fast: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    queries = load_queries(artifacts_dir, limit)
    run_dynamic_retrieval = import_retrieval()

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
                elapsed_ms = round((time.perf_counter() - mode_started) * 1000, 2)
                mode_payload = compact_result(result, top_k)
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
        "elapsed_seconds": elapsed_seconds,
        "error_count": len(raw_results["errors"]),
    }
    raw_results["metadata"] = metadata
    return raw_results, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HotpotQA retrieval benchmark.")
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--modes", default="all", help="Comma-separated modes or `all`.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = parse_modes(args.modes)
    raw_results, metadata = run_benchmark(
        artifacts_dir=args.artifacts_dir,
        output_dir=args.output_dir,
        modes=modes,
        limit=args.limit,
        top_k=args.top_k,
        fail_fast=args.fail_fast,
    )
    write_json(args.output_dir / "hotpot_raw_results.json", raw_results)
    write_json(args.output_dir / "hotpot_run_metadata.json", metadata)
    print(f"Wrote raw results: {args.output_dir / 'hotpot_raw_results.json'}")
    print(f"Wrote run metadata: {args.output_dir / 'hotpot_run_metadata.json'}")


if __name__ == "__main__":
    main()
