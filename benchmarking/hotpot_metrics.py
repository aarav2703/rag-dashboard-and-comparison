#!/usr/bin/env python
"""Compute HotpotQA retrieval benchmark metrics from raw mode results."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "benchmarking" / "artifacts"
RESULTS_DIR = ROOT / "benchmarking" / "results"
K_VALUES = (3, 5, 10)
SUMMARY_METRICS = (
    "ndcg@3",
    "ndcg@5",
    "ndcg@10",
    "recall@3",
    "recall@5",
    "recall@10",
    "precision@3",
    "precision@5",
    "precision@10",
    "mrr",
    "map",
    "supporting_fact_hit_count",
    "supporting_fact_hit_rate",
    "all_supporting_facts_found",
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python benchmarking\\run_hotpot_benchmark.py --limit 25 --modes all --top-k 10` first."
        )
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def dcg_at_k(relevance: list[float], k: int) -> float:
    return sum((2**rel - 1) / math.log2(index + 2) for index, rel in enumerate(relevance[:k]))


def ndcg_at_k(relevance: list[float], k: int) -> float:
    if k <= 0:
        return 0.0
    dcg = dcg_at_k(relevance, k)
    ideal = sorted(relevance, reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    return dcg / ideal_dcg if ideal_dcg else 0.0


def mrr(relevance: list[float]) -> float:
    for index, rel in enumerate(relevance):
        if rel > 0:
            return 1.0 / (index + 1)
    return 0.0


def recall_at_k(relevance: list[float], total_relevant: int, k: int) -> float:
    if total_relevant <= 0:
        return 0.0
    return sum(1 for rel in relevance[:k] if rel > 0) / total_relevant


def precision_at_k(relevance: list[float], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for rel in relevance[:k] if rel > 0) / k


def average_precision(relevance: list[float], total_relevant: int) -> float:
    if total_relevant <= 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for index, rel in enumerate(relevance):
        if rel > 0:
            hits += 1
            precision_sum += hits / (index + 1)
    return precision_sum / total_relevant


def evaluate_query_mode(
    query_id: str,
    mode: str,
    query_payload: dict[str, Any],
    mode_payload: dict[str, Any],
    relevance_payload: dict[str, Any],
) -> dict[str, Any]:
    judgment = relevance_payload.get("queries", {}).get(query_id, {})
    relevant_chunks = set(judgment.get("relevant_chunks", []))
    results = mode_payload.get("results", [])
    retrieved_ids = [result.get("chunk_id") for result in results]
    relevance = [1.0 if chunk_id in relevant_chunks else 0.0 for chunk_id in retrieved_ids]

    support_count = len(relevant_chunks)
    hit_ids = sorted(set(retrieved_ids) & relevant_chunks)
    hit_count = len(hit_ids)
    metrics = {
        "query_id": query_id,
        "mode": mode,
        "question": query_payload.get("question"),
        "type": query_payload.get("type") or judgment.get("type"),
        "level": query_payload.get("level") or judgment.get("level"),
        "total_retrieved": len(results),
        "total_relevant": support_count,
        "supporting_fact_hit_count": hit_count,
        "supporting_fact_hit_rate": hit_count / support_count if support_count else 0.0,
        "all_supporting_facts_found": 1.0 if support_count and hit_count == support_count else 0.0,
        "retrieved_relevant_chunks": hit_ids,
        "error": mode_payload.get("error"),
        "elapsed_ms": mode_payload.get("elapsed_ms"),
    }
    for k in K_VALUES:
        effective_k = min(k, len(relevance))
        metrics[f"ndcg@{k}"] = ndcg_at_k(relevance, effective_k)
        metrics[f"recall@{k}"] = recall_at_k(relevance, support_count, effective_k)
        metrics[f"precision@{k}"] = precision_at_k(relevance, effective_k)
    metrics["mrr"] = mrr(relevance)
    metrics["map"] = average_precision(relevance, support_count)
    return round_metric_values(metrics)


def round_metric_values(metrics: dict[str, Any]) -> dict[str, Any]:
    rounded = {}
    for key, value in metrics.items():
        rounded[key] = round(value, 4) if isinstance(value, float) else value
    return rounded


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {"query_count": len(rows), "aggregate": {}}
    for metric in SUMMARY_METRICS:
        values = [row.get(metric, 0.0) for row in rows if isinstance(row.get(metric), (int, float))]
        summary["aggregate"][metric] = {
            "mean": round(statistics.fmean(values), 4) if values else 0.0,
            "min": round(min(values), 4) if values else 0.0,
            "max": round(max(values), 4) if values else 0.0,
        }
        if len(values) > 1:
            summary["aggregate"][metric]["std"] = round(statistics.pstdev(values), 4)
        else:
            summary["aggregate"][metric]["std"] = 0.0
    return summary


def group_summaries(rows: list[dict[str, Any]], group_key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key) or "unknown")].append(row)
    return {name: summarize_rows(group_rows) for name, group_rows in sorted(grouped.items())}


def compute_metrics(raw_results: dict[str, Any], relevance_payload: dict[str, Any]) -> dict[str, Any]:
    per_query_mode = []
    mode_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for query_id, query_payload in raw_results.get("queries", {}).items():
        for mode, mode_payload in query_payload.get("modes", {}).items():
            row = evaluate_query_mode(
                query_id=query_id,
                mode=mode,
                query_payload=query_payload,
                mode_payload=mode_payload,
                relevance_payload=relevance_payload,
            )
            per_query_mode.append(row)
            mode_rows[mode].append(row)

    by_mode = {}
    for mode, rows in sorted(mode_rows.items()):
        by_mode[mode] = {
            "overall": summarize_rows(rows),
            "by_type": group_summaries(rows, "type"),
            "by_level": group_summaries(rows, "level"),
        }

    leaderboard = {}
    for metric in ("recall@5", "ndcg@5", "mrr", "map", "all_supporting_facts_found"):
        scores = [
            {
                "mode": mode,
                "score": payload["overall"]["aggregate"][metric]["mean"],
            }
            for mode, payload in by_mode.items()
        ]
        leaderboard[metric] = sorted(scores, key=lambda item: item["score"], reverse=True)

    return {
        "metadata": {
            "raw_result_metadata": raw_results.get("metadata", {}),
            "relevance_metadata": relevance_payload.get("metadata", {}),
            "query_mode_count": len(per_query_mode),
            "mode_count": len(by_mode),
        },
        "by_mode": by_mode,
        "leaderboard": leaderboard,
        "per_query_mode": per_query_mode,
        "errors": raw_results.get("errors", []),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute HotpotQA benchmark metrics.")
    parser.add_argument("--raw-results", type=Path, default=RESULTS_DIR / "hotpot_raw_results.json")
    parser.add_argument("--relevance", type=Path, default=ARTIFACTS_DIR / "hotpot_relevance.json")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "hotpot_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_results = load_json(args.raw_results)
    relevance_payload = load_json(args.relevance)
    metrics = compute_metrics(raw_results, relevance_payload)
    write_json(args.output, metrics)
    print(f"Wrote metrics: {args.output}")
    print(f"Query-mode rows: {metrics['metadata']['query_mode_count']}")
    for metric, ranking in metrics["leaderboard"].items():
        leader = ranking[0] if ranking else {"mode": "none", "score": 0.0}
        print(f"{metric}: {leader['mode']}={leader['score']}")


if __name__ == "__main__":
    main()
