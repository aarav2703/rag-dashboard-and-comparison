#!/usr/bin/env python
"""Compute HotpotQA retrieval benchmark metrics from raw mode results."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import string
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "benchmarking" / "artifacts"
RESULTS_DIR = ROOT / "benchmarking" / "results"
K_VALUES = (3, 5, 10)
REPORT_METRICS = (
    "precision@5",
    "precision@10",
    "recall@5",
    "recall@10",
    "hit@5",
    "hit@10",
    "ndcg@5",
    "ndcg@10",
    "mrr",
    "map",
    "all_supporting_facts_found",
    "exact_match",
    "token_f1",
    "faithfulness",
    "answer_relevancy",
)
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
    "hit@5",
    "hit@10",
    "mrr",
    "map",
    "supporting_fact_hit_count",
    "supporting_fact_hit_rate",
    "all_supporting_facts_found",
    "exact_match",
    "token_f1",
    "faithfulness",
    "answer_relevancy",
)
METRIC_CATALOG = {
    "precision@5": {"group": "retrieval", "label": "Precision@5", "higher_is_better": True},
    "precision@10": {"group": "retrieval", "label": "Precision@10", "higher_is_better": True},
    "recall@5": {"group": "retrieval", "label": "Recall@5", "higher_is_better": True},
    "recall@10": {"group": "retrieval", "label": "Recall@10", "higher_is_better": True},
    "hit@5": {"group": "retrieval", "label": "Hit@5", "higher_is_better": True},
    "hit@10": {"group": "retrieval", "label": "Hit@10", "higher_is_better": True},
    "ndcg@5": {"group": "retrieval", "label": "NDCG@5", "higher_is_better": True},
    "ndcg@10": {"group": "retrieval", "label": "NDCG@10", "higher_is_better": True},
    "mrr": {"group": "retrieval", "label": "MRR", "higher_is_better": True},
    "map": {"group": "retrieval", "label": "MAP", "higher_is_better": True},
    "all_supporting_facts_found": {"group": "retrieval", "label": "All Facts Found", "higher_is_better": True},
    "exact_match": {"group": "answer", "label": "Exact Match", "higher_is_better": True},
    "token_f1": {"group": "answer", "label": "Token F1", "higher_is_better": True},
    "faithfulness": {"group": "rag_specific", "label": "Faithfulness", "higher_is_better": True},
    "answer_relevancy": {"group": "rag_specific", "label": "Answer Relevancy", "higher_is_better": True},
}


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


def normalize_answer(text: str) -> str:
    text = str(text or "").lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def token_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def hit_at_k(relevance: list[float], k: int) -> float:
    return 1.0 if any(rel > 0 for rel in relevance[:k]) else 0.0


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
        if k in {5, 10}:
            metrics[f"hit@{k}"] = hit_at_k(relevance, effective_k)
    metrics["mrr"] = mrr(relevance)
    metrics["map"] = average_precision(relevance, support_count)
    generated_answer = mode_payload.get("answer")
    gold_answer = query_payload.get("answer") or judgment.get("answer")
    if generated_answer is not None and gold_answer is not None:
        metrics["exact_match"] = exact_match_score(generated_answer, gold_answer)
        metrics["token_f1"] = token_f1_score(generated_answer, gold_answer)
    for judge_metric in ("faithfulness", "answer_relevancy"):
        if isinstance(mode_payload.get(judge_metric), (int, float)):
            metrics[judge_metric] = float(mode_payload[judge_metric])
    return round_metric_values(metrics)


def round_metric_values(metrics: dict[str, Any]) -> dict[str, Any]:
    rounded = {}
    for key, value in metrics.items():
        rounded[key] = round(value, 4) if isinstance(value, float) else value
    return rounded


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {"query_count": len(rows), "aggregate": {}}
    for metric in SUMMARY_METRICS:
        values = [row[metric] for row in rows if isinstance(row.get(metric), (int, float))]
        if not values:
            continue
        summary["aggregate"][metric] = {
            "mean": round(statistics.fmean(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
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
    for metric in ("recall@5", "hit@5", "hit@10", "ndcg@5", "mrr", "map", "all_supporting_facts_found", "exact_match", "token_f1", "faithfulness", "answer_relevancy"):
        scores = [
            {
                "mode": mode,
                "score": payload["overall"]["aggregate"][metric]["mean"],
            }
            for mode, payload in by_mode.items()
            if metric in payload["overall"]["aggregate"]
        ]
        if scores:
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


def metric_mean(summary: dict[str, Any], metric: str) -> float | None:
    aggregate = summary.get("overall", {}).get("aggregate", {})
    if metric not in aggregate:
        return None
    value = aggregate[metric].get("mean")
    return round(float(value), 4) if isinstance(value, (int, float)) else None


def compact_group_summary(group_payload: dict[str, Any]) -> dict[str, Any]:
    compact = {}
    for group_name, summary in group_payload.items():
        row = {"query_count": summary.get("query_count", 0), "metrics": {}}
        aggregate = summary.get("aggregate", {})
        for metric in REPORT_METRICS:
            if metric in aggregate:
                row["metrics"][metric] = aggregate[metric].get("mean")
        compact[group_name] = row
    return compact


def summarize_errors(errors: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, int] = defaultdict(int)
    examples = []
    for error in errors:
        by_mode[str(error.get("mode") or "unknown")] += 1
        if len(examples) < 12:
            examples.append(
                {
                    "query_id": error.get("query_id"),
                    "mode": error.get("mode"),
                    "error": str(error.get("error", ""))[:220],
                    "elapsed_ms": error.get("elapsed_ms"),
                }
            )
    return {"count": len(errors), "by_mode": dict(sorted(by_mode.items())), "examples": examples}


def notable_queries(rows: list[dict[str, Any]], limit: int = 18) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("query_id"))].append(row)

    notable = []
    for query_id, query_rows in grouped.items():
        scored = [row for row in query_rows if isinstance(row.get("recall@10"), (int, float))]
        if not scored:
            continue
        best = max(scored, key=lambda row: (row.get("recall@10", 0), row.get("ndcg@5", 0), row.get("mrr", 0)))
        worst = min(scored, key=lambda row: (row.get("recall@10", 0), row.get("ndcg@5", 0), row.get("mrr", 0)))
        spread = round(float(best.get("recall@10", 0)) - float(worst.get("recall@10", 0)), 4)
        missed_count = sum(1 for row in scored if row.get("hit@10") == 0)
        if spread > 0 or missed_count:
            notable.append(
                {
                    "query_id": query_id,
                    "question": best.get("question"),
                    "type": best.get("type"),
                    "level": best.get("level"),
                    "best_mode": best.get("mode"),
                    "best_recall@10": best.get("recall@10"),
                    "worst_mode": worst.get("mode"),
                    "worst_recall@10": worst.get("recall@10"),
                    "recall@10_spread": spread,
                    "modes_missing_hit@10": missed_count,
                }
            )
    notable.sort(key=lambda row: (row["modes_missing_hit@10"], row["recall@10_spread"]), reverse=True)
    return notable[:limit]


def build_compact_report(metrics: dict[str, Any]) -> dict[str, Any]:
    by_mode = metrics.get("by_mode", {})
    leaderboards = {
        metric: ranking
        for metric, ranking in metrics.get("leaderboard", {}).items()
        if metric in REPORT_METRICS
    }
    method_summary = {}
    per_method_by_type = {}
    per_method_by_level = {}
    for mode, payload in sorted(by_mode.items()):
        method_summary[mode] = {
            "query_count": payload.get("overall", {}).get("query_count", 0),
            "metrics": {
                metric: value
                for metric in REPORT_METRICS
                if (value := metric_mean(payload, metric)) is not None
            },
        }
        per_method_by_type[mode] = compact_group_summary(payload.get("by_type", {}))
        per_method_by_level[mode] = compact_group_summary(payload.get("by_level", {}))

    return {
        "metadata": {
            **metrics.get("metadata", {}),
            "report_format": "hotpot_benchmark_report.v1",
            "report_note": "Compact dashboard-ready summary. Use hotpot_metrics.json for full per-query details.",
        },
        "metric_catalog": {
            metric: METRIC_CATALOG[metric]
            for metric in REPORT_METRICS
            if metric in METRIC_CATALOG and any(metric in row.get("metrics", {}) for row in method_summary.values())
        },
        "method_summary": method_summary,
        "leaderboards": leaderboards,
        "best_method_by_metric": {
            metric: ranking[0]
            for metric, ranking in leaderboards.items()
            if ranking
        },
        "per_method_by_type": per_method_by_type,
        "per_method_by_level": per_method_by_level,
        "error_summary": summarize_errors(metrics.get("errors", [])),
        "notable_queries": notable_queries(metrics.get("per_query_mode", [])),
    }


def assert_close(actual: float, expected: float, label: str, tolerance: float = 1e-4) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def run_self_test() -> None:
    relevance = [0.0, 1.0, 0.0, 1.0, 0.0]
    assert_close(precision_at_k(relevance, 5), 0.4, "precision@5")
    assert_close(recall_at_k(relevance, 3, 5), 2 / 3, "recall@5")
    assert_close(hit_at_k(relevance, 1), 0.0, "hit@1")
    assert_close(hit_at_k(relevance, 5), 1.0, "hit@5")
    assert_close(mrr(relevance), 0.5, "mrr")
    assert_close(average_precision(relevance, 3), ((1 / 2) + (2 / 4)) / 3, "map")
    assert_close(ndcg_at_k([1.0, 0.0, 1.0], 3), 0.9197, "ndcg@3")
    if normalize_answer("The, Paris!") != "paris":
        raise AssertionError("normalize_answer failed punctuation/article/case handling")
    assert_close(exact_match_score("The Paris", "paris"), 1.0, "exact_match")
    assert_close(token_f1_score("red blue green", "red blue"), 0.8, "token_f1 partial")
    assert_close(token_f1_score("alpha", "beta"), 0.0, "token_f1 no overlap")

    raw_results = {
        "queries": {
            "q1": {
                "query_id": "q1",
                "question": "Who?",
                "answer": "Paris",
                "type": "bridge",
                "level": "easy",
                "modes": {
                    "naive": {
                        "results": [{"chunk_id": "c1"}, {"chunk_id": "c2"}],
                        "elapsed_ms": 10,
                    },
                    "hybrid": {
                        "results": [{"chunk_id": "c3"}, {"chunk_id": "c4"}],
                        "answer": "The Paris",
                        "elapsed_ms": 12,
                    },
                },
            }
        },
        "metadata": {"query_count": 1, "modes": ["naive", "hybrid"]},
        "errors": [],
    }
    relevance_payload = {
        "queries": {"q1": {"answer": "Paris", "relevant_chunks": ["c2", "c3"], "type": "bridge", "level": "easy"}},
        "metadata": {"query_count": 1},
    }
    metrics = compute_metrics(raw_results, relevance_payload)
    naive_row = next(row for row in metrics["per_query_mode"] if row["mode"] == "naive")
    hybrid_row = next(row for row in metrics["per_query_mode"] if row["mode"] == "hybrid")
    assert "exact_match" not in naive_row, "retrieval-only rows should not include answer metrics"
    assert_close(naive_row["hit@5"], 1.0, "retrieval aggregation hit@5")
    assert_close(hybrid_row["exact_match"], 1.0, "answer exact match aggregation")
    report = build_compact_report(metrics)
    if "method_summary" not in report or "leaderboards" not in report:
        raise AssertionError("compact report shape is incomplete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute HotpotQA benchmark metrics.")
    parser.add_argument("--raw-results", type=Path, default=RESULTS_DIR / "hotpot_raw_results.json")
    parser.add_argument("--relevance", type=Path, default=ARTIFACTS_DIR / "hotpot_relevance.json")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "hotpot_metrics.json")
    parser.add_argument("--report-output", type=Path, default=RESULTS_DIR / "hotpot_benchmark_report.json")
    parser.add_argument("--self-test", action="store_true", help="Run deterministic metric checks and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        print("Metric self-test passed")
        return
    raw_results = load_json(args.raw_results)
    relevance_payload = load_json(args.relevance)
    metrics = compute_metrics(raw_results, relevance_payload)
    report = build_compact_report(metrics)
    write_json(args.output, metrics)
    write_json(args.report_output, report)
    print(f"Wrote metrics: {args.output}")
    print(f"Wrote compact report: {args.report_output}")
    print(f"Query-mode rows: {metrics['metadata']['query_mode_count']}")
    for metric, ranking in metrics["leaderboard"].items():
        leader = ranking[0] if ranking else {"mode": "none", "score": 0.0}
        print(f"{metric}: {leader['mode']}={leader['score']}")


if __name__ == "__main__":
    main()
