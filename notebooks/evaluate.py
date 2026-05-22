"""
Evaluation metrics for RAG comparison project.
Computes NDCG@K, MRR, Recall@K, Precision@K, and MAP against ground-truth relevance judgments.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL_DATA_PATH = ROOT / "notebooks" / "evaluation_data.json"


def load_relevance_judgments() -> dict:
    with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def dcg_at_k(relevance_scores: list[float], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k]):
        dcg += (2**rel - 1) / math.log2(i + 2)
    return dcg


def ndcg_at_k(relevance_scores: list[float], k: int) -> float:
    dcg = dcg_at_k(relevance_scores, k)
    ideal = sorted(relevance_scores, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def mrr(relevance_scores: list[float]) -> float:
    for i, rel in enumerate(relevance_scores):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(relevance_scores: list[float], total_relevant: int, k: int) -> float:
    if total_relevant == 0:
        return 0.0
    return sum(1 for r in relevance_scores[:k] if r > 0) / total_relevant


def precision_at_k(relevance_scores: list[float], k: int) -> float:
    return sum(1 for r in relevance_scores[:k] if r > 0) / k


def map_at_k(all_relevance: list[list[float]], k: int) -> float:
    avg_precisions = []
    for rels in all_relevance:
        hits = 0
        prec_sum = 0.0
        for i, rel in enumerate(rels[:k]):
            if rel > 0:
                hits += 1
                prec_sum += hits / (i + 1)
        if hits > 0:
            avg_precisions.append(prec_sum / min(k, sum(1 for r in rels if r > 0) or hits))
        else:
            avg_precisions.append(0.0)
    return np.mean(avg_precisions) if avg_precisions else 0.0


def compute_binary_relevance(results: list[dict], relevant_chunk_ids: set[str]) -> list[float]:
    return [1.0 if r.get("chunk_id") in relevant_chunk_ids else 0.0 for r in results]


def evaluate_method(query_id: str, method: str, results: list[dict], judgments: dict) -> dict:
    query_judgments = judgments.get("queries", {}).get(query_id)
    if not query_judgments:
        return {"error": f"No ground truth for query '{query_id}'"}

    relevant_ids = set(query_judgments.get("relevant_chunks", []))
    total_relevant = len(relevant_ids)
    relevance = compute_binary_relevance(results, relevant_ids)

    k_values = [3, 5, 10]
    metrics = {
        "query_id": query_id,
        "method": method,
        "total_retrieved": len(results),
        "total_relevant": total_relevant,
    }
    for k in k_values:
        effective_k = min(k, len(relevance))
        metrics[f"ndcg@{k}"] = round(ndcg_at_k(relevance, effective_k), 4)
        metrics[f"recall@{k}"] = round(recall_at_k(relevance, total_relevant, effective_k), 4)
        metrics[f"precision@{k}"] = round(precision_at_k(relevance, effective_k), 4)

    metrics["mrr"] = round(mrr(relevance), 4)
    metrics["map"] = round(map_at_k([relevance], len(relevance)), 4)

    return metrics


def evaluate_all_queries(method_results: dict[str, list[dict]]) -> dict:
    judgments = load_relevance_judgments()
    all_metrics = []

    for query_id, results in method_results.items():
        metrics = evaluate_method(query_id, "method", results, judgments)
        all_metrics.append(metrics)

    query_count = len(all_metrics)
    if not query_count:
        return {"error": "No query results to evaluate", "method_metrics": []}

    summary = {"query_count": query_count, "aggregate": {}}
    for metric in ["ndcg@3", "ndcg@5", "ndcg@10", "recall@3", "recall@5", "recall@10",
                    "precision@3", "precision@5", "precision@10", "mrr", "map"]:
        values = [m.get(metric, 0.0) for m in all_metrics if metric in m]
        summary["aggregate"][metric] = {
            "mean": round(np.mean(values), 4) if values else 0.0,
            "std": round(np.std(values), 4) if values else 0.0,
            "min": round(np.min(values), 4) if values else 0.0,
            "max": round(np.max(values), 4) if values else 0.0,
        }

    return {"method_metrics": all_metrics, "summary": summary}


def evaluate_across_methods(all_results: dict[str, dict[str, list[dict]]]) -> dict:
    judgments = load_relevance_judgments()
    query_ids = list(judgments.get("queries", {}).keys())
    method_names = list(all_results.keys())
    comparison = {"methods": method_names, "queries": query_ids, "table": {}}

    for metric in ["ndcg@5", "mrr", "recall@5", "precision@5"]:
        comparison["table"][metric] = {}
        for method in method_names:
            method_data = all_results[method]
            values = []
            for qid in query_ids:
                results = method_data.get(qid, [])
                query_judgments = judgments["queries"].get(qid, {})
                relevant_ids = set(query_judgments.get("relevant_chunks", []))
                relevance = compute_binary_relevance(results, relevant_ids)
                k = 5
                if metric.startswith("ndcg"):
                    values.append(ndcg_at_k(relevance, min(k, len(relevance))))
                elif metric == "mrr":
                    values.append(mrr(relevance))
                elif metric.startswith("recall"):
                    values.append(recall_at_k(relevance, len(relevant_ids), min(k, len(relevance))))
                elif metric.startswith("precision"):
                    values.append(precision_at_k(relevance, min(k, len(relevance))))
            comparison["table"][metric][method] = round(np.mean(values), 4) if values else 0.0

    comparison["ranking"] = {}
    for metric, method_scores in comparison["table"].items():
        sorted_methods = sorted(method_scores.items(), key=lambda x: -x[1])
        comparison["ranking"][metric] = [{"method": m, "score": s} for m, s in sorted_methods]

    return comparison
