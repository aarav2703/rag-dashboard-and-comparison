#!/usr/bin/env python
"""Convert HotpotQA examples into benchmark artifacts for this RAG app."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "benchmarking" / "artifacts"
DATASET_NAME = "hotpotqa/hotpot_qa"
DEFAULT_CONFIG = "distractor"
DEFAULT_SPLIT = "validation"


def normalize_title(title: str) -> str:
    """Create a stable, readable chunk-id title segment."""
    normalized = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return normalized or "untitled"


def make_chunk_id(example_id: str, title: str, sentence_index: int) -> str:
    return f"hotpot::{example_id}::{normalize_title(title)}::{sentence_index}"


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(value)


def iter_context_items(context: Any) -> list[tuple[str, list[str]]]:
    """Return `(title, sentences)` pairs across common HF dataset formats."""
    if isinstance(context, dict):
        titles = as_list(context.get("title"))
        sentence_groups = as_list(context.get("sentences"))
        return [
            (str(title), [str(sentence) for sentence in as_list(sentences)])
            for title, sentences in zip(titles, sentence_groups)
        ]

    items = []
    for entry in as_list(context):
        if isinstance(entry, dict):
            title = entry.get("title", "")
            sentences = entry.get("sentences", [])
        else:
            title = entry[0] if len(entry) > 0 else ""
            sentences = entry[1] if len(entry) > 1 else []
        items.append((str(title), [str(sentence) for sentence in as_list(sentences)]))
    return items


def iter_supporting_facts(supporting_facts: Any) -> list[tuple[str, int]]:
    """Return `(title, sentence_index)` labels across common HF formats."""
    if isinstance(supporting_facts, dict):
        titles = as_list(supporting_facts.get("title"))
        sent_ids = as_list(supporting_facts.get("sent_id"))
        return [(str(title), int(sent_id)) for title, sent_id in zip(titles, sent_ids)]

    facts = []
    for fact in as_list(supporting_facts):
        if isinstance(fact, dict):
            title = fact.get("title", "")
            sent_id = fact.get("sent_id", 0)
        else:
            title = fact[0] if len(fact) > 0 else ""
            sent_id = fact[1] if len(fact) > 1 else 0
        facts.append((str(title), int(sent_id)))
    return facts


def convert_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    chunks_by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    queries = []
    relevance = {"queries": {}}

    for example_number, row in enumerate(examples, start=1):
        example_id = str(row.get("id") or row.get("_id") or example_number)
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        context_items = iter_context_items(row.get("context", []))
        supporting_facts = iter_supporting_facts(row.get("supporting_facts", []))

        context_lookup: dict[tuple[str, int], str] = {}
        for title, sentences in context_items:
            for sentence_index, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue
                chunk_id = make_chunk_id(example_id, title, sentence_index)
                dedupe_key = (title, sentence_index, sentence)
                context_lookup[(title, sentence_index)] = chunk_id
                chunks_by_key.setdefault(
                    dedupe_key,
                    {
                        "chunk_id": chunk_id,
                        "page_number": example_number,
                        "chunk_index": len(chunks_by_key),
                        "chunk_text": f"[{title}] {sentence}",
                        "word_count": len(sentence.split()) + len(title.split()),
                        "preview": f"[{title}] {sentence}"[:220],
                        "source": "hotpotqa",
                        "example_id": example_id,
                        "title": title,
                        "sentence_index": sentence_index,
                    },
                )

        relevant_chunk_ids = []
        missing_supporting_facts = []
        for title, sentence_index in supporting_facts:
            chunk_id = context_lookup.get((title, sentence_index))
            if chunk_id:
                relevant_chunk_ids.append(chunk_id)
            else:
                missing_supporting_facts.append(
                    {"title": title, "sentence_index": sentence_index}
                )

        queries.append(
            {
                "query_id": example_id,
                "question": question,
                "answer": answer,
                "type": row.get("type"),
                "level": row.get("level"),
                "supporting_fact_count": len(supporting_facts),
                "relevant_chunk_count": len(set(relevant_chunk_ids)),
            }
        )
        relevance["queries"][example_id] = {
            "question": question,
            "answer": answer,
            "type": row.get("type"),
            "level": row.get("level"),
            "relevant_chunks": sorted(set(relevant_chunk_ids)),
            "supporting_facts": [
                {"title": title, "sentence_index": sentence_index}
                for title, sentence_index in supporting_facts
            ],
            "missing_supporting_facts": missing_supporting_facts,
        }

    chunks = list(chunks_by_key.values())
    for index, chunk in enumerate(chunks):
        chunk["chunk_index"] = index

    return {
        "chunks": chunks,
        "queries": queries,
        "relevance": relevance,
        "metadata": {
            "dataset": DATASET_NAME,
            "example_count": len(examples),
            "query_count": len(queries),
            "chunk_count": len(chunks),
        },
    }


def load_hotpot_dataset(config: str, split: str, limit: int | None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: install `datasets` before loading HotpotQA."
        ) from exc

    dataset = load_dataset(DATASET_NAME, config, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [dict(row) for row in dataset]


def write_artifacts(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "hotpot_chunks.json": {
            "metadata": payload["metadata"],
            "chunks": payload["chunks"],
        },
        "hotpot_queries.json": {
            "metadata": payload["metadata"],
            "queries": payload["queries"],
        },
        "hotpot_relevance.json": {
            "metadata": payload["metadata"],
            **payload["relevance"],
        },
    }
    for filename, content in files.items():
        with open(output_dir / filename, "w", encoding="utf-8") as file:
            json.dump(content, file, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HotpotQA benchmark artifacts.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_hotpot_dataset(args.config, args.split, args.limit)
    payload = convert_examples(examples)
    payload["metadata"].update(
        {
            "config": args.config,
            "split": args.split,
            "limit": args.limit,
        }
    )
    write_artifacts(payload, args.output_dir)
    print(f"Wrote {payload['metadata']['query_count']} queries")
    print(f"Wrote {payload['metadata']['chunk_count']} chunks")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
