# HotpotQA Benchmarking Plan

Goal: add a structured HotpotQA adapter that can load benchmark examples into the existing shared corpus format, run the public retrieval modes, and report retrieval plus answer-quality metrics.

## Current Repo Hooks

- Corpus store lives in `backend_or_exports/current_corpus`.
- Retrieval modes already route through `backend.run_dynamic_retrieval(mode, query_text)`.
- Supported public mode keys: `naive`, `hybrid`, `graph`, `agentic`, `crag`.
- Retrieval evaluation already works on returned `results` lists using `chunk_id`.
- Current PDF pipeline builds `chunks.json`, `embeddings.npy`, `faiss.index`, `metadata.json`, and `projection.json`.

## Benchmark Design

Use `hotpotqa/hotpot_qa`, starting with the `distractor` config and `validation` split.

Each HotpotQA row has:

- `id`: benchmark query id.
- `question`: query text.
- `answer`: gold answer text.
- `context`: article titles plus sentence lists.
- `supporting_facts`: gold `(title, sent_id)` evidence labels.
- `type`: `bridge` or `comparison`.
- `level`: `easy`, `medium`, or `hard`.

Represent each sentence as one retrievable chunk so evidence scoring is exact:

```text
chunk_id = hotpot::<example_id>::<normalized_title>::<sentence_index>
chunk_text = [<title>] <sentence>
```

For first-stage benchmarking, build one store from a sample of examples. This preserves realistic cross-example distractors while keeping runs fast.

## Stage 1: Dataset Adapter

Create `benchmarking/hotpot_adapter.py`.

Responsibilities:

- Load HotpotQA via Hugging Face `datasets`.
- Select configurable split/config/sample size.
- Convert rows into repo-compatible `chunks`.
- Build a manifest of benchmark queries.
- Build gold relevance by mapping each supporting fact to its generated `chunk_id`.
- Deduplicate chunks by `(title, sentence_index, sentence_text)` when building multi-example corpora, while preserving query-specific relevance mappings.

Output files:

- `benchmarking/artifacts/hotpot_queries.json`
- `benchmarking/artifacts/hotpot_relevance.json`
- `benchmarking/artifacts/hotpot_chunks.json`

## Stage 2: Store Builder

Create `benchmarking/build_hotpot_store.py`.

Status: implemented. Structural smoke test passed with synthetic Stage 1 artifacts and `--skip-embeddings`.

Responsibilities:

- Reuse `shared_rag_store.load_embedding_model`, `embed_texts`, `build_faiss_index`, `project_embeddings`, and `write_json`.
- Write a HotpotQA corpus into the existing `STORE_DIR`.
- Include metadata with dataset name, config, split, sample size, chunk count, and embedding model.
- Write frontend placeholder JSONs for public mode prefixes, matching `build_store`.

Expected store files:

- `backend_or_exports/current_corpus/chunks.json`
- `backend_or_exports/current_corpus/embeddings.npy`
- `backend_or_exports/current_corpus/faiss.index`
- `backend_or_exports/current_corpus/metadata.json`
- `backend_or_exports/current_corpus/projection.json`

## Stage 3: Benchmark Runner

Create `benchmarking/run_hotpot_benchmark.py`.

Status: implemented. Smoke test passed in the `rag-multimodal` conda environment against synthetic artifacts.

Responsibilities:

- Load `hotpot_queries.json` and `hotpot_relevance.json`.
- Import `backend.run_dynamic_retrieval`.
- For each query and each mode, run retrieval.
- Save raw results per query/mode.
- Avoid DeepSeek during benchmark by default; retrieval benchmarking should be deterministic and cheap.
- Add CLI flags for `--limit`, `--modes`, `--top-k`, and output path.

Output files:

- `benchmarking/results/hotpot_raw_results.json`
- `benchmarking/results/hotpot_run_metadata.json`

## Stage 4: Metrics

Create `benchmarking/hotpot_metrics.py`.

Status: implemented. Smoke test passed against synthetic BM25 benchmark output.

Retrieval metrics:

- Recall@3/5/10
- Precision@3/5/10
- MRR
- NDCG@3/5/10
- MAP
- Supporting-fact hit count
- All-supporting-facts-found rate

Slices:

- Overall
- By mode
- By `type`: `bridge`, `comparison`
- By `level`: `easy`, `medium`, `hard`

Optional answer metrics:

- Exact match
- Token F1

Answer metrics should be a later pass because retrieval-only metrics are the cleanest first benchmark.

## Stage 5: Backend/API Integration

After the offline benchmark scripts work, add optional Flask endpoints:

- `POST /api/benchmark/hotpot/build`
- `POST /api/benchmark/hotpot/run`
- `GET /api/benchmark/hotpot/results`

Keep this separate from the PDF upload path so the existing app flow remains stable.

## Stage 6: Frontend Integration

Add a benchmark view only after offline metrics are verified.

Useful UI:

- Mode leaderboard.
- Metric selector.
- Type/level filters.
- Per-query inspection table.
- Evidence hit/miss display using supporting facts.

## Implementation Order

1. Add adapter and generated artifact schema.
2. Add HotpotQA store builder.
3. Smoke-test one tiny sample, e.g. 5 validation examples.
4. Add runner for all public modes.
5. Add metrics aggregation.
6. Run a real mini validation sample, e.g. 25 examples.
7. Run a medium validation sample, e.g. 100 examples.
8. Add API endpoints, if needed.
9. Add frontend benchmark dashboard, if needed.

Current next step: run the real HotpotQA mini benchmark using `HOTPOT_BENCHMARK_COMMANDS.txt`.

## Main Risks

- `datasets` is not currently in `requirements.txt`; add it when implementing Stage 1.
- First run may need network access to download HotpotQA and possibly embeddings.
- Exact sentence-level chunks may be short for vector retrieval; this is acceptable for evidence benchmarking, but we can later compare sentence chunks against paragraph chunks.
- Some modes assume PDF fields such as `page_number`; adapter should provide compatible placeholder fields.
