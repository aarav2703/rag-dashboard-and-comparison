# RAG Evidence Lab

**Comparative Retrieval-Augmented Generation Diagnostics**

RAG Evidence Lab is a full-stack research prototype for comparing Retrieval-Augmented Generation (RAG) methods on the same PDF corpus. The goal is not just to return an answer, but to show how the answer was supported. A user can upload a PDF, build one shared document index, ask a question, and then inspect how different retrieval strategies selected evidence.

The project also includes a HotpotQA benchmark pipeline. This gives the work a more formal evaluation setting because HotpotQA contains multi-hop questions with gold supporting facts.

## Main Contributions

- Implements 5 public RAG retrieval modes in one application.
- Uses one shared corpus store, so the methods are compared on the same chunks.
- Adds Corrective RAG with reranking, evidence grading, query correction, fallback retrieval, and groundedness checking.
- Builds visual diagnostics for each method instead of using a plain chatbot interface.
- Includes a real HotpotQA adapter and benchmark runner for retrieval evaluation.
- Reports retrieval metrics such as NDCG, Recall, Precision, Hit@K, MRR, and MAP, with optional answer metrics for EM, token F1, faithfulness, and answer relevancy.

## Methods Implemented

| # | Method | Main Idea |
|---|--------|-----------|
| 1 | **Naive Vector RAG** | Uses SentenceTransformer embeddings and FAISS semantic search. |
| 2 | **Hybrid RAG** | Combines dense vector retrieval and BM25 sparse retrieval with rank fusion. |
| 3 | **GraphRAG** | Builds entity, section, and claim relationships and retrieves answer subgraphs. |
| 4 | **Agentic Multi-hop RAG** | Plans tool use, extracts bridge clues, performs second-hop retrieval, and shows agent decisions. |
| 5 | **Corrective RAG with Reranking** | Reranks candidates, grades evidence, rewrites or falls back when needed, and checks groundedness. |

BM25, reranking, and multi-hop retrieval are still used internally, but they are no longer standalone public modes.

## Screenshots

The screenshots below show the current frontend after all methods are run on the same PDF and query. They are included because the main contribution of this project is the evidence inspection interface, not only the backend retrieval code.

### Full Interface

![RAG Evidence Lab main interface](docs/screenshots/rag13.png)

The main workspace keeps the selected retrieval method, pipeline console, PDF upload, question input, and diagnostics on one screen. This helped me compare retrieval behavior without switching between separate notebooks.

### Method Comparison

![Compare all methods view](docs/screenshots/rag1.png)

The comparison view runs all methods for the same question and shows evidence overlap, citation coverage, and answer differences. This is useful for seeing whether methods agree on the same pages or find different support.

### Naive Vector RAG

![Naive vector RAG analytics](docs/screenshots/rag12.png)

Naive vector retrieval shows a 2D embedding projection, retrieved chunks, cosine similarity bars, and citation coverage. This makes the semantic search baseline easier to inspect than a list of top-k chunks.

### Hybrid RAG

![Hybrid retrieval merge](docs/screenshots/rag9.png)

Hybrid retrieval merges vector and BM25 candidates. The overlap matrix and rank-fusion views show which chunks came from semantic search, lexical search, or both.

![Hybrid overlap and rank fusion details](docs/screenshots/rag8.png)

### Corrective RAG with Reranking

![Rerank RAG overview](docs/screenshots/rag7.png)

![Rerank movement details](docs/screenshots/rag6.png)

The corrective view shows how candidate chunks move after reranking, how the evidence grader branches, and whether CRAG answered directly, rewrote the query, or used fallback retrieval.

![Rerank histogram and promoted chunk list](docs/screenshots/rag5.png)

### GraphRAG

![GraphRAG-lite evidence graph](docs/screenshots/rag4.png)

GraphRAG now extracts relationship triples, builds relationship edges, detects graph communities, summarizes those communities with DeepSeek when available, and returns a subgraph retrieval trace.

### Agentic Multi-hop RAG

![Agentic RAG control flow](docs/screenshots/rag2.png)

Agentic Multi-hop RAG shows the decision trace: planning, tool calls, bridge extraction, second-hop retrieval, evidence decisions, and final answer support.

## Corrective Answer Generation

CRAG is the only public mode that runs the groundedness critic. It grades retrieved evidence, can rewrite the query or use fallback retrieval, and then checks whether the generated answer is grounded in the retrieved evidence.

Other modes generate a normal answer from their retrieved evidence without displaying critic verdicts or retry metadata.

## Frontend Visualizations

The frontend is a retrieval diagnostics interface rather than a minimal chatbot.

- **Naive Vector RAG**: embedding space, zoom/pan controls, similarity bars, citation coverage, and answer panel.
- **Hybrid**: vector/BM25 overlap views, source badges, rank-fusion bump chart, and fusion table.
- **GraphRAG**: evidence graph, answer subgraph, relationship paths, score breakdowns, community hits, and cited graph evidence.
- **Agentic Multi-hop RAG**: planner decisions, graph/rerank/web tool loop, bridge terms, hop queries, Gantt-style tool timeline, accepted evidence, and rejected evidence.
- **Corrective RAG with Reranking**: candidate movement summary, evidence grading branch, score histogram, correction query, local or web fallback evidence, and grounded answer panel.
- **Compare All**: runs all methods for the same question and shows answer and evidence differences side by side.

## HotpotQA Benchmark

The benchmark pipeline uses the real Hugging Face dataset:

```text
hotpotqa/hotpot_qa
config: distractor
split: validation
```

The adapter converts HotpotQA examples into the same chunk schema used by the app. Each supporting fact is mapped to a relevant chunk ID, so retrieval can be evaluated against gold evidence instead of synthetic labels.

### Previous 150-Query Benchmark

Before the public method cleanup, one run used 150 HotpotQA validation questions, 6,111 chunks, top-k = 10, and the earlier 8-mode set. The current benchmark defaults now evaluate the 5 public modes: Naive, Hybrid, GraphRAG, Agentic Multi-hop, and CRAG.

| Rank | Mode | NDCG@5 | Recall@5 | Recall@10 | MRR | MAP | Supporting Fact Hit Rate | All Facts Found |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Rerank | 0.8249 | 0.6647 | 0.7421 | 0.8727 | 0.5762 | 0.7421 | 0.4733 |
| 2 | Hybrid | 0.8028 | 0.6649 | 0.7532 | 0.8436 | 0.5668 | 0.7532 | 0.4933 |
| 3 | Agentic | 0.7936 | 0.6552 | 0.7527 | 0.8421 | 0.5661 | 0.7527 | 0.4733 |
| 4 | Vectorless | 0.7893 | 0.6427 | 0.7499 | 0.8513 | 0.5726 | 0.7499 | 0.4733 |
| 5 | BM25 | 0.7868 | 0.6427 | 0.7499 | 0.8480 | 0.5709 | 0.7499 | 0.4733 |
| 6 | Multi-hop | 0.7482 | 0.6500 | 0.7504 | 0.7776 | 0.5322 | 0.7504 | 0.4933 |
| 7 | Naive Vector | 0.7090 | 0.5836 | 0.7019 | 0.7453 | 0.4717 | 0.7019 | 0.4133 |
| 8 | GraphRAG-lite | 0.2808 | 0.2150 | 0.3183 | 0.2905 | 0.1668 | 0.3183 | 0.1200 |

The strongest method in this historical run was **Rerank RAG**, which is now folded into **Corrective RAG with Reranking**. GraphRAG remains a roadmap focus because the current graph construction is still moving toward full relationship extraction, community detection, and community summaries.

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Flask, FAISS, PyMuPDF, Sentence Transformers, NumPy, pandas, scikit-learn |
| Frontend | React 18, Vite, D3.js, d3-sankey |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Search | FAISS `IndexFlatIP` |
| LLM | DeepSeek Chat API |
| Benchmarking | HotpotQA adapter, custom runner, custom metrics script |

## Getting Started

### Prerequisites

- Python with conda
- Node.js 18+
- DeepSeek API key for answer generation and critic mode

### Setup

```powershell
conda create -n rag-multimodal python=3.10 -y
conda activate rag-multimodal
python -m pip install -r requirements.txt

cd frontend
npm install
```

Create a `.env` file in the project root:

```text
DEEPSEEK_API_KEY=your_key_here
TAVILY_API_KEY=your_tavily_key_here
```

`TAVILY_API_KEY` is optional. Without it, Agentic and CRAG still run local retrieval tools, but web fallback is marked unavailable in metadata. Set `RAG_ALLOW_EXTERNAL_TOOLS=false` to disable DeepSeek and Tavily calls for deterministic local testing.

### Run the App

Terminal 1:

```powershell
conda activate rag-multimodal
C:\Users\aarav\anaconda3\envs\rag-multimodal\python.exe backend.py
```

Terminal 2:

```powershell
cd frontend
npm run dev
```

Then open:

```text
http://localhost:5173
```

The backend runs on:

```text
http://127.0.0.1:5000
```

The same commands are also listed in `PROJECT_COMMANDS.txt`.

## HotpotQA Benchmark Commands

The project keeps the full benchmark commands in `HOTPOT_BENCHMARK_COMMANDS.txt`. A short version is:

```powershell
conda activate rag-multimodal
python -m pip install datasets
python benchmarking\hotpot_adapter.py --config distractor --split validation --limit 150
python -m pip uninstall -y datasets pyarrow
python benchmarking\build_hotpot_store.py
python benchmarking\run_hotpot_benchmark.py --limit 150 --modes all --top-k 10
python benchmarking\hotpot_metrics.py
```

Answer generation is opt-in to avoid surprise API cost:

```powershell
python benchmarking\run_hotpot_benchmark.py --limit 150 --modes all --top-k 10 --include-answers
python benchmarking\run_hotpot_benchmark.py --limit 150 --modes all --top-k 10 --include-answers --include-llm-metrics
python benchmarking\hotpot_metrics.py
```

The `datasets` and `pyarrow` packages are kept out of the main runtime requirements because they caused instability in this local environment when combined with the embedding stack.

## Project Structure

```text
RAG comparison/
|-- backend.py
|-- requirements.txt
|-- PROJECT_COMMANDS.txt
|-- HOTPOT_BENCHMARK_COMMANDS.txt
|-- docs/
|   `-- screenshots/
|-- benchmarking/
|   |-- PLAN.md
|   |-- hotpot_adapter.py
|   |-- build_hotpot_store.py
|   |-- run_hotpot_benchmark.py
|   `-- hotpot_metrics.py
|-- backend_or_exports/
|   `-- current_corpus/
|-- notebooks/
|   |-- shared_rag_store.py
|   |-- 00_build_faiss_corpus.py
|   |-- 01_naive_vector_rag.py
|   |-- 02_bm25_lexical_rag.py
|   `-- 03_hybrid_rag.py
|-- frontend/
|   |-- package.json
|   |-- vite.config.mjs
|   `-- src/
|       |-- App.jsx
|       |-- styles.css
|       |-- lib/
|       `-- components/
|           |-- AnswerCriticPanel.jsx
|           |-- EmbeddingConstellation.jsx
|           |-- HybridAnalytics.jsx
|           |-- RerankAnalytics.jsx
|           |-- GraphRagAnalytics.jsx
|           |-- AgenticRagAnalytics.jsx
|           |-- MethodComparison.jsx
|           `-- ConsolePanel.jsx
`-- data/
```

## Evaluation Metrics

The HotpotQA metrics script computes:

- NDCG@3, NDCG@5, NDCG@10
- Recall@3, Recall@5, Recall@10
- Precision@3, Precision@5, Precision@10
- Hit@5 and Hit@10
- MRR
- MAP
- supporting fact hit count
- supporting fact hit rate
- all supporting facts found
- Exact Match and token F1 when the benchmark is run with `--include-answers`
- Faithfulness and answer relevancy when the benchmark is run with `--include-llm-metrics`

By default these are retrieval metrics. Answer metrics are only written when answer generation is explicitly enabled.

## Notes and Limitations

- The current benchmark uses sentence-level HotpotQA chunks, which is useful for evidence evaluation but different from long PDF paragraph retrieval.
- GraphRAG is still on the roadmap toward deeper relationship extraction, community detection, and community summarization.
- The groundedness critic is intentionally scoped to CRAG; the other modes generate normal evidence-backed answers.
- The frontend is designed for interpretability and comparison, not minimal chatbot interaction.
- This is a local research prototype. It is not packaged as a production desktop application or cloud service.

## License

MIT
