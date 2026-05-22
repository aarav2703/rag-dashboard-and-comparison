# RAG Evidence Lab

**Comparative Retrieval-Augmented Generation Diagnostics**

A full-stack application that implements, visualizes, and evaluates **8 distinct RAG paradigms** on a shared document corpus. Built for data science portfolio demonstration.

## Methods Implemented

| # | Method | Description |
|---|--------|-------------|
| 1 | **Naive Vector RAG** | FAISS semantic search with UMAP 2D projection |
| 2 | **BM25 Lexical** | From-scratch BM25 with term rarity bars and highlight spans |
| 3 | **Hybrid RAG** | Vector + BM25 rank fusion with reciprocal rank fusion (RRF) |
| 4 | **Rerank RAG** | Wide candidate retrieval + lexical-semantic reranker |
| 5 | **GraphRAG-lite** | Entity-section knowledge graph traversal |
| 6 | **Vectorless Markdown** | Document structure tree navigation (no embeddings) |
| 7 | **Agentic RAG** | Plan-critique-retry orchestration loop |
| 8 | **Multi-hop RAG** | Bridge-term extraction and hop-by-hop retrieval |

## Features

- Interactive per-method visualizations (D3.js force graphs, slopegraphs, Sankey diagrams, tree layouts)
- Side-by-side comparison dashboard with evidence overlap matrix
- **Evaluation metrics**: NDCG@K, MRR, Recall@K, Precision@K, MAP
- **Keyword-auto-labeled ground truth** relevance judgments
- DeepSeek LLM integration for answer generation
- Real-time pipeline status via SSE
- Dark theme with amber-accent research-lab aesthetic

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Flask, FAISS, Sentence Transformers, PyMuPDF, UMAP |
| Frontend | React 18, Vite, D3.js (v7), d3-sankey |
| Embedding | all-MiniLM-L6-v2 (384-dim) |
| LLM | DeepSeek Chat API |
| Visualization | D3 force, tree, sankey, zoom; custom SVGs |

## Getting Started

### Prerequisites
- Python 3.10+ with conda
- Node.js 18+
- DeepSeek API key

### Setup

```bash
# Clone and enter the project
cd "RAG comparison"

# Create conda environment
conda create -n rag-multimodal python=3.10 -y
conda activate rag-multimodal
pip install -r requirements.txt

# Set your DeepSeek API key
echo "DEEPSEEK_API_KEY=your_key_here" > .env

# Install frontend dependencies
cd frontend
npm install
```

### Run

```bash
# Terminal 1: Backend
conda activate rag-multimodal
python backend.py

# Terminal 2: Frontend
cd frontend
npm run dev
```

Then open `http://localhost:5173`, upload a PDF, and start exploring retrieval methods.

## Project Structure

```
RAG comparison/
├── backend.py              # Flask API with 8 RAG endpoints + evaluation
├── notebooks/
│   ├── shared_rag_store.py # Shared FAISS corpus, BM25, embedding utils
│   ├── evaluate.py         # NDCG, MRR, Recall/Precision@K metrics
│   ├── evaluation_data.json# Keyword-auto-labeled ground truth
│   ├── 00_build_faiss_corpus.py
│   ├── 01_naive_vector_rag.py
│   ├── 02_bm25_lexical_rag.py
│   └── 03_hybrid_rag.py
├── frontend/
│   ├── src/
│   │   ├── App.jsx                        # Sidebar layout + routing
│   │   ├── styles.css                     # Dark theme + animations
│   │   └── components/
│   │       ├── EmbeddingConstellation.jsx # UMAP scatter + particle trails
│   │       ├── Bm25Analytics.jsx          # Highlight matrix + term bars
│   │       ├── HybridAnalytics.jsx        # Sankey + bump chart
│   │       ├── RerankAnalytics.jsx        # Slopegraph + histogram
│   │       ├── GraphRagAnalytics.jsx      # D3 force graph
│   │       ├── VectorlessMarkdownAnalytics.jsx # D3 tree
│   │       ├── AgenticRagAnalytics.jsx    # Flow diagram + Gantt
│   │       ├── MultiHopRagAnalytics.jsx   # Hop reasoning graph
│   │       ├── MethodComparison.jsx       # Leaderboard + overlap matrix
│   │       └── EvaluationPanel.jsx        # Radar chart + metrics table
│   └── public/data/       # Pipeline artifacts (JSON)
└── data/
    └── MachineLearningNotes.pdf
```

## Evaluation

The `/api/evaluate` endpoint computes retrieval quality metrics using keyword-auto-labeled ground truth. Each query has required/bonus terms; chunks matching ≥2 required terms are marked relevant.

**Metrics**: NDCG@3/5/10, MRR, Recall@3/5/10, Precision@3/5/10, MAP

The EvaluationPanel shows a performance radar chart and ranked leaderboard across all 8 methods.

## Design

Dark theme with amber (#f59e0b) accent, DM Serif Display headings + JetBrains Mono body. Dot-grid background pattern with radial gradient vignettes. Staggered reveal animations and particle trail effects for retrieval visualization.

## License

MIT
