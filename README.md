# Clinical Psychology Assistant

A production-ready RAG-powered clinical analysis system for psychologists and clinicians. Supports single client statements, symptom lists, and full case studies through a multi-stage analysis pipeline.

**Stack:** FastAPI · ChromaDB · BAAI/bge-large-en-v1.5 · Ollama (Llama 3.1 8B) · LangChain · structlog · Pydantic

---

## Features

- **Multi-stage clinical pipeline** — input classification → feature extraction → evidence retrieval → synthesis → response generation
- **RAG ingestion** — PDF extraction (PyMuPDF), semantic chunking, vector embeddings, ChromaDB storage
- **Three input modes** — single statements, symptom checklists, full case studies (auto-detected)
- **Structured clinical output** — analysis, formulation, evidence-based recommendations, confidence scoring, limitations
- **REST API** — FastAPI with OpenAPI docs, CORS, structured error handling, request ID tracking
- **Production logging** — structured JSON logs via structlog, PII censoring, audit trail
- **Fully tested** — 379 unit tests with mocked external dependencies

---

## Quick Start

### Prerequisites

- Python ≥ 3.11
- [Ollama](https://ollama.ai) running with `llama3.1:8b` pulled
- Windows (tested) or Linux/macOS

### 1. Setup

```bash
git clone https://github.com/NimMish27/clinical-psychology-assistant.git
cd clinical-psychology-assistant

# Create virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

### 2. Configure environment

```bash
copy .env.example .env       # Windows
# cp .env.example .env       # Linux/macOS
```

Edit `.env` — at minimum review `ALLOWED_ORIGINS` (JSON array format: `["http://localhost:3000"]`).

### 3. Start Ollama

```bash
ollama serve
ollama pull llama3.1:8b
```

### 4. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

### 5. Ingest clinical documents

Place PDFs anywhere, then run the ingestion pipeline:

```bash
python scripts/run_full_pipeline.py --dir data/raw
```

The pipeline extracts text → chunks → embeds → stores in ChromaDB.

### 6. Analyze

```powershell
curl -X POST http://localhost:8000/api/v1/clinical/analyze ^
  -H "Content-Type: application/json" ^
  -d "{\"text\": \"27-year-old male with 2 weeks of depressed mood, anhedonia, fatigue. No prior episodes.\"}"
```

Or use the simple Q&A endpoint:
```powershell
curl -X POST http://localhost:8000/api/v1/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"What are the DSM-5 criteria for MDD?\"}"
```

---

## Architecture

### Pipeline Stages

```
Input (raw_text)
  │
  ▼
Stage 1-2: InputProcessor
  │  Classifies input type (single_statement / symptom_list / case_study)
  │  Produces: CaseUnderstanding (summary, key_topics, clinical_context)
  ▼
Stage 3: FeatureExtractor
  │  Extracts structured clinical features via LLM
  │  Produces: ClinicalFeatures (symptoms, diagnoses, history, risks, ...)
  ▼
Stage 4: QueryGenerator
  │  Generates 3-6 targeted retrieval queries with weights
  │  Produces: list[RetrievalQuery]
  ▼
Stage 5: Retriever (rag/retriever.py)
  │  Embeds queries → ChromaDB similarity search → threshold filtering
  │  Produces: list[RetrievedChunk] (deduplicated across queries)
  ▼
Stage 6: EvidenceSynthesizer
  │  Synthesizes evidence, identifies supporting/contradicting findings
  │  Produces: EvidenceSynthesis
  ▼
Stage 7: ResponseGenerator
  │  Generates comprehensive clinical response
  │  Produces: ClinicalResponse (analysis, formulation, recommendations, ...)
```

### Project Structure

```
clinical-psychology-assistant/
├── api/                    # FastAPI application
│   ├── main.py             # App factory, middleware, exception handlers
│   ├── middleware.py        # Request ID + process time middleware
│   ├── routers/
│   │   ├── chat.py         # POST /api/v1/chat — simple RAG Q&A
│   │   ├── clinical.py     # POST /api/v1/clinical/analyze — full pipeline
│   │   ├── health.py       # GET /health — deep dependency probes
│   │   └── ingest.py       # POST /api/v1/ingest — PDF upload & process
│   ├── schemas/models.py   # Pydantic request/response models
│   └── dependencies/       # FastAPI dependency injection
├── clinical/               # Clinical analysis pipeline
│   ├── models.py           # All pipeline data models
│   ├── llm.py              # Unified LLM service (Ollama via LangChain)
│   ├── input_processor.py  # Stage 1-2: input classification
│   ├── feature_extractor.py # Stage 3: feature extraction
│   ├── query_generator.py  # Stage 4: retrieval query generation
│   ├── evidence_synthesizer.py # Stage 6: evidence synthesis
│   ├── response_generator.py   # Stage 7: response generation
│   └── pipeline.py         # Orchestrator
├── rag/                    # Retrieval-Augmented Generation core
│   ├── embeddings.py       # BGE embedding model singleton
│   ├── vector_store.py     # ChromaDB client wrapper
│   └── retriever.py        # Query → embed → search → filter
├── ingestion/              # Document processing
│   ├── loaders/pdf_loader.py  # PyMuPDF extraction
│   └── processors/chunker.py # RecursiveCharacterTextSplitter
├── config/settings.py      # Pydantic settings (from .env)
├── app_logging/logger.py   # structlog configuration
├── scripts/
│   ├── run_full_pipeline.py # End-to-end PDF ingestion
│   └── create_sample_pdf.py # Test PDF generator
├── data/                   # Runtime data (gitignored)
│   ├── raw/                # Source documents
│   └── chroma/             # ChromaDB persistence
├── tests/                  # 379 unit tests
│   ├── unit/
│   │   ├── api/            # API endpoint tests (39)
│   │   ├── clinical/       # Pipeline tests (24)
│   │   ├── rag/            # RAG tests (241)
│   │   └── ingestion/      # Ingestion tests (75)
└── requirements.txt
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/clinical/analyze` | Full clinical analysis pipeline |
| POST | `/api/v1/chat` | Simple RAG question answering |
| POST | `/api/v1/ingest` | Upload and ingest PDF documents |
| GET | `/health` | Deep health check (Ollama, ChromaDB, embeddings, disk) |

---

## Running Tests

```bash
pytest                           # all 379 tests
pytest tests/unit/               # unit tests
pytest tests/unit/api/           # API only
pytest tests/unit/clinical/      # pipeline only
pytest tests/unit/rag/           # RAG only
pytest -v -o "addopts="          # verbose, no coverage
```

---

## Configuration

All settings via `.env` file (see `.env.example` for all options):

| Variable | Default | Description |
|----------|---------|-------------|
| `ALLOWED_ORIGINS` | `["http://localhost:3000"]` | CORS origins (JSON array) |
| `OLLAMA_MODEL` | `llama3.1:8b` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Embedding model |
| `CHROMA_COLLECTION` | `clinical_knowledge_base` | ChromaDB collection name |
| `RAG_TOP_K` | `5` | Documents retrieved per query |
| `RAG_SIMILARITY_THRESHOLD` | `0.35` | Minimum similarity score |

---

## License

MIT
