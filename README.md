# CCPA Compliance Checker

A RAG-based (Retrieval Augmented Generation) system that analyzes text prompts and scenarios to detect CCPA (California Consumer Privacy Act) violations using an LLM and vector embeddings.

## Overview

This project provides an intelligent compliance checker that:
- Analyzes user prompts and scenarios for potential CCPA violations
- Identifies specific violated CCPA articles and sections
- Leverages a local Qwen2.5-1.5B language model for efficient analysis
- Uses ChromaDB with semantic search for grounding responses in the CCPA statute
- Exposes a FastAPI endpoint for easy integration

## Features

- **RAG Architecture**: Combines document retrieval with LLM generation for accurate, sourced responses
- **Lightweight Model**: Uses Qwen2.5-1.5B Instruct model (~8GB VRAM requirement)
- **Vector Database**: ChromaDB with BAAI/bge-large-en-v1.5 embeddings for semantic search
- **Structured Output**: Returns JSON with compliance status and relevant CCPA sections
- **FastAPI Server**: RESTful API with health checks and validation
- **Docker Ready**: Containerizable for deployment and testing

## Architecture

### Components

1. **app.py**: Core compliance analysis logic
   - Vector database initialization and document loading
   - LLM pipeline configuration
   - Retrieval chain setup for RAG

2. **api.py**: FastAPI server
   - `/analyze` endpoint for compliance checking
   - `/health` endpoint for server status
   - Request/response validation

3. **validate_format.py**: Test suite
   - Organizer-side validation script
   - Test case definitions
   - Endpoint response verification

4. **download_models.py**: Model pre-caching utility
   - Pre-downloads LLM and embedding models to cache
   - Speeds up initial startup

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for 8GB+ VRAM)
- HuggingFace API token (for gated models, optional)

### Setup Steps

1. **Clone the repository** (if applicable)

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set HuggingFace token** (optional but recommended):
   ```bash
   export HF_TOKEN="your_huggingface_token"
   ```

4. **Prepare the CCPA statute PDF**:
   - Place `ccpa_statute.pdf` in the project root directory
   - The PDF will be automatically loaded and indexed into ChromaDB

5. **Pre-download models** (optional, recommended for faster startup):
   ```bash
   python download_models.py
   ```

## Usage

### Starting the Server

```bash
python api.py
```

The FastAPI server will start on `http://localhost:8000` and load models on startup (may take 2-5 minutes).

### API Endpoints

#### Health Check
```
GET /health
```

Returns: `{"status": "ready"}` when models are loaded

#### Analyze for CCPA Compliance
```
POST /analyze
Content-Type: application/json

{
  "prompt": "We are selling customers' personal data without consent"
}
```

**Response** (successful):
```json
{
  "harmful": true,
  "articles": ["Section 1798.120", "Section 1798.100"]
}
```

**Response** (compliant):
```json
{
  "harmful": false,
  "articles": []
}
```

### Response Format

- **harmful** (boolean): `true` if the scenario describes a CCPA violation, `false` otherwise
- **articles** (list): 
  - Non-empty list of CCPA section numbers when `harmful=true`
  - Empty list when `harmful=false`

## Testing

Run the test suite to validate the system:

```bash
python validate_format.py
```

This script:
- Waits for server readiness with a 5-minute timeout
- Executes predefined test cases
- Validates response format and accuracy
- Reports pass/fail results for each test

## Model Details

- **LLM**: Qwen/Qwen2.5-1.5B-Instruct
  - Lightweight (~3.3B parameters in quantized form)
  - Supports 8B context window
  - Instruction-tuned for compliance analysis

- **Embeddings**: BAAI/bge-large-en-v1.5
  - 1024-dimensional embeddings
  - Optimized for semantic search

- **Vector Store**: ChromaDB
  - Persistent storage in `./chroma_db`
  - Automatic persistence between runs

## Environment Variables

- `HF_TOKEN`: HuggingFace API token (optional)
  - If not set, public models work without authentication
  - Gated models may fail without a valid token

## Performance

- **Startup time**: 2-5 minutes (model loading + initialization)
- **Query latency**: 5-30 seconds per prompt (LLM inference time varies)
- **Memory usage**: ~8GB VRAM (GPU recommended), ~4GB RAM

## Docker Deployment

To containerize for submission:

1. Ensure `Dockerfile` and `docker-compose.yml` are configured
2. Build and run:
   ```bash
   docker compose up -d
   python validate_format.py
   docker compose down
   ```

## Troubleshooting

### Models fail to download
- **Issue**: HuggingFace token not set
- **Solution**: Set `HF_TOKEN` environment variable with valid token

### "Could not find ccpa_statute.pdf"
- **Issue**: PDF not in project root
- **Solution**: Ensure `ccpa_statute.pdf` exists in the same directory as `app.py`

### Slow inference on CPU
- **Issue**: Using CPU instead of GPU
- **Solution**: Ensure CUDA is installed and PyTorch is configured for GPU

### "CUDA out of memory"
- **Issue**: GPU has insufficient VRAM
- **Solution**: Reduce batch size or use a smaller model

## Dependencies

Key packages:
- `langchain`: RAG framework
- `langchain-community`, `langchain-classic`: Integration modules
- `chromadb`: Vector database
- `huggingface-hub`, `transformers`: Model downloads and inference
- `fastapi`: Web framework
- `pydantic`: Data validation
- `torch`: Deep learning backend

See `requirements.txt` for complete dependency list.

## Author Notes

This system is designed for hackathon evaluation and production use. It balances accuracy with efficiency using a lightweight model while leveraging semantic search for relevant context.
