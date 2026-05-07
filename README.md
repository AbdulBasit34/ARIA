# ARIA

Autonomous Research Intelligence Agent.

## Build Plan

1. Module 1: ArXiv ingestion, config, logging, chunking.
2. Module 2: embeddings, Qdrant indexing, BM25, RRF hybrid retrieval.
3. Module 3: Ollama Llama-3 orchestration, structured JSON LLM-call logs.
4. Module 4: LangGraph research workflow and report schema.
5. Module 5: FastAPI serving endpoint.
6. Module 6: Unsloth QLoRA fine-tuning pipeline for QASPER.
7. Module 7: Docker Compose for Qdrant and local smoke tests.

## Conda Setup

```powershell
conda create -n aria python=3.11 -y
conda activate aria
python -m pip install --upgrade pip
python -m pip install -e .
```

Install fine-tuning extras only when you are ready to train:

```powershell
python -m pip install -e ".[finetune]"
```

## Module 1 Verification

```powershell
python -m pytest tests/test_chunker.py -q
python -m src.ingestion.cli "retrieval augmented generation for scientific question answering" --output data/ingestion/arxiv_chunks.json
```

## Module 2 Verification

```powershell
python -m pytest tests/test_retrieval.py -q
python -m src.retrieval.cli index --input data/ingestion/arxiv_chunks.json
python -m src.retrieval.cli search "How does hybrid retrieval improve research question answering?" --input data/ingestion/arxiv_chunks.json
```

Docker is optional. Set `qdrant.mode: http` in `configs/config.yaml` before using:

```powershell
docker compose up -d qdrant
```
