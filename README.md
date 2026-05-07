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

## Module 3 Verification

```powershell
python -m pytest tests/test_ollama_client.py -q
ollama pull llama3:8b-instruct-q4_K_M
python -m src.agent.cli prompt "Write one sentence about hybrid retrieval for research QA."
```

## Module 4 Verification

```powershell
python -m pytest tests/test_workflow.py -q
python -m src.agent.cli research "How does hybrid retrieval improve research question answering?"
```

The default `arxiv.mode` is `local` while the project is being completed. Set it to `online` later to use live ArXiv search with retry/backoff.

## Module 5 Verification

```powershell
python -m pytest tests/test_serving.py -q
python -m src.serving.cli
```

In another PowerShell window:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/research -ContentType "application/json" -Body '{"question":"How does hybrid retrieval improve research question answering?"}'
```

## Module 6 Verification

Install the fine-tuning stack:

```powershell
python -m pip install -e ".[finetune]"
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Run smoke tests:

```powershell
python -m pytest tests/test_finetuning.py -q
python -m src.finetuning.train_qlora --config configs/config.yaml --preflight
```

Start QLoRA training:

```powershell
python -m src.finetuning.train_qlora --config configs/config.yaml
```

Create an Ollama Modelfile after training:

```powershell
python -m src.finetuning.export_ollama --config configs/config.yaml --output models/Modelfile.aria
ollama create aria-llama3-8b-q4 -f models/Modelfile.aria
```

## Module 7 Verification

```powershell
python -m pytest tests/test_ops.py -q
python -m src.ops.cli --profile configs/local.yaml
python -m src.ops.cli --profile configs/local.yaml --require-ollama
```

## Module 8 Verification

```powershell
python -m pytest tests/test_e2e.py -q
python -m src.ops.e2e "How does hybrid retrieval improve research question answering?" --profile configs/local.yaml --output data/reports/latest_report.json
```

Switch to online ArXiv at the end:

```powershell
python -m src.ops.e2e "How does hybrid retrieval improve research question answering?" --profile configs/online.yaml --output data/reports/online_report.json
```
