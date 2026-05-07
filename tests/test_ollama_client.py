import json
from pathlib import Path
from uuid import uuid4

import httpx
from omegaconf import OmegaConf

from src.agent.ollama_client import OllamaClient, estimate_tokens


def test_estimate_tokens_counts_words() -> None:
    assert estimate_tokens("hybrid retrieval improves grounding") == 4
    assert estimate_tokens("") == 0


def test_ollama_client_logs_success() -> None:
    log_path = make_test_log_path()
    config = OmegaConf.create(
        {
            "ollama": {
                "base_url": "http://ollama.test",
                "model": "aria-test",
                "fallback_model": "llama3-test",
                "timeout_seconds": 5,
                "temperature": 0.2,
                "num_ctx": 2048,
                "num_predict": 128,
            },
            "logging": {"llm_log_path": str(log_path)},
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["stream"] is False
        return httpx.Response(
            200,
            json={
                "model": payload["model"],
                "response": "Hybrid retrieval combines dense and sparse signals.",
                "prompt_eval_count": 7,
                "eval_count": 8,
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    response = OllamaClient(config, http_client=client).generate("Explain hybrid retrieval.")

    assert response.model == "aria-test"
    assert response.total_tokens == 15
    records = log_path.read_text(encoding="utf-8").splitlines()
    assert len(records) == 1
    assert json.loads(records[0])["success"] is True


def test_ollama_client_falls_back_when_main_model_missing() -> None:
    log_path = make_test_log_path()
    config = OmegaConf.create(
        {
            "ollama": {
                "base_url": "http://ollama.test",
                "model": "aria-missing",
                "fallback_model": "llama3-test",
                "timeout_seconds": 5,
                "temperature": 0.2,
                "num_ctx": 2048,
                "num_predict": 128,
            },
            "logging": {"llm_log_path": str(log_path)},
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        if payload["model"] == "aria-missing":
            return httpx.Response(404, text="model not found")
        return httpx.Response(200, json={"response": "ok", "prompt_eval_count": 2, "eval_count": 1})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    response = OllamaClient(config, http_client=client).generate("Hello")

    assert response.model == "llama3-test"
    records = log_path.read_text(encoding="utf-8").splitlines()
    assert len(records) == 2
    assert json.loads(records[0])["success"] is False
    assert json.loads(records[1])["success"] is True


def make_test_log_path() -> Path:
    return Path(".test_tmp") / f"llm_calls_{uuid4().hex}.jsonl"
