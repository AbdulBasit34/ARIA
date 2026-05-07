from time import perf_counter

import httpx
from loguru import logger
from omegaconf import DictConfig

from src.agent.llm_logger import LLMCallLog, append_llm_call_log, now_utc_iso
from src.agent.schemas import LLMResponse


class OllamaClient:
    def __init__(self, config: DictConfig, http_client: httpx.Client | None = None) -> None:
        self.config = config
        self.base_url = str(config.ollama.base_url).rstrip("/")
        self.model = str(config.ollama.model)
        self.fallback_model = str(config.ollama.fallback_model)
        self.timeout_seconds = float(config.ollama.timeout_seconds)
        self.temperature = float(config.ollama.temperature)
        self.num_ctx = int(config.ollama.num_ctx)
        self.num_predict = int(config.ollama.num_predict)
        self.log_path = str(config.logging.llm_log_path)
        self.http_client = http_client or httpx.Client(timeout=self.timeout_seconds)

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        try:
            return self._generate_with_model(self.model, prompt, system_prompt)
        except httpx.HTTPStatusError as error:
            if not self._should_retry_with_fallback(error):
                raise
            logger.warning(
                "Ollama model {} unavailable; retrying with {}",
                self.model,
                self.fallback_model,
            )
            return self._generate_with_model(self.fallback_model, prompt, system_prompt)

    def _generate_with_model(
        self,
        model: str,
        prompt: str,
        system_prompt: str | None,
    ) -> LLMResponse:
        started = perf_counter()
        prompt_tokens = estimate_tokens(prompt)
        try:
            response = self.http_client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_ctx": self.num_ctx,
                        "num_predict": self.num_predict,
                    },
                },
            )
            response.raise_for_status()
            payload = response.json()
            latency_ms = elapsed_ms(started)
            output_text = str(payload.get("response", ""))
            completion_tokens = int(payload.get("eval_count") or estimate_tokens(output_text))
            prompt_tokens = int(payload.get("prompt_eval_count") or prompt_tokens)
            llm_response = LLMResponse(
                model=model,
                prompt=prompt,
                response=output_text,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                raw=dict(payload),
            )
            self._log_call(llm_response, success=True)
            return llm_response
        except Exception as error:
            latency_ms = elapsed_ms(started)
            completion_tokens = 0
            self._log_record(
                LLMCallLog(
                    timestamp=now_utc_iso(),
                    provider="ollama",
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    latency_ms=latency_ms,
                    success=False,
                    error=str(error),
                )
            )
            raise

    def _log_call(self, response: LLMResponse, success: bool) -> None:
        self._log_record(
            LLMCallLog(
                timestamp=now_utc_iso(),
                provider="ollama",
                model=response.model,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
                latency_ms=response.latency_ms,
                success=success,
            )
        )

    def _log_record(self, record: LLMCallLog) -> None:
        append_llm_call_log(self.log_path, record)

    def _should_retry_with_fallback(self, error: httpx.HTTPStatusError) -> bool:
        if self.fallback_model == self.model:
            return False
        if error.response.status_code == 404:
            return True
        return "not found" in error.response.text.lower()


def estimate_tokens(text: str) -> int:
    return max(1, len(text.split())) if text.strip() else 0


def elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000.0, 3)
