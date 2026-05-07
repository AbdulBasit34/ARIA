from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_checks(config: DictConfig) -> list[CheckResult]:
    return [
        check_config(config),
        check_arxiv_mode(config),
        check_ingestion_cache(config),
        check_embedding_mode(config),
        check_qdrant_mode(config),
        check_ollama(config),
    ]


def all_checks_ok(results: list[CheckResult], require_ollama: bool = False) -> bool:
    if require_ollama:
        return all(result.ok for result in results)
    return all(result.ok for result in results if result.name != "ollama")


def check_config(config: DictConfig) -> CheckResult:
    required_paths = ["arxiv.mode", "embeddings.model_name", "qdrant.mode", "ollama.base_url"]
    missing = [path for path in required_paths if OmegaConf.select(config, path) is None]
    if missing:
        return CheckResult("config", False, f"Missing config keys: {', '.join(missing)}")
    return CheckResult("config", True, "Required config keys are present")


def check_arxiv_mode(config: DictConfig) -> CheckResult:
    mode = str(config.arxiv.mode).lower()
    if mode not in {"local", "online"}:
        return CheckResult("arxiv", False, f"Unsupported arxiv.mode: {mode}")
    return CheckResult("arxiv", True, f"arxiv.mode={mode}")


def check_ingestion_cache(config: DictConfig) -> CheckResult:
    if str(config.arxiv.mode).lower() != "local":
        return CheckResult("cache", True, "Not required in online mode")
    cache_path = Path(str(config.arxiv.cache_path))
    if not cache_path.exists():
        return CheckResult("cache", False, f"Missing local cache: {cache_path}")
    return CheckResult("cache", True, f"Found local cache: {cache_path}")


def check_embedding_mode(config: DictConfig) -> CheckResult:
    batch_size = int(config.embeddings.batch_size)
    if batch_size > 32:
        return CheckResult("embeddings", False, "Embedding batch_size must be <= 32")
    return CheckResult("embeddings", True, f"device={config.embeddings.device}, batch_size={batch_size}")


def check_qdrant_mode(config: DictConfig) -> CheckResult:
    mode = str(config.qdrant.mode).lower()
    if mode not in {"local", "memory", "http"}:
        return CheckResult("qdrant", False, f"Unsupported qdrant.mode: {mode}")
    return CheckResult("qdrant", True, f"qdrant.mode={mode}")


def check_ollama(config: DictConfig) -> CheckResult:
    try:
        response = httpx.get(f"{str(config.ollama.base_url).rstrip('/')}/api/tags", timeout=2.0)
        response.raise_for_status()
    except Exception as error:
        return CheckResult("ollama", False, f"Ollama not reachable: {error}")
    return CheckResult("ollama", True, "Ollama is reachable")
