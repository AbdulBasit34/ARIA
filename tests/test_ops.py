from pathlib import Path

from omegaconf import OmegaConf

from src.ops.checks import all_checks_ok, check_ingestion_cache, run_checks
from src.utils.config import load_config


def test_load_config_merges_profile() -> None:
    config = load_config("configs/config.yaml", "configs/online.yaml")

    assert str(config.arxiv.mode) == "online"
    assert bool(config.arxiv.use_cache_fallback) is False


def test_local_cache_check_reports_missing_cache() -> None:
    config = OmegaConf.create({"arxiv": {"mode": "local", "cache_path": "missing.json"}})

    result = check_ingestion_cache(config)

    assert result.ok is False
    assert "Missing local cache" in result.detail


def test_all_checks_can_ignore_ollama() -> None:
    config = OmegaConf.create(
        {
            "arxiv": {"mode": "online", "cache_path": "data/ingestion/arxiv_chunks.json"},
            "embeddings": {"batch_size": 32, "device": "cuda", "model_name": "x"},
            "qdrant": {"mode": "local"},
            "ollama": {"base_url": "http://127.0.0.1:1"},
        }
    )

    results = run_checks(config)

    assert all_checks_ok(results, require_ollama=False) is True
    assert all_checks_ok(results, require_ollama=True) is False


def test_profiles_exist() -> None:
    assert Path("configs/local.yaml").exists()
    assert Path("configs/online.yaml").exists()
