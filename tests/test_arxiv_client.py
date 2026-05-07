import arxiv
from omegaconf import OmegaConf

from src.ingestion.arxiv_client import _status_from_error


def test_status_from_error_detects_rate_limit() -> None:
    error = arxiv.HTTPError("https://export.arxiv.org/api/query", 0, 429)

    assert _status_from_error(error) == 429


def test_arxiv_config_keeps_online_retry_settings() -> None:
    config = OmegaConf.load("configs/config.yaml")

    assert str(config.arxiv.mode) == "local"
    assert float(config.arxiv.request_delay_seconds) >= 5.0
    assert int(config.arxiv.num_retries) >= 6
