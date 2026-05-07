import sys
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig


def setup_logging(config: DictConfig) -> None:
    log_path = Path(str(config.logging.app_log_path))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        level=str(config.logging.level),
        enqueue=False,
        backtrace=False,
        diagnose=False,
    )
    logger.add(
        log_path,
        level=str(config.logging.level),
        rotation=str(config.logging.rotation),
        retention=str(config.logging.retention),
        enqueue=False,
        backtrace=False,
        diagnose=False,
    )
