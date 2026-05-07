from functools import lru_cache
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


@lru_cache(maxsize=1)
def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> DictConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return OmegaConf.load(path)


def get_config_value(config: DictConfig, key: str, default: Any | None = None) -> Any:
    value = OmegaConf.select(config, key)
    return default if value is None else value
