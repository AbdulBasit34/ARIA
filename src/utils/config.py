from functools import lru_cache
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


DEFAULT_CONFIG_PATH = Path("configs/config.yaml")


@lru_cache(maxsize=1)
def load_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    profile_path: str | Path | None = None,
) -> DictConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    config = OmegaConf.load(path)
    if profile_path is None:
        return config

    profile = Path(profile_path)
    if not profile.exists():
        raise FileNotFoundError(f"Config profile not found: {profile}")
    return OmegaConf.merge(config, OmegaConf.load(profile))


def get_config_value(config: DictConfig, key: str, default: Any | None = None) -> Any:
    value = OmegaConf.select(config, key)
    return default if value is None else value
