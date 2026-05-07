import argparse
from pathlib import Path

from omegaconf import DictConfig

from src.utils.config import load_config


def build_modelfile(config: DictConfig, output_path: Path, require_adapter: bool = True) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_path = Path(str(config.finetuning.output_dir)).resolve()
    if require_adapter:
        validate_adapter_path(adapter_path)
    output_path.write_text(
        "\n".join(
            [
                f"FROM {config.ollama.fallback_model}",
                f"ADAPTER {adapter_path}",
                'SYSTEM """You are ARIA, a research intelligence agent. Answer with grounded, structured research synthesis."""',
                f"PARAMETER temperature {config.ollama.temperature}",
                f"PARAMETER num_ctx {config.ollama.num_ctx}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def validate_adapter_path(adapter_path: Path) -> None:
    expected_file = adapter_path / "adapter_model.safetensors"
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"LoRA adapter directory not found: {adapter_path}. Run training first."
        )
    if not expected_file.exists():
        raise FileNotFoundError(
            f"LoRA adapter file not found: {expected_file}. Training did not finish correctly."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create an Ollama Modelfile for the ARIA LoRA adapter.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("models/Modelfile.aria"))
    parser.add_argument("--allow-missing-adapter", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_modelfile(
        load_config(args.config),
        args.output,
        require_adapter=not args.allow_missing_adapter,
    )


if __name__ == "__main__":
    main()
