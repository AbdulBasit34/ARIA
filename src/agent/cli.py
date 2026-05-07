import argparse
import json
from pathlib import Path

from src.agent.ollama_client import OllamaClient
from src.utils.config import load_config
from src.utils.logging import setup_logging


def run_prompt(prompt: str, config_path: Path) -> None:
    config = load_config(config_path)
    setup_logging(config)
    response = OllamaClient(config).generate(prompt)
    print(
        json.dumps(
            {
                "model": response.model,
                "response": response.response,
                "latency_ms": response.latency_ms,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens,
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call the configured Ollama model.")
    parser.add_argument("prompt", type=str)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_prompt(args.prompt, args.config)


if __name__ == "__main__":
    main()
