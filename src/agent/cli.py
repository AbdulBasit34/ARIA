import argparse
import json
from pathlib import Path

from src.agent.ollama_client import OllamaClient
from src.agent.workflow import ResearchWorkflow
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


def run_research(question: str, config_path: Path) -> None:
    config = load_config(config_path)
    setup_logging(config)
    report = ResearchWorkflow(config).run(question)
    print(report.model_dump_json(indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ARIA agent commands.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    subparsers = parser.add_subparsers(dest="command")

    prompt_parser = subparsers.add_parser("prompt")
    prompt_parser.add_argument("prompt", type=str)

    research_parser = subparsers.add_parser("research")
    research_parser.add_argument("question", type=str)

    parser.add_argument("legacy_prompt", nargs="?", type=str)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prompt":
        run_prompt(args.prompt, args.config)
    elif args.command == "research":
        run_research(args.question, args.config)
    elif args.legacy_prompt:
        run_prompt(args.legacy_prompt, args.config)
    else:
        build_parser().print_help()


if __name__ == "__main__":
    main()
