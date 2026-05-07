import argparse
import json
from pathlib import Path

from src.ops.checks import all_checks_ok, run_checks
from src.utils.config import load_config
from src.utils.logging import setup_logging


def run_check_command(config_path: Path, profile_path: Path | None, require_ollama: bool) -> int:
    config = load_config(config_path, profile_path)
    setup_logging(config)
    results = run_checks(config)
    print(json.dumps([result.to_dict() for result in results], indent=2))
    return 0 if all_checks_ok(results, require_ollama=require_ollama) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ARIA operational checks.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--profile", type=Path, default=None)
    parser.add_argument("--require-ollama", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(run_check_command(args.config, args.profile, args.require_ollama))


if __name__ == "__main__":
    main()
