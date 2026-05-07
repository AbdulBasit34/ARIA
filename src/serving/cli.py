import argparse
from pathlib import Path

import uvicorn

from src.utils.config import load_config
from src.serving.app import create_app


def serve(config_path: Path, profile_path: Path | None = None) -> None:
    config = load_config(config_path, profile_path)
    uvicorn.run(
        create_app(config),
        host=str(config.serving.host),
        port=int(config.serving.port),
        reload=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ARIA FastAPI server.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--profile", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    serve(args.config, args.profile)


if __name__ == "__main__":
    main()
