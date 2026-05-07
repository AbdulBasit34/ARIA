import argparse
from pathlib import Path

import uvicorn

from src.utils.config import load_config


def serve(config_path: Path) -> None:
    config = load_config(config_path)
    uvicorn.run(
        "src.serving.app:app",
        host=str(config.serving.host),
        port=int(config.serving.port),
        reload=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ARIA FastAPI server.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    serve(args.config)


if __name__ == "__main__":
    main()
