import argparse
from pathlib import Path
from typing import Protocol

from loguru import logger
from omegaconf import DictConfig

from src.agent.schemas import ResearchReport
from src.utils.config import load_config
from src.utils.logging import setup_logging


class WorkflowRunner(Protocol):
    def run(self, question: str) -> ResearchReport:
        ...


def run_e2e(
    question: str,
    config_path: Path,
    profile_path: Path | None,
    output_path: Path,
    workflow: WorkflowRunner | None = None,
) -> ResearchReport:
    config = load_config(config_path, profile_path)
    setup_logging(config)
    runner = workflow or build_workflow(config)
    report = runner.run(question)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Wrote E2E report to {}", output_path)
    return report


def build_workflow(config: DictConfig) -> WorkflowRunner:
    from src.agent.workflow import ResearchWorkflow

    return ResearchWorkflow(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ARIA end-to-end and write a report JSON file.")
    parser.add_argument("question", type=str)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--profile", type=Path, default=Path("configs/local.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/latest_report.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_e2e(args.question, args.config, args.profile, args.output)


if __name__ == "__main__":
    main()
