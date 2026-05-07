from collections.abc import Callable
from typing import Protocol

from fastapi import FastAPI, HTTPException
from loguru import logger
from omegaconf import DictConfig

from src.agent.schemas import ResearchReport
from src.serving.schemas import ResearchRequest, ResearchResponse
from src.utils.config import load_config
from src.utils.logging import setup_logging


class WorkflowRunner(Protocol):
    def run(self, question: str) -> ResearchReport:
        ...


WorkflowFactory = Callable[[DictConfig], WorkflowRunner]


def create_app(
    config: DictConfig | None = None,
    workflow_factory: WorkflowFactory | None = None,
) -> FastAPI:
    resolved_config = config or load_config()
    setup_logging(resolved_config)
    factory = workflow_factory or build_workflow

    api = FastAPI(title="ARIA", version="0.1.0")
    api.state.config = resolved_config
    api.state.workflow_factory = factory
    api.state.workflow = None

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "aria"}

    @api.post("/research", response_model=ResearchResponse)
    def research(request: ResearchRequest) -> ResearchResponse:
        try:
            workflow = get_workflow(api)
            report = workflow.run(request.question)
            return ResearchResponse(report=report)
        except Exception as error:
            logger.exception("Research request failed")
            raise HTTPException(status_code=500, detail=str(error)) from error

    return api


def get_workflow(api: FastAPI) -> WorkflowRunner:
    if api.state.workflow is None:
        api.state.workflow = api.state.workflow_factory(api.state.config)
    return api.state.workflow


def build_workflow(config: DictConfig) -> WorkflowRunner:
    from src.agent.workflow import ResearchWorkflow

    return ResearchWorkflow(config)


app = create_app()
