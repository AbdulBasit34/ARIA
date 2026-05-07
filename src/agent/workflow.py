import json
from pathlib import Path
from typing import Any, TypedDict

from loguru import logger
from langgraph.graph import END, StateGraph
from omegaconf import DictConfig, OmegaConf

from src.agent.ollama_client import OllamaClient
from src.agent.reporting import SYSTEM_PROMPT, build_report_from_llm, build_report_prompt
from src.agent.schemas import ResearchReport
from src.ingestion.arxiv_client import search_papers
from src.ingestion.chunker import chunk_paper
from src.retrieval import BM25Index, HybridRetriever, QdrantVectorStore
from src.retrieval.hybrid import RetrievalResult


class ResearchState(TypedDict, total=False):
    question: str
    chunks: list[dict[str, object]]
    contexts: list[RetrievalResult]
    llm_text: str
    report: ResearchReport


class ResearchWorkflow:
    def __init__(
        self,
        config: DictConfig,
        retriever: HybridRetriever | None = None,
        llm_client: OllamaClient | None = None,
    ) -> None:
        self.config = config
        self.retriever = retriever or HybridRetriever(
            config=config,
            embedder=build_embedder(config),
            vector_store=QdrantVectorStore(config),
            bm25_index=BM25Index(),
        )
        self.llm_client = llm_client or OllamaClient(config)
        self.graph = self._build_graph()

    def run(self, question: str) -> ResearchReport:
        result = self.graph.invoke({"question": question})
        report = result["report"]
        if not isinstance(report, ResearchReport):
            raise TypeError("Workflow did not return a ResearchReport")
        return report

    def _build_graph(self) -> Any:
        graph = StateGraph(ResearchState)
        graph.add_node("ingest", self._ingest)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("answer", self._answer)
        graph.add_node("report", self._report)
        graph.set_entry_point("ingest")
        graph.add_edge("ingest", "retrieve")
        graph.add_edge("retrieve", "answer")
        graph.add_edge("answer", "report")
        graph.add_edge("report", END)
        return graph.compile()

    def _ingest(self, state: ResearchState) -> ResearchState:
        if str(OmegaConf.select(self.config, "arxiv.mode", default="online")).lower() == "local":
            return {"chunks": load_cached_chunks(Path(str(self.config.arxiv.cache_path)))}

        try:
            papers = search_papers(state["question"], self.config)
            chunks = [chunk for paper in papers for chunk in chunk_paper(paper, self.config)]
            return {"chunks": [chunk.__dict__ for chunk in chunks]}
        except Exception as error:
            if not bool(self.config.arxiv.use_cache_fallback):
                raise
            logger.warning("Live ArXiv ingestion failed: {}. Falling back to cached chunks.", error)
            return {"chunks": load_cached_chunks(Path(str(self.config.arxiv.cache_path)))}

    def _retrieve(self, state: ResearchState) -> ResearchState:
        self.retriever.index_chunks(state["chunks"])
        contexts = self.retriever.retrieve(state["question"])
        max_chunks = int(self.config.report.max_context_chunks)
        return {"contexts": contexts[:max_chunks]}

    def _answer(self, state: ResearchState) -> ResearchState:
        prompt = build_report_prompt(
            state["question"],
            state["contexts"],
            max_chars=int(self.config.report.max_context_chars),
        )
        response = self.llm_client.generate(prompt, system_prompt=SYSTEM_PROMPT)
        return {"llm_text": response.response}

    def _report(self, state: ResearchState) -> ResearchState:
        return {
            "report": build_report_from_llm(
                question=state["question"],
                llm_text=state["llm_text"],
                contexts=state["contexts"],
            )
        }


def load_cached_chunks(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"ArXiv cache not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"ArXiv cache has no chunks: {path}")
    return [dict(chunk) for chunk in chunks]


def build_embedder(config: DictConfig) -> Any:
    from src.retrieval.embeddings import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(config)
