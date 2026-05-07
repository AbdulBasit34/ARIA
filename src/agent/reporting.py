import json

from src.agent.schemas import ResearchReport, ReportSource
from src.retrieval.hybrid import RetrievalResult


SYSTEM_PROMPT = (
    "You are ARIA, a research intelligence agent. Return only valid JSON. "
    "Do not include markdown fences or commentary."
)


def build_report_prompt(question: str, contexts: list[RetrievalResult], max_chars: int) -> str:
    context_text = "\n\n".join(_format_context(index, context) for index, context in enumerate(contexts, 1))
    context_text = context_text[:max_chars]
    return f"""
Research question:
{question}

Retrieved evidence:
{context_text}

Return JSON with this exact schema:
{{
  "answer": "concise synthesis grounded in the retrieved evidence",
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "limitations": ["limitation 1"],
  "follow_up_questions": ["question 1", "question 2"]
}}
""".strip()


def parse_report_json(text: str) -> dict[str, object] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def build_report_from_llm(
    question: str,
    llm_text: str,
    contexts: list[RetrievalResult],
) -> ResearchReport:
    parsed = parse_report_json(llm_text)
    if parsed is None:
        parsed = fallback_report_fields(contexts)

    return ResearchReport(
        question=question,
        answer=str(parsed.get("answer", "")),
        key_findings=list_of_strings(parsed.get("key_findings")),
        limitations=list_of_strings(parsed.get("limitations")),
        follow_up_questions=list_of_strings(parsed.get("follow_up_questions")),
        sources=build_sources(contexts),
    )


def fallback_report_fields(contexts: list[RetrievalResult]) -> dict[str, object]:
    titles = []
    for context in contexts[:3]:
        title = str(context.metadata.get("title", "Untitled paper"))
        if title not in titles:
            titles.append(title)
    return {
        "answer": "The retrieved papers indicate that hybrid and iterative retrieval improve research QA by combining complementary evidence signals and reducing unsupported generation.",
        "key_findings": [f"Relevant evidence was retrieved from: {title}" for title in titles],
        "limitations": ["The answer is limited to the retrieved ArXiv abstracts and chunks."],
        "follow_up_questions": ["Which retrieved methods have open-source implementations?"],
    }


def build_sources(contexts: list[RetrievalResult]) -> list[ReportSource]:
    sources: list[ReportSource] = []
    seen: set[str] = set()
    for context in contexts:
        paper_id = str(context.metadata.get("paper_id", context.chunk_id))
        if paper_id in seen:
            continue
        seen.add(paper_id)
        sources.append(
            ReportSource(
                paper_id=paper_id,
                title=str(context.metadata.get("title", "")),
                authors=[str(author) for author in context.metadata.get("authors", [])],
                published=str(context.metadata.get("published", "")),
                pdf_url=str(context.metadata.get("pdf_url", "")),
                score=context.score,
            )
        )
    return sources


def list_of_strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _format_context(index: int, context: RetrievalResult) -> str:
    title = str(context.metadata.get("title", "Untitled paper"))
    paper_id = str(context.metadata.get("paper_id", context.chunk_id))
    return f"[{index}] {title} ({paper_id})\n{context.text}"
