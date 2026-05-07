import json
from pathlib import Path
from uuid import uuid4

from src.agent.schemas import ResearchReport, ReportSource
from src.ops.e2e import run_e2e


class FakeWorkflow:
    def run(self, question: str) -> ResearchReport:
        return ResearchReport(
            question=question,
            answer="ARIA completed an end-to-end smoke run.",
            key_findings=["The workflow returned a structured report."],
            limitations=["This is a fake workflow test."],
            follow_up_questions=["Run against live ArXiv later?"],
            sources=[
                ReportSource(
                    paper_id="test",
                    title="Test Paper",
                    authors=["Tester"],
                    published="2026-01-01T00:00:00+00:00",
                    pdf_url="https://example.com/test.pdf",
                    score=1.0,
                )
            ],
        )


def test_run_e2e_writes_report() -> None:
    output_path = Path(".test_tmp") / f"report_{uuid4().hex}.json"

    report = run_e2e(
        "Can ARIA run end to end?",
        Path("configs/config.yaml"),
        Path("configs/local.yaml"),
        output_path,
        workflow=FakeWorkflow(),
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert report.question == "Can ARIA run end to end?"
    assert payload["answer"] == "ARIA completed an end-to-end smoke run."
