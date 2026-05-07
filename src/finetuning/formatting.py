from typing import Any


PROMPT_TEMPLATE = """### Instruction:
Answer the research question using the provided paper context. If the answer is not present, say the evidence is insufficient.

### Question:
{question}

### Context:
{context}

### Answer:
{answer}"""


def format_qasper_example(example: dict[str, Any], max_context_chars: int = 2500) -> str:
    question = extract_question(example)
    context = extract_context(example, max_context_chars=max_context_chars)
    answer = extract_answer(example)
    return PROMPT_TEMPLATE.format(question=question, context=context, answer=answer)


def extract_question(example: dict[str, Any]) -> str:
    return str(example.get("question") or example.get("qas", {}).get("question") or "").strip()


def extract_context(example: dict[str, Any], max_context_chars: int) -> str:
    abstract = example.get("abstract", "")
    if isinstance(abstract, dict):
        abstract_text = " ".join(str(value) for value in abstract.values())
    elif isinstance(abstract, list):
        abstract_text = " ".join(str(value) for value in abstract)
    else:
        abstract_text = str(abstract)

    full_text = example.get("full_text", "")
    if isinstance(full_text, dict):
        sections = full_text.get("paragraphs") or full_text.get("section_name") or full_text.values()
        full_text_value = " ".join(_flatten_to_strings(sections))
    elif isinstance(full_text, list):
        full_text_value = " ".join(_flatten_to_strings(full_text))
    else:
        full_text_value = str(full_text)

    context = f"{abstract_text}\n{full_text_value}".strip()
    return context[:max_context_chars]


def extract_answer(example: dict[str, Any]) -> str:
    answer = example.get("answer")
    if answer:
        return normalize_answer(answer)

    answers = example.get("answers")
    if isinstance(answers, list) and answers:
        return normalize_answer(answers[0])

    qas = example.get("qas")
    if isinstance(qas, dict):
        qas_answers = qas.get("answers")
        if isinstance(qas_answers, list) and qas_answers:
            return normalize_answer(qas_answers[0])

    return "The evidence is insufficient."


def normalize_answer(answer: Any) -> str:
    if isinstance(answer, str):
        return answer.strip()
    if isinstance(answer, dict):
        for key in ("free_form_answer", "extractive_spans", "yes_no", "unanswerable"):
            value = answer.get(key)
            if value in (None, "", []):
                continue
            if key == "unanswerable" and bool(value):
                return "The evidence is insufficient."
            if isinstance(value, list):
                return " ".join(str(item) for item in value).strip()
            return str(value).strip()
    if isinstance(answer, list):
        return " ".join(str(item) for item in answer).strip()
    return str(answer).strip()


def _flatten_to_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [part for item in value.values() for part in _flatten_to_strings(item)]
    if isinstance(value, list):
        return [part for item in value for part in _flatten_to_strings(item)]
    return [str(value)]
