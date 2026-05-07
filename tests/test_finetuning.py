import importlib.util
from pathlib import Path
from uuid import uuid4

import tomllib

from omegaconf import OmegaConf

from src.finetuning.export_ollama import build_modelfile, validate_adapter_path
from src.finetuning.formatting import extract_answer, format_qasper_example
from src.finetuning.train_qlora import require_torchao_compatibility


def test_format_qasper_example_builds_instruction_prompt() -> None:
    text = format_qasper_example(
        {
            "question": "What improves research QA?",
            "abstract": "Hybrid retrieval combines dense and sparse search.",
            "answer": {"free_form_answer": "Hybrid retrieval improves grounding."},
        }
    )

    assert "### Instruction:" in text
    assert "What improves research QA?" in text
    assert "Hybrid retrieval improves grounding." in text


def test_extract_answer_handles_unanswerable() -> None:
    answer = extract_answer({"answer": {"unanswerable": True}})

    assert answer == "The evidence is insufficient."


def test_finetuning_config_is_memory_safe() -> None:
    config = OmegaConf.load("configs/config.yaml")

    assert bool(config.finetuning.load_in_4bit) is True
    assert int(config.finetuning.batch_size) == 1
    assert int(config.finetuning.gradient_accumulation_steps) == 4
    assert int(config.finetuning.max_seq_len) <= 512


def test_finetuning_dependencies_are_unpinned() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    finetune_deps = pyproject["project"]["optional-dependencies"]["finetune"]

    assert "transformers" in finetune_deps
    assert "torchao" in finetune_deps


def test_torchao_compatibility_check_has_clear_error(monkeypatch) -> None:
    import torch

    if hasattr(torch, "int1"):
        return
    original_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: object() if name == "torchao" else original_find_spec(name),
    )

    try:
        require_torchao_compatibility()
    except RuntimeError as error:
        assert "Force-upgrade CUDA PyTorch" in str(error)
    else:
        raise AssertionError("Expected torchao compatibility check to fail clearly")


def test_build_modelfile_references_adapter() -> None:
    config = OmegaConf.create(
        {
            "ollama": {
                "fallback_model": "llama3:8b-instruct-q4_K_M",
                "temperature": 0.2,
                "num_ctx": 4096,
            },
            "finetuning": {"output_dir": "models/aria-llama3-qasper-lora"},
        }
    )
    output_path = Path(".test_tmp") / f"Modelfile.{uuid4().hex}"

    build_modelfile(config, output_path, require_adapter=False)

    text = output_path.read_text(encoding="utf-8")
    assert "FROM llama3:8b-instruct-q4_K_M" in text
    assert f"ADAPTER {Path('models/aria-llama3-qasper-lora').resolve()}" in text


def test_validate_adapter_path_requires_safetensors() -> None:
    missing_path = Path(".test_tmp") / f"missing_adapter_{uuid4().hex}"

    try:
        validate_adapter_path(missing_path)
    except FileNotFoundError as error:
        assert "Run training first" in str(error)
    else:
        raise AssertionError("Expected missing adapter path to fail")
