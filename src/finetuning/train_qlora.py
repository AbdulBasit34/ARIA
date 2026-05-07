import argparse
from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.finetuning.formatting import format_qasper_example
from src.utils.config import load_config
from src.utils.logging import setup_logging


def train(config: DictConfig) -> None:
    setup_logging(config)
    logger.info("QLoRA output directory: {}", Path(str(config.finetuning.output_dir)).resolve())
    logger.info(
        "QLoRA memory settings: 4bit={}, max_seq_len={}, batch_size={}, grad_accum={}",
        bool(config.finetuning.load_in_4bit),
        int(config.finetuning.max_seq_len),
        int(config.finetuning.batch_size),
        int(config.finetuning.gradient_accumulation_steps),
    )
    require_cuda()

    logger.info("Importing fine-tuning libraries")
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    from unsloth import FastLanguageModel, is_bfloat16_supported
    logger.info("Fine-tuning libraries imported")

    logger.info("Loading base model {}", str(config.finetuning.base_model))
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(config.finetuning.base_model),
        max_seq_length=int(config.finetuning.max_seq_len),
        dtype=None,
        load_in_4bit=bool(config.finetuning.load_in_4bit),
    )
    logger.info("Base model loaded")
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(config.finetuning.lora.r),
        target_modules=list(config.finetuning.lora.target_modules),
        lora_alpha=int(config.finetuning.lora.alpha),
        lora_dropout=float(config.finetuning.lora.dropout),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=int(config.finetuning.seed),
    )

    dataset = load_qasper_dataset(config)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=int(config.finetuning.max_seq_len),
        args=SFTConfig(
            output_dir=str(config.finetuning.output_dir),
            per_device_train_batch_size=int(config.finetuning.batch_size),
            gradient_accumulation_steps=int(config.finetuning.gradient_accumulation_steps),
            warmup_steps=int(config.finetuning.warmup_steps),
            max_steps=int(config.finetuning.max_steps),
            learning_rate=float(config.finetuning.learning_rate),
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=int(config.finetuning.logging_steps),
            save_steps=int(config.finetuning.save_steps),
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=int(config.finetuning.seed),
            report_to="none",
        ),
    )

    logger.info("Starting QLoRA training")
    trainer.train()
    output_dir = Path(str(config.finetuning.output_dir))
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Saved LoRA adapter to {}", output_dir)


def preflight(config: DictConfig) -> None:
    setup_logging(config)
    require_cuda()
    require_imports(["unsloth", "datasets", "trl", "bitsandbytes", "peft", "transformers"])
    require_torchao_compatibility()
    logger.info("Importing fine-tuning stack")
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    from unsloth import FastLanguageModel

    _ = (load_dataset, SFTTrainer, SFTConfig, FastLanguageModel)
    output_dir = Path(str(config.finetuning.output_dir)).resolve()
    logger.info("CUDA available")
    logger.info("Output directory will be {}", output_dir)
    logger.info("Base model: {}", str(config.finetuning.base_model))
    logger.info("Dataset: {} [{}]", dataset_name(config), dataset_split(config))
    logger.info("Preflight passed")


def load_qasper_dataset(config: DictConfig) -> Any:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name(config), split=dataset_split(config))
    max_samples = int(config.finetuning.max_train_samples)
    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    def map_example(example: dict[str, Any]) -> dict[str, str]:
        return {"text": format_qasper_example(example)}

    return dataset.map(map_example, remove_columns=dataset.column_names)


def dataset_name(config: DictConfig) -> str:
    return str(OmegaConf.select(config, "finetuning.dataset_name", default="allenai/qasper"))


def dataset_split(config: DictConfig) -> str:
    return str(OmegaConf.select(config, "finetuning.dataset_split", default="train"))


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Unsloth QLoRA training. Install CUDA PyTorch first.")
    logger.info("CUDA device: {}", torch.cuda.get_device_name(0))


def require_torchao_compatibility() -> None:
    import importlib.util
    import torch

    if importlib.util.find_spec("torchao") is None:
        return
    if not hasattr(torch, "int1"):
        raise RuntimeError(
            "Installed torchao expects torch.int1, but this PyTorch build does not provide it. "
            "Force-upgrade CUDA PyTorch, torchvision, and torchaudio before training."
        )


def require_imports(module_names: list[str]) -> None:
    import importlib.util

    missing = [name for name in module_names if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Missing fine-tuning dependencies: {', '.join(missing)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune Llama-3-8B on QASPER with Unsloth QLoRA.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--preflight", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.preflight:
        preflight(config)
    else:
        train(config)


if __name__ == "__main__":
    main()
