"""Quantize a causal language model to AWQ W4A16 for vLLM.

Example:
    quant-awq-env/bin/python tools/quantization/quantize_qwen_awq.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --output ./Qwen2.5-1.5B-Instruct-AWQ-custom

For a local calibration file, use one JSON object per line with a ``text``
field and pass ``--calibration-file path/to/calibration.jsonl``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_CALIBRATION_TEXTS = [
    "北京是中华人民共和国的首都，拥有悠久的历史和丰富的文化。",
    "请介绍中国近现代史中的重要事件。",
    "什么是机器学习？请用简单的语言解释。",
    "请解释快速排序算法的基本思想和时间复杂度。",
    "光合作用是如何发生的？",
    "请用 Python 写一个读取 JSON 文件的示例。",
    "Machine learning is a method for learning patterns from data.",
    "Explain the difference between training and inference in a language model.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a vLLM-compatible AWQ W4A16 checkpoint."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Hugging Face model ID or local model directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory in which to save the quantized checkpoint.",
    )
    parser.add_argument(
        "--calibration-file",
        type=Path,
        help="JSONL file containing objects with a non-empty 'text' field.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=512,
        help="Number of calibration samples. Defaults to 512.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum calibration sequence length. Defaults to 2048.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed for calibration data.",
    )
    return parser.parse_args()


def load_calibration_dataset(
    tokenizer,
    calibration_file: Path | None,
    max_seq_length: int,
    seed: int,
) -> Dataset:
    from datasets import Dataset

    if calibration_file is None:
        texts = DEFAULT_CALIBRATION_TEXTS
    else:
        records = []
        with calibration_file.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                text = record.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(
                        f"{calibration_file}:{line_number} must contain a "
                        "non-empty string field named 'text'"
                    )
                records.append(text)
        if not records:
            raise ValueError(f"No calibration records found in {calibration_file}")
        texts = records

    dataset = Dataset.from_dict({"text": texts}).shuffle(seed=seed)

    def tokenize(example):
        return tokenizer(
            example["text"],
            padding=False,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        )

    return dataset.map(tokenize)


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier

    if args.calibration_samples < 1:
        raise ValueError("--calibration-samples must be positive")
    if args.max_seq_length < 1:
        raise ValueError("--max-seq-length must be positive")

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    dataset = load_calibration_dataset(
        tokenizer, args.calibration_file, args.max_seq_length, args.seed
    )
    num_calibration_samples = min(args.calibration_samples, len(dataset))
    if num_calibration_samples < args.calibration_samples:
        print(
            f"Warning: only {len(dataset)} calibration records are available; "
            f"using {num_calibration_samples} samples."
        )

    recipe = [
        AWQModifier(),
        QuantizationModifier(
            targets=["Linear"],
            scheme="W4A16",
            ignore=["lm_head"],
        ),
    ]

    print(
        f"Calibrating with {num_calibration_samples} samples "
        f"and max sequence length {args.max_seq_length}"
    )
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=num_calibration_samples,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)

    print(f"Saved AWQ checkpoint to: {output_dir}")


if __name__ == "__main__":
    main()
