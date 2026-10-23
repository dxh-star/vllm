"""Train a memory-friendly QAT adapter and export a vLLM W4A16 checkpoint.

This script uses LoRA-QAT: the FP16 base weights are frozen, LoRA parameters
are trained, and every targeted Linear layer uses a straight-through 4-bit
fake-quantized weight during the forward pass. After training, the adapter is
merged into the FP16 weights and LLM Compressor exports a vLLM-compatible
compressed-tensors W4A16 checkpoint.

Example:
    quant-awq-env/bin/python tools/quantization/qat_qwen_awq.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --output ./Qwen2.5-1.5B-Instruct-QAT-AWQ \
        --calibration-file ./calibration.jsonl \
        --steps 200
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration-file", type=Path, required=True)
    parser.add_argument("--calibration-samples", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="LoRA learning rate. FP32 LoRA parameters default to 2e-5.",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_texts(path: Path, limit: int) -> list[str]:
    records = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{line_number} requires a non-empty text")
            records.append(text)
    if not records:
        raise ValueError(f"No calibration records found in {path}")
    random.shuffle(records)
    return records[: min(limit, len(records))]


def fake_quantize(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """Return symmetric INT4 dequantization with a straight-through gradient."""
    original_shape = weight.shape
    if weight.shape[-1] % group_size:
        raise ValueError(
            f"Linear input dimension {weight.shape[-1]} is not divisible by "
            f"--group-size {group_size}"
        )
    grouped = weight.reshape(-1, group_size)
    scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7.0
    quantized = (grouped / scale).round().clamp(-8, 7)
    dequantized = (quantized * scale).reshape(original_shape)
    return weight + (dequantized - weight).detach()


class QATLinear(nn.Module):
    """Frozen linear layer with LoRA training and fake-quantized weights."""

    def __init__(
        self, layer: nn.Linear, rank: int, alpha: float, group_size: int
    ) -> None:
        super().__init__()
        self.base = layer
        self.group_size = group_size
        self.scaling = alpha / rank if rank else 0.0
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        if rank:
            self.lora_a = nn.Parameter(
                torch.empty(
                    rank,
                    layer.in_features,
                    device=layer.weight.device,
                    dtype=torch.float32,
                )
            )
            self.lora_b = nn.Parameter(
                torch.zeros(
                    layer.out_features,
                    rank,
                    device=layer.weight.device,
                    dtype=torch.float32,
                )
            )
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = fake_quantize(self.base.weight, self.group_size)
        output = F.linear(inputs, weight, self.base.bias)
        if self.lora_a is not None:
            update = F.linear(F.linear(inputs.float(), self.lora_a), self.lora_b)
            update = update.to(dtype=inputs.dtype)
            output = output + update * self.scaling
        return output


def replace_linears(model: nn.Module, rank: int, alpha: float, group_size: int):
    replaced = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or name.endswith("lm_head"):
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        qat_layer = QATLinear(module, rank, alpha, group_size)
        setattr(parent, child_name, qat_layer)
        replaced.append(name)
    return replaced


def merge_and_restore(model: nn.Module) -> None:
    for name, module in list(model.named_modules()):
        if not isinstance(module, QATLinear):
            continue
        merged = module.base
        if module.lora_a is not None:
            update = module.lora_b @ module.lora_a
            merged.weight.data.add_(update, alpha=module.scaling)
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, merged)


def collate(tokenizer, texts: list[str], max_length: int, device: torch.device):
    batch = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    batch = {key: value.to(device) for key, value in batch.items()}
    batch["labels"] = batch["input_ids"].clone()
    batch["labels"][batch["attention_mask"] == 0] = -100
    return batch


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.lora_rank < 0:
        raise ValueError("steps, batch size, and LoRA rank must be non-negative")
    if args.calibration_samples < 1 or args.max_seq_length < 2:
        raise ValueError("calibration samples and sequence length are invalid")
    if not torch.cuda.is_available():
        raise RuntimeError("QAT training requires a CUDA device")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading FP16 model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    texts = read_texts(args.calibration_file, args.calibration_samples)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    replaced = replace_linears(model, args.lora_rank, args.lora_alpha, args.group_size)
    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters were created")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    optimizer.zero_grad(set_to_none=True)

    print(f"QAT layers: {len(replaced)}, samples: {len(texts)}, steps: {args.steps}")
    for step in range(args.steps):
        step_loss = 0.0
        for micro_step in range(args.gradient_accumulation_steps):
            start = (step * args.gradient_accumulation_steps + micro_step) * args.batch_size
            batch_texts = [texts[(start + index) % len(texts)] for index in range(args.batch_size)]
            batch = collate(tokenizer, batch_texts, args.max_seq_length, device)
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"QAT loss became non-finite at step {step + 1}; "
                    "try a lower --learning-rate or shorter --max-seq-length"
                )
            loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            step_loss += loss.item()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if step == 0 or (step + 1) % 10 == 0:
            print(f"step {step + 1}/{args.steps}, loss={step_loss:.4f}")

    merge_and_restore(model)
    model.config.use_cache = True
    model.eval()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)

    print("Exporting merged QAT weights as compressed-tensors W4A16")
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier

    dataset = Dataset.from_dict({"text": texts})
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        recipe=[
            AWQModifier(),
            QuantizationModifier(
                targets=["Linear"], scheme="W4A16", ignore=["lm_head"]
            ),
        ],
        max_seq_length=args.max_seq_length,
        num_calibration_samples=len(texts),
    )
    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved QAT-AWQ checkpoint to: {output_dir}")


if __name__ == "__main__":
    main()
