"""Benchmark Qwen2.5-1.5B-Instruct fp16 vs AWQ: memory, speed, quality.

Run one model per process (fp16|awq), save results to JSON.
Usage: .venv python bench_one.py {fp16|awq} {out.json}
"""

import json
import sys
import threading
import time

from vllm import LLM, SamplingParams

VARIANT = sys.argv[1]  # "fp16" | "awq"
OUT_PATH = sys.argv[2]

MODEL = {
    "fp16": "Qwen/Qwen2.5-1.5B-Instruct",
    "awq": "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
}[VARIANT]


def sample_memory(stop, peak):
    import pynvml
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    while not stop.is_set():
        peak[0] = max(peak[0], pynvml.nvmlDeviceGetMemoryInfo(h).used)
        time.sleep(0.05)


def main():
    import os
    os.environ.setdefault("VLLM_USE_MODELSCOPE", "True")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    peak = [0]
    stop = threading.Event()
    mem_thread = threading.Thread(target=sample_memory, args=(stop, peak), daemon=True)
    mem_thread.start()

    llm = LLM(model=MODEL, gpu_memory_utilization=0.80, disable_log_stats=False)

    # --- 1. single-request decode speed ---
    single_prompt = "请详细介绍北京的历史、文化和现代发展，要求内容丰富、结构清晰。"
    single = llm.generate(
        [single_prompt],
        SamplingParams(temperature=0.0, max_tokens=512),
    )
    single_elapsed = None  # derived from metrics below

    # --- 2. batch throughput ---
    batch_prompts = [
        "解释什么是机器学习",
        "写一首关于秋天的诗",
        "Translate to English: 今天天气很好。",
        "用Python写一个快速排序",
        "简述光合作用的过程",
        "为什么天空是蓝色的？",
        "介绍一段中国历史",
        "如何提高编程能力？",
    ] * 4  # 32 requests
    t0 = time.perf_counter()
    batch = llm.generate(
        batch_prompts,
        SamplingParams(temperature=0.0, max_tokens=256),
    )
    batch_elapsed = time.perf_counter() - t0
    batch_out_tokens = sum(len(o.outputs[0].token_ids) for o in batch)

    stop.set()
    mem_thread.join()

    stats = single[0].metrics
    gen_tokens = stats.num_generation_tokens if stats else None
    ttft = stats.first_token_ts - stats.queued_ts if stats else None
    e2e = stats.last_token_ts - stats.queued_ts if stats else None

    result = {
        "variant": VARIANT,
        "model": MODEL,
        "gpu_peak_mib": round(peak[0] / 1024**2, 1),
        "single": {
            "prompt": single_prompt,
            "output": single[0].outputs[0].text,
            "out_tokens": gen_tokens,
            "ttft_s": round(ttft, 3) if ttft else None,
            "e2e_s": round(e2e, 3) if e2e else None,
            "decode_tps": round((gen_tokens - 1) / (e2e - ttft), 1)
            if gen_tokens and ttft and e2e and e2e > ttft else None,
        },
        "batch": {
            "num_requests": len(batch_prompts),
            "elapsed_s": round(batch_elapsed, 2),
            "total_out_tokens": batch_out_tokens,
            "throughput_tok_s": round(batch_out_tokens / batch_elapsed, 1),
        },
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[{VARIANT}] done -> {OUT_PATH}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
