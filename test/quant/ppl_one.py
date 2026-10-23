"""Perplexity comparison for Qwen2.5-1.5B-Instruct fp16 vs AWQ.

Uses vLLM prompt_logprobs so both variants are measured identically.
Usage: python ppl_one.py {fp16|awq|custom|qat} {out.json}
"""

import json
import math
import os
import sys

from vllm import LLM, SamplingParams

VARIANT = sys.argv[1]
OUT_PATH = sys.argv[2]

MODEL = {
    "fp16": "Qwen/Qwen2.5-1.5B-Instruct",
    "awq": "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
    "custom": os.environ.get(
        "VLLM_CUSTOM_MODEL", "./Qwen2.5-1.5B-Instruct-AWQ-custom"
    ),
    "qat": os.environ.get(
        "VLLM_QAT_MODEL", "./Qwen2.5-1.5B-Instruct-QAT-AWQ"
    ),
}[VARIANT]

# Fixed eval corpus: varied Chinese/English passages.
CORPUS = [
    "北京是中华人民共和国的首都，位于华北平原北部，有着三千余年的建城史和八百多年的建都史。作为全国政治中心、文化中心、国际交往中心和科技创新中心，北京汇聚了故宫、天坛、颐和园等众多世界文化遗产。",
    "光合作用是绿色植物利用光能将二氧化碳和水转化为有机物并释放氧气的过程。这一过程主要发生在叶绿体中，是地球上绝大多数生命活动能量的根本来源，对维持大气中的氧气含量至关重要。",
    "Machine learning is a subfield of artificial intelligence that gives computers the ability to learn from data without being explicitly programmed. Deep learning, a subset of machine learning, uses multi-layered neural networks to model complex patterns in data.",
    "黄河是中国第二长河，发源于青藏高原巴颜喀拉山北麓，流经九个省区，最终注入渤海。黄河流域是中华文明最重要的发祥地之一，两岸孕育了灿烂的仰韶文化和龙山文化。",
    "快速排序是由 Tony Hoare 在 1960 年提出的一种分治排序算法。它选择一个基准元素，将数组划分为小于和大于基准的两个子数组，再递归地对子数组排序，平均时间复杂度为 O(n log n)。",
    "The Great Wall of China is a series of fortifications built across the historical northern borders of China. Several walls were built from as early as the 7th century BC, and the most well-known sections were built by the Ming dynasty.",
    "云计算通过互联网按需提供计算资源，包括服务器、存储、数据库和网络。用户无需购买和维护物理硬件，只需按使用量付费，这大幅降低了企业的 IT 成本并提高了资源利用效率。",
    "量子计算利用量子比特的叠加态和纠缠特性进行信息处理。与经典比特只能处于 0 或 1 不同，量子比特可以同时处于两种状态的叠加，这使得量子计算机在特定问题上具有指数级的加速潜力。",
]


def main():
    import os
    os.environ.setdefault("VLLM_USE_MODELSCOPE", "True")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    llm = LLM(model=MODEL, gpu_memory_utilization=0.80, max_model_len=1024)
    outputs = llm.generate(
        CORPUS,
        SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0),
    )

    total_nll, total_tokens = 0.0, 0
    per_text = []
    for text, out in zip(CORPUS, outputs):
        # prompt_logprobs[i] is None for the first token; skip it.
        tok_lp = [
            next(iter(p.values())).logprob
            for p in out.prompt_logprobs
            if p is not None
        ]
        nll = -sum(tok_lp)
        total_nll += nll
        total_tokens += len(tok_lp)
        per_text.append({"nll": nll, "tokens": len(tok_lp)})

    ppl = math.exp(total_nll / total_tokens)
    result = {
        "variant": VARIANT,
        "model": MODEL,
        "corpus_texts": len(CORPUS),
        "corpus_tokens": total_tokens,
        "perplexity": round(ppl, 3),
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
