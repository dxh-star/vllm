# Qwen2.5-1.5B-Instruct 量化对比：fp16 vs AWQ (4-bit)

- 日期：2026-09-28
- 硬件：RTX 3060 12GB，vLLM 0.30.0（V1 引擎），CUDA graph 开启
- 对比对象：`Qwen/Qwen2.5-1.5B-Instruct`（fp16）vs `Qwen/Qwen2.5-1.5B-Instruct-AWQ`（官方 4-bit AWQ）
- 脚本：`bench_one.py`（显存/速度/质量）、`ppl_one.py`（困惑度）

## 结果汇总

| 指标 | fp16 | AWQ 4-bit | 变化 |
|---|---|---|---|
| 权重文件 | 2.9 GB | 1.6 GB | -45% |
| 引擎权重+非 torch 显存占用 | 3.25 GiB | ~1 GiB（见 bench 日志） | ↓ |
| 峰值显存（util=0.8 预留满） | 11036 MiB | 11064 MiB | 基本持平* |
| TTFT（单请求） | 13 ms | 8 ms | -38% |
| 单请求解码速度 | 79.8 tok/s | 185.1 tok/s | **+132%** |
| 批量吞吐（32 并发） | 1754 tok/s | 3486 tok/s | **+99%** |
| 困惑度 PPL（8 篇中英文语料，401 token） | **4.027** | 4.843 | +20.3%（劣化） |

\* 峰值显存受 `gpu_memory_utilization=0.80` 支配，两边都会把预留空间占满；
真实差异体现在权重占用（3.25 GiB vs ~1 GiB）和小显存卡上的可部署性
（7B 模型 fp16 装不下 12GB 卡，AWQ 可以）。

## 质量观察

- 同一 prompt（"请详细介绍北京的历史、文化和现代发展…"）下，fp16 输出 381 token
  自然收尾；AWQ 输出 512 token（顶满 max_tokens），结尾出现明显句子级循环重复
  （"北京的历史、文化和现代发展相互交织…"重复 7 次）——4-bit 量化的典型退化。
- PPL 4.027 → 4.843（+20%）定量印证了该退化。

## 结论

1. **速度**：AWQ 解码吞吐约为 fp16 的 2 倍（RTX 3060 上解码是显存带宽受限，
   4-bit 权重减半使带宽瓶颈直接缓解）。
2. **显存**：权重 -45%，同等显存可换更大 KV cache 或部署更大模型；
   本实验中因 KV cache 预留拉平，峰值显存差异未体现。
3. **精度**：PPL +20% 且长文本出现循环重复，日常对话尚可，长文生成质量明显受损。
4. **建议**：12GB 卡跑 ≤1.5B 模型优先 fp16（质量好且装得下）；
   7B 及以上模型才用 AWQ 换取可部署性。

## 复现

```bash
source vllm-env/bin/activate
export VLLM_USE_MODELSCOPE=True VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME=$(dirname $(dirname $(find vllm-env -name nvcc -type f | head -1)))
python test/quant/bench_one.py fp16 test/quant/result_fp16.json
python test/quant/bench_one.py awq  test/quant/result_awq.json
python test/quant/ppl_one.py   fp16 test/quant/ppl_fp16.json
python test/quant/ppl_one.py   awq  test/quant/ppl_awq.json
```

原始数据：`result_fp16.json` / `result_awq.json` / `ppl_fp16.json` / `ppl_awq.json`
