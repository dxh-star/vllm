<!-- markdownlint-disable MD001 MD041 -->
**简体中文** | [English](README.md)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
为所有人提供简单、快速且经济的 LLM 服务
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>文档</b></a> | <a href="https://blog.vllm.ai/"><b>博客</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>论文</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>用户论坛</b></a> | <a href="https://slack.vllm.ai"><b>开发者 Slack</b></a> |
</p>

🔥 我们创建了 vLLM 网站，帮助你快速上手 vLLM。欢迎访问 [vllm.ai](https://vllm.ai) 了解更多信息。
如需了解活动信息，请访问 [vllm.ai/events](https://vllm.ai/events) 加入我们。

---

## 项目简介

vLLM 是一个快速、易用的 LLM 推理与服务库。

vLLM 最初由加州大学伯克利分校的 [Sky Computing Lab](https://sky.cs.berkeley.edu) 开发，如今已经发展为最活跃的开源 AI 项目之一。来自全球 2000 多位贡献者以及数十所高校和企业的社区成员共同维护和推动着 vLLM 的发展。

vLLM 具有以下高性能特性：

- 业界领先的服务吞吐量
- 使用 [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html) 高效管理注意力机制中的键值内存
- 支持请求的连续批处理、分块预填充和前缀缓存
- 使用分段式和完整 CUDA/HIP 图实现快速、灵活的模型执行
- 支持量化：FP8、MXFP8/MXFP4、NVFP4、INT8、INT4、GPTQ/AWQ、GGUF、compressed-tensors、ModelOpt、TorchAO 等，详见[更多量化方案](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- 优化的注意力算子，包括 FlashAttention、FlashInfer、TRTLLM-GEN、FlashMLA 和 Triton
- 使用 CUTLASS、TRTLLM-GEN、CuTeDSL，为多种精度优化 GEMM/MoE 算子
- 支持包括 n-gram、suffix、EAGLE 和 DFlash 在内的推测解码方法
- 使用 torch.compile 自动生成算子并进行图级变换
- 支持预填充、解码和编码阶段的解耦部署

vLLM 灵活且易于使用：

- 与 Hugging Face 上的主流模型无缝集成
- 支持多种解码算法的高吞吐服务，包括*并行采样*、*束搜索*等
- 支持张量、流水线、数据、专家和上下文并行，用于分布式推理
- 支持流式输出
- 使用 xgrammar 或 guidance 生成结构化输出
- 支持工具调用和推理解析器
- 高效支持稠密层和 MoE 层的多 LoRA
- 支持 NVIDIA GPU、AMD GPU、Intel GPU 以及 x86/ARM/PowerPC CPU；此外还支持 Google TPU、Intel Gaudi、IBM Spyre、华为 Ascend、Rebellions NPU、Apple Silicon、MetaX GPU 等多种硬件插件

vLLM 原生支持 Hugging Face 上 200 多种模型架构，包括：

- 仅解码器架构的 LLM（例如 Llama、Qwen、Gemma）
- 混合专家（MoE）LLM（例如 Mixtral、DeepSeek-V3、Qwen-MoE、GPT-OSS）
- 混合注意力和状态空间模型（例如 Mamba、Qwen3.5）
- 多模态模型（例如 LLaVA、Qwen-VL、Pixtral）
- 嵌入和检索模型（例如 E5-Mistral、GTE、ColBERT）
- 奖励模型和分类模型（例如 Qwen-Math）

完整的支持模型列表请参见[此处](https://docs.vllm.ai/en/latest/models/supported_models.html)。

## 快速开始

使用 [`uv`](https://docs.astral.sh/uv/)（推荐）或 `pip` 安装 vLLM：

```bash
uv pip install vllm
```

开发时也可以按照[源码构建指南](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source)从源码构建。

欢迎阅读我们的[文档](https://docs.vllm.ai/en/latest/)了解更多信息：

- [安装](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [快速入门](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [支持的模型列表](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Qwen2.5-1.5B 模型搭建、推理与优化结果

本项目使用 vLLM 进行推理和服务，不负责重新训练 Qwen 基础模型。当前对比的两个模型如下：

1. FP16 主模型：`Qwen/Qwen2.5-1.5B-Instruct`
2.  量化模型：`Qwen/Qwen2.5-1.5B-Instruct-AWQ`

### 模型构成

- `Qwen/Qwen2.5-1.5B-Instruct` 是 Qwen2.5 约 1.5B 参数的指令微调模型，以 FP16 权重直接加载到 vLLM。
- `Qwen/Qwen2.5-1.5B-Instruct-AWQ` 是 AWQ 4-bit 预量化模型。由 vLLM 识别量化配置后执行推理。
- 两个模型属于同一模型系列和相同规模，适合在相同 prompt、相同采样参数和相同硬件上进行对照。

### 环境与启动

以下命令对应本项目的 RTX 3060 12GB 测试环境。两个服务使用同一个端口，测试另一个模型前需要先停止当前服务。

```bash
cd /home/dxh/DXH/code/vllm-main
source .venv/bin/activate

export VLLM_USE_MODELSCOPE=True
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME="$(dirname "$(dirname "$(find "$VIRTUAL_ENV" -name nvcc -type f | head -1)")")"
```

启动 FP16 主模型：

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --gpu-memory-utilization 0.80 \
  --port 8000
```

启动 AWQ 量化模型：

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct-AWQ \
  --gpu-memory-utilization 0.80 \
  --port 8000
```

`--gpu-memory-utilization 0.80` 用于限制 vLLM 的显存使用比例，并为 KV Cache 和运行时开销预留空间。AWQ 模型已经包含量化配置，通常不需要额外添加 `--quantization awq`。

### 推理命令

启动服务后，模型名称必须与当前服务实际加载的模型一致。

FP16 推理：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{
      "role": "user",
      "content": "请用300字左右介绍北京的历史、文化和现代发展。"
    }],
    "temperature": 0.0,
    "max_tokens": 512
  }'
```

AWQ 推理：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
    "messages": [{
      "role": "user",
      "content": "请用300字左右介绍北京的历史、文化和现代发展。"
    }],
    "temperature": 0.0,
    "max_tokens": 512
  }'
```

如果需要控制过短的回答，可以使用 vLLM 扩展参数 `min_tokens`，但应同时设置合理的 `max_tokens`，避免量化模型在长文本中出现重复生成：

```json
{
  "max_tokens": 512,
  "min_tokens": 200
}
```

### 基准测试与质量评估命令

仓库中的对比脚本会分别测试显存、单请求解码速度、32 并发吞吐和困惑度（PPL）。需要分别启动或加载两个模型，每个脚本进程只测试一个模型：

```bash
cd /home/dxh/DXH/code/vllm-main
source .venv/bin/activate
export VLLM_USE_MODELSCOPE=True
export VLLM_USE_FLASHINFER_SAMPLER=0

.venv/bin/python test/quant/bench_one.py \
  fp16 test/quant/result_fp16.json
.venv/bin/python test/quant/bench_one.py \
  awq test/quant/result_awq.json

.venv/bin/python test/quant/ppl_one.py \
  fp16 test/quant/ppl_fp16.json
.venv/bin/python test/quant/ppl_one.py \
  awq test/quant/ppl_awq.json
```

> 注：上述脚本会直接创建 vLLM 引擎，不需要先启动 HTTP 服务；FP16 和 AWQ 应分开执行，避免两个模型同时占用 GPU 显存。

### 推理优化结果

测试条件：RTX 3060 12GB、vLLM 0.30.0、CUDA Graph 开启、`gpu_memory_utilization=0.80`。

| 指标 | FP16 | AWQ 4-bit | AWQ 相对 FP16 |
| --- | ---: | ---: | ---: |
| 权重文件大小 | 2.9 GB | 1.6 GB | 减少约 45% |
| TTFT，单请求 | 13 ms | 8 ms | 约减少 38% |
| 单请求解码速度 | 79.8 tok/s | 185.1 tok/s | 提升约 132% |
| 32 并发吞吐 | 1754 tok/s | 3486 tok/s | 提升约 99% |
| PPL，8 篇中英文文本、401 tokens | 4.027 | 4.843 | 劣化约 20.3% |

结果分析：

- **速度**：AWQ 通过 4-bit 权重减少显存带宽读取，在 RTX 3060 上单请求解码约为 FP16 的 2.3 倍，32 并发吞吐约为 2 倍。
- **显存**：AWQ 的权重文件从 2.9 GB 降至 1.6 GB，可以为 KV Cache 留出更多空间，也提高 12 GB 显卡部署更大模型的可能性。由于两种测试都设置了 `gpu_memory_utilization=0.80`，峰值显存都会接近预留上限，不能只看峰值显存判断权重节省效果。
- **质量**：FP16 在测试 prompt 下约生成 381 tokens 后自然结束；AWQ 生成到 512 tokens 上限，末尾出现句子级重复。PPL 从 4.027 上升到 4.843，说明 4-bit 量化带来了可观的质量退化。
- **适用场景**：1.5B 模型在 12 GB 显卡上能够使用 FP16 时，优先选择 FP16；显存紧张、并发吞吐优先，或需要部署 7B 及以上模型时，再选择 AWQ。

完整报告和原始结果：

- [量化对比报告](test/quant/REPORT.md)
- [FP16 benchmark 结果](test/quant/result_fp16.json)
- [AWQ benchmark 结果](test/quant/result_awq.json)
- [FP16 PPL 结果](test/quant/ppl_fp16.json)
- [AWQ PPL 结果](test/quant/ppl_awq.json)

### 原始结果摘要

以下数据直接来自上述 JSON 文件。由于 benchmark 存在不同运行轮次，原始 JSON 与报告中的代表性汇总值略有差异；两者的结论一致。

FP16 原始 benchmark：

```json
{
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "gpu_peak_mib": 11047.8,
  "single": {
    "out_tokens": 381,
    "ttft_s": 0.017,
    "e2e_s": 4.158,
    "decode_tps": 91.8
  },
  "batch": {
    "num_requests": 32,
    "elapsed_s": 3.7,
    "total_out_tokens": 7248,
    "throughput_tok_s": 1959.5
  }
}
```

AWQ 原始 benchmark：

```json
{
  "model": "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
  "gpu_peak_mib": 11077.9,
  "single": {
    "out_tokens": 512,
    "ttft_s": 0.012,
    "e2e_s": 2.438,
    "decode_tps": 210.7
  },
  "batch": {
    "num_requests": 32,
    "elapsed_s": 1.84,
    "total_out_tokens": 7160,
    "throughput_tok_s": 3889.9
  }
}
```

PPL 原始结果：

| 模型 | 评估文本数 | 评估 tokens | PPL |
| --- | ---: | ---: | ---: |
| `Qwen/Qwen2.5-1.5B-Instruct` | 8 | 401 | 4.027 |
| `Qwen/Qwen2.5-1.5B-Instruct-AWQ` | 8 | 401 | 4.843 |

在相同的北京历史、文化和现代发展 prompt 下，FP16 生成 381 tokens 后自然结束；AWQ 生成到 512 tokens 上限，结尾出现“北京的历史、文化和现代发展相互交织……”等句子级重复。该现象与 AWQ PPL 从 4.027 上升到 4.843 的结果一致。

## 参与贡献

我们欢迎并重视各种贡献与合作。
如需了解如何参与，请阅读[为 vLLM 做贡献](https://docs.vllm.ai/en/latest/contributing/index.html)。

## 引用

如果你在研究中使用了 vLLM，请引用我们的[论文](https://arxiv.org/abs/2309.06180)：

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## 联系我们

<!-- --8<-- [start:contact-us] -->
- 技术问题和功能请求，请使用 GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- 与其他用户交流，请使用 [vLLM 用户论坛](https://discuss.vllm.ai)
- 协调贡献与开发，请使用 [Slack](https://slack.vllm.ai)
- 安全问题披露，请使用 GitHub 的 [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) 功能
- 合作与商务事宜，请联系 [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## 媒体资源

- 如需使用 vLLM 的 Logo，请参考我们的 [媒体资源仓库](https://github.com/vllm-project/media-kit)。
