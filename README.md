<!-- markdownlint-disable MD001 MD041 -->
[简体中文](README.zh-CN.md) | **English**

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, Intel GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Qwen2.5-1.5B Setup, Inference, and Optimization Results

This project uses vLLM for inference and serving. It does not fully retrain the Qwen base model. The four models compared here are:

1. FP16 base model: `Qwen/Qwen2.5-1.5B-Instruct`
2. Official AWQ 4-bit quantized model: `Qwen/Qwen2.5-1.5B-Instruct-AWQ`
3. PTQ AWQ 4-bit quantized model: `qwen2.5-1.5b-awq-custom`
4. QAT-exported AWQ 4-bit model: `Qwen2.5-1.5B-Instruct-QAT-AWQ`

### Model composition

- `Qwen/Qwen2.5-1.5B-Instruct` is an instruction-tuned Qwen2.5 model with approximately 1.5B parameters. Its FP16 weights are loaded directly by vLLM.
- `Qwen/Qwen2.5-1.5B-Instruct-AWQ` is an official 4-bit AWQ pre-quantized model. It is not retrained in this project; vLLM loads it directly and detects its quantization configuration.
- `qwen2.5-1.5b-awq-custom` is the PTQ AWQ model generated offline from the FP16 base model with `tools/quantization/quantize_qwen_awq.py`. It is saved as a `compressed-tensors` W4A16 checkpoint and then loaded by vLLM.
- `Qwen2.5-1.5B-Instruct-QAT-AWQ` is exported after LoRA-QAT training from the FP16 base model. Its training forward pass uses 4-bit fake quantization, and the merged result is saved as a vLLM-compatible `compressed-tensors` W4A16 checkpoint.
- The four models belong to the same model family and have the same approximate size, making them suitable for comparison with the same prompts, sampling parameters, and hardware.

### Environment and serving

The following commands correspond to the project's RTX 3060 12GB test environment. The four services use the same port, so stop the current service before testing the other model.

```bash
cd /home/dxh/DXH/code/vllm-main
source vllm-env/bin/activate

export VLLM_USE_MODELSCOPE=True
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME="$(dirname "$(dirname "$(find "$VIRTUAL_ENV" -name nvcc -type f | head -1)")")"
```

Start the FP16 model:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --gpu-memory-utilization 0.80 \
  --port 8000
```

Start the AWQ model:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct-AWQ \
  --gpu-memory-utilization 0.80 \
  --port 8000
```

Start the PTQ AWQ model:

```bash
vllm serve ./Qwen2.5-1.5B-Instruct-AWQ-custom \
  --served-model-name qwen2.5-1.5b-awq-custom \
  --gpu-memory-utilization 0.80 \
  --port 8000
```

Start the QAT-exported AWQ model:

```bash
vllm serve ./Qwen2.5-1.5B-Instruct-QAT-AWQ \
  --served-model-name qwen2.5-1.5b-qat-awq \
  --gpu-memory-utilization 0.80 \
  --port 8000
```

In the CUDA 11.8 environment used for this project, keep `VLLM_USE_FLASHINFER_SAMPLER=0` to avoid the unsupported `--compress-mode=size` compiler option.

`--gpu-memory-utilization 0.80` limits vLLM's GPU memory usage and leaves room for the KV cache and runtime overhead. The AWQ model already contains its quantization configuration, so `--quantization awq` is normally not required.

### PTQ AWQ quantization command

AWQ is post-training quantization (PTQ), not language-model retraining. Run the offline quantization in the separate quantization environment:

```bash
cd /home/dxh/DXH/code/vllm-main
source quant-awq-env/bin/activate
uv pip install llmcompressor transformers datasets

quant-awq-env/bin/python \
  tools/quantization/quantize_qwen_awq.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output ./Qwen2.5-1.5B-Instruct-AWQ-custom \
  --calibration-file /home/dxh/DXH/code/vllm-main/calibration.jsonl \
  --calibration-samples 512 \
  --max-seq-length 2048
```

`calibration.jsonl` must contain one JSON object per line with a non-empty `text` field. The resulting checkpoint uses `compressed-tensors`, 4-bit `W4A16`, `group_size=128`, and keeps `lm_head` unquantized.

### QAT 4-bit training and export

The repository also provides a memory-friendly LoRA-QAT path. The FP16 base
weights remain frozen, LoRA parameters are trained while the forward pass uses
4-bit fake quantization, and the merged result is exported as a vLLM-compatible
`compressed-tensors` W4A16 checkpoint. This is QAT training followed by AWQ
export; it is not full-parameter retraining.

```bash
cd /home/dxh/DXH/code/vllm-main
source quant-awq-env/bin/activate

quant-awq-env/bin/python \
  tools/quantization/qat_qwen_awq.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output ./Qwen2.5-1.5B-Instruct-QAT-AWQ \
  --calibration-file /home/dxh/DXH/code/vllm-main/calibration.jsonl \
  --calibration-samples 512 \
  --max-seq-length 512 \
  --steps 200 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --lora-rank 8 \
  --group-size 128
```

Use a small `--steps` value first as a smoke test. The default LoRA-QAT
configuration is intended for a 12GB GPU; increase steps only after checking
loss and available memory.

### Inference commands

The model name in each request must match the model currently loaded by the server.

FP16 inference:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{
      "role": "user",
      "content": "Please introduce Beijing's history, culture, and modern development in about 300 Chinese characters."
    }],
    "temperature": 0.0,
    "max_tokens": 512
  }'
```

AWQ inference:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
    "messages": [{
      "role": "user",
      "content": "Please introduce Beijing's history, culture, and modern development in about 300 Chinese characters."
    }],
    "temperature": 0.0,
    "max_tokens": 512
  }'
```

PTQ AWQ inference:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-1.5b-awq-custom",
    "messages": [{
      "role": "user",
      "content": "Please introduce Beijing's history, culture, and modern development in about 300 Chinese characters."
    }],
    "temperature": 0.0,
    "max_tokens": 512
  }'
```

To prevent responses from being too short, vLLM's `min_tokens` extension parameter can be used together with a reasonable `max_tokens` value. Avoid setting an unnecessarily high minimum for long-form AWQ generation, because it may amplify repetition:

```json
{
  "max_tokens": 512,
  "min_tokens": 200
}
```

### Benchmark and quality evaluation commands

The repository scripts measure GPU memory, single-request decoding speed, 32-request batch throughput, and perplexity (PPL). They support `fp16`, `awq`, `custom`, and `qat`; run each model separately so that multiple models do not occupy the GPU at the same time:

```bash
cd /home/dxh/DXH/code/vllm-main
source vllm-env/bin/activate
export VLLM_USE_MODELSCOPE=True
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm-env/bin/python test/quant/bench_one.py \
  fp16 test/quant/result_fp16.json
vllm-env/bin/python test/quant/bench_one.py \
  awq test/quant/result_awq.json
vllm-env/bin/python test/quant/bench_one.py \
  custom test/quant/result_custom.json
vllm-env/bin/python test/quant/bench_one.py \
  qat test/quant/result_qat.json

vllm-env/bin/python test/quant/ppl_one.py \
  fp16 test/quant/ppl_fp16.json
vllm-env/bin/python test/quant/ppl_one.py \
  awq test/quant/ppl_awq.json
vllm-env/bin/python test/quant/ppl_one.py \
  custom test/quant/ppl_custom.json
vllm-env/bin/python test/quant/ppl_one.py \
  qat test/quant/ppl_qat.json
```

These scripts create a vLLM engine directly and do not require an HTTP server to be running.

### Inference optimization results

Test conditions: RTX 3060 12GB, vLLM 0.30.0, CUDA Graph enabled, and `gpu_memory_utilization=0.80`.

| Metric | FP16 | Official AWQ 4-bit | PTQ AWQ 4-bit | QAT AWQ 4-bit |
| --- | ---: | ---: | ---: | ---: |
| Weight file size | 2.9 GB | 1.6 GB | 1.06 GiB | 1.06 GiB |
| vLLM model loading memory | Not recorded separately | Not recorded separately | 1.12 GiB | 1.12 GiB |
| Single-request TTFT | 13 ms | 8 ms | 9 ms | 8 ms |
| Single-request decoding | 79.8 tok/s | 185.1 tok/s | 205.4 tok/s | 207.1 tok/s |
| Throughput, 32 concurrent requests | 1754 tok/s | 3486 tok/s | 3982.9 tok/s | 3940.4 tok/s |
| PPL, 8 Chinese/English texts and 401 tokens | 4.027 | 4.843 | 4.972 | 5.398 |

Results:

- **Speed:** AWQ reduces memory-bandwidth reads through 4-bit weights. On the RTX 3060, single-request decoding is about 2.3x faster and 32-request throughput is about 2x higher than FP16.
- **Memory:** AWQ reduces the weight files from 2.9 GB to 1.6 GB, leaving more room for the KV cache and improving the feasibility of deploying larger models on a 12GB GPU. Since both tests use `gpu_memory_utilization=0.80`, peak allocated memory is close to the reservation limit and does not show the full weight-memory difference.
- **Quality:** FP16 naturally stopped after about 381 tokens for the test prompt. The official AWQ model reached the 512-token limit and showed sentence-level repetition at the end. PPL increased from 4.027 to 4.843, indicating measurable quality degradation from 4-bit quantization. The PTQ model's PPL was 4.972.
- **PTQ model:** The PTQ AWQ model completed quantization, saving, vLLM loading, and the full benchmark. Its TTFT was 9 ms, decoding speed was 205.4 tok/s, 32-request throughput was 3982.9 tok/s, and PPL was 4.972. The startup log reported about 1.12 GiB for model loading and about 279,920 KV-cache tokens.
- **PTQ versus official AWQ:** The PTQ model was about 11% faster for single-request decoding and about 14% faster in 32-request throughput, but TTFT was 1 ms higher and PPL was about 2.7% higher. These are local results from different runs and are reference values rather than cross-machine guarantees.
- **QAT model:** The QAT AWQ model completed vLLM loading and the full benchmark. Its TTFT was 8 ms, decoding speed was 207.1 tok/s, 32-request throughput was 3940.4 tok/s, and PPL was 5.398; its weight file was 1.06 GiB and model loading used 1.12 GiB.
- **Storage and loading memory:** PTQ AWQ and QAT AWQ are both exported in the same `compressed-tensors` W4A16 format, so both have a 1.06 GiB weight file and use 1.12 GiB for vLLM model loading. QAT improves adaptation to quantization error during training; it does not further reduce the final 4-bit checkpoint size.
- **QAT versus PTQ:** QAT was about 0.8% faster for single-request decoding but about 1.1% lower in 32-request throughput. Its PPL increased from 4.972 to 5.398, so this run did not show a quality or throughput advantage over PTQ.
- **Recommended use:** Prefer FP16 for a 1.5B model when it fits on a 12GB GPU. Prefer official or PTQ AWQ when memory is constrained or throughput is the priority. The QAT path is currently useful for validating quantization-aware training rather than as the best measured deployment variant.

Full report and raw results:

- [Quantization comparison report](test/quant/REPORT.md)
- [FP16 benchmark result](test/quant/result_fp16.json)
- [AWQ benchmark result](test/quant/result_awq.json)
- [FP16 PPL result](test/quant/ppl_fp16.json)
- [AWQ PPL result](test/quant/ppl_awq.json)
- [PTQ AWQ benchmark result](test/quant/result_custom.json)
- [PTQ AWQ PPL result](test/quant/ppl_custom.json)
- [PTQ AWQ quantization script](tools/quantization/quantize_qwen_awq.py)
- [QAT training and export script](tools/quantization/qat_qwen_awq.py)

### Raw result summary

The following values are taken directly from the JSON result files. The raw JSON results and the representative values in the report come from different benchmark runs, so their absolute speeds differ slightly while the conclusions remain the same.

FP16 raw benchmark:

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

AWQ raw benchmark:

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

PTQ AWQ raw benchmark:

```json
{
  "model": "./Qwen2.5-1.5B-Instruct-AWQ-custom",
  "gpu_peak_mib": 10481.6,
  "single": {
    "out_tokens": 415,
    "ttft_s": 0.009,
    "e2e_s": 2.025,
    "decode_tps": 205.4
  },
  "batch": {
    "num_requests": 32,
    "elapsed_s": 1.93,
    "total_out_tokens": 7700,
    "throughput_tok_s": 3982.9
  }
}
```

QAT AWQ raw benchmark:

```json
{
  "variant": "qat",
  "model": "./Qwen2.5-1.5B-Instruct-QAT-AWQ",
  "gpu_peak_mib": 10352.1,
  "single": {
    "out_tokens": 512,
    "ttft_s": 0.008,
    "e2e_s": 2.476,
    "decode_tps": 207.1
  },
  "batch": {
    "num_requests": 32,
    "elapsed_s": 1.92,
    "total_out_tokens": 7561,
    "throughput_tok_s": 3940.4
  }
}
```

Raw PPL results:

| Model | Evaluation texts | Evaluation tokens | PPL |
| --- | ---: | ---: | ---: |
| `Qwen/Qwen2.5-1.5B-Instruct` | 8 | 401 | 4.027 |
| `Qwen/Qwen2.5-1.5B-Instruct-AWQ` | 8 | 401 | 4.843 |
| `./Qwen2.5-1.5B-Instruct-AWQ-custom` | 8 | 401 | 4.972 |
| `./Qwen2.5-1.5B-Instruct-QAT-AWQ` | 8 | 401 | 5.398 |

With the same Beijing history, culture, and modern development prompt, FP16 naturally stopped after 381 tokens. AWQ reached the 512-token limit and ended with sentence-level repetition such as “Beijing's history, culture, and modern development are intertwined...”. This is consistent with AWQ PPL increasing from 4.027 to 4.843.

The PTQ AWQ model used the same prompt, concurrency, and sampling parameters. Compared with official AWQ, its decoding speed was 205.4 versus 185.1 tok/s and its 32-request throughput was 3982.9 versus 3486 tok/s; its PPL was 4.972 versus 4.843. The QAT AWQ model measured 8 ms TTFT, 207.1 tok/s decoding, 3940.4 tok/s at 32-request throughput, and 5.398 PPL. Compared with PTQ, QAT was about 0.8% faster in single-request decoding but about 1.1% lower in batch throughput, with higher PPL. Because the four measured model runs were not performed in the same second, absolute speeds should be treated as local reference measurements rather than performance guarantees across machines.

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
