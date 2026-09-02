# Run v0.1.0-rc.11 (NVFP4)

## Requirements

- Linux/amd64 with the NVIDIA driver and Container Toolkit installed.
- Two NVIDIA RTX PRO 6000 Blackwell GPUs (SM120).
- The ModelOpt NVFP4 checkpoint at revision
  `7b719225242aacd3dbd3f9407468c2ee9a9d2594`.
- A persistent directory for compiled kernels and measured tactic caches.

Download the model separately; weights are not distributed here:

```bash
hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --revision 7b719225242aacd3dbd3f9407468c2ee9a9d2594 \
  --local-dir /models/RadixArk/Qwen3.8-Flash-Next-NVFP4
```

Build the image from the repository root:

```bash
docker build --platform linux/amd64 -f Containerfile \
  --build-arg IMAGE_SOURCE=https://github.com/ormandj/sglang-qwen38-flash-next-sm120 \
  --build-arg IMAGE_SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t sglang-qwen38-flash-next-sm120:v0.1.0-rc.11 .
```

Start the tested profile:

```bash
export MODEL_DIR=/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
export CACHE_DIR=/srv/cache/sglang-qwen38-flash-next-sm120-v9
export IMAGE=sglang-qwen38-flash-next-sm120:v0.1.0-rc.11
./examples/serve-qwen38-flash-next.sh
```

The launcher selects TP2/EP2, native 262,144 context, PLE host offload,
FlashInfer GDN prefill/decode, reasoning and tool parsers, and NEXTN MTP-3.
Set `NEXTN=0` only for a non-speculative diagnostic. MTP-2 was measured and
rejected because its higher forward rate did not offset lower accepted output.

The launcher pins NCCL to the Simple protocol with 16 channels and 256 threads;
on this dual-GPU PCIe topology that measured 3.5-5.3% lower prefill TTFT than
NCCL's auto-selected LL protocol for the 41.9 MB per-layer all-reduce payloads.
It also enables `SGLANG_FLASHINFER_AUTOTUNE_EXTEND=1`, which tunes the
FlashInfer CUTLASS MoE tactics up to the 16,384-token extend ceiling during
the warmup pass. The first start compiles architecture- and shape-specific
kernels and seeds the measured per-GPU tactic cache from the image; with a
cold cache the extend-bucket autotune adds a few minutes to warmup. Keep
`CACHE_DIR` unique to cache schema v9 and persistent across restarts. HiCache
is deliberately absent; do not point hierarchical caching at the model or
compiled-kernel directories.

Check readiness and the served model:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
```

Send a chat request:

```bash
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen38-flash-next-sm120",
    "messages": [{"role": "user", "content": "What is 17 times 23?"}],
    "temperature": 0,
    "max_tokens": 256
  }'
```

The launcher intentionally rejects other TP/EP layouts and context lengths so
an unqualified experiment is not mistaken for the tested envelope.
