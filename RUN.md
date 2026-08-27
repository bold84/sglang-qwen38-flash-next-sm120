# Run v0.1.0-rc.9

## Requirements

- Linux/amd64 with the NVIDIA driver and Container Toolkit installed.
- Two NVIDIA RTX PRO 6000 Blackwell GPUs (SM120).
- The official FP8 checkpoint at revision
  `bcd9f01ddc9cff2316eb84281bebcd5b058bddce`.
- A persistent directory for compiled kernels.

Download the model separately; weights are not distributed here:

```bash
hf download Qwen/Qwen3.8-Flash-Next-FP8 \
  --revision bcd9f01ddc9cff2316eb84281bebcd5b058bddce \
  --local-dir /models/Qwen/Qwen3.8-Flash-Next-FP8
```

Build the image from the repository root:

```bash
docker build --platform linux/amd64 \
  --build-arg IMAGE_SOURCE=https://github.com/ormandj/sglang-qwen38-flash-next-sm120 \
  --build-arg IMAGE_SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t sglang-qwen38-flash-next-sm120:v0.1.0-rc.9 .
```

Start the tested profile:

```bash
export MODEL_DIR=/models/Qwen/Qwen3.8-Flash-Next-FP8
export CACHE_DIR=/srv/cache/sglang-qwen38-flash-next-sm120-v8
export IMAGE=sglang-qwen38-flash-next-sm120:v0.1.0-rc.9
./examples/serve-qwen38-flash-next.sh
```

The launcher selects TP2/EP2, native 262,144 context, PLE host offload,
FlashInfer GDN prefill/decode, reasoning and tool parsers, and NEXTN MTP-3.
Set `NEXTN=0` only for a non-speculative diagnostic. MTP-2 was measured and
rejected because its higher forward rate did not offset lower accepted output.

The first start compiles architecture- and shape-specific kernels. Keep
`CACHE_DIR` unique to cache schema v8 and persistent across restarts. HiCache
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
