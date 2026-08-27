# SGLang Qwen3.8-Flash-Next SM120

Reproducible SGLang container source for the official
[`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)
checkpoint on two NVIDIA RTX PRO 6000 Blackwell GPUs (SM120).

This is an experimental `v0.1.0-rc.9` source release. The exact candidate has
passed dual-SM120 startup, text, reasoning, tool-call, image-input, decode,
cold-prefill, and full GSM8K checks. Near-native-context, non-speculative, and
uniform publication-performance panels have not been completed, so there is no
stable or cross-hardware performance claim yet.

## Tested configuration

- Linux/amd64 and 2x RTX PRO 6000 Blackwell (SM120).
- TP=2 and EP=2. TP2/EP1 is invalid for the checkpoint's 640-wide experts and
  128-column FP8 quantization blocks.
- Native 262,144-token context.
- NEXTN MTP-3: three steps, top-k 1, four draft tokens.
- FlashInfer GDN prefill/decode and QSA sparse attention.
- The 51B-parameter PLE table offloaded to pinned host memory.
- Text, image input, separated reasoning, and tool calls.
- HiCache disabled. A dedicated, bounded HiCache volume is deferred to v0.2.

The tested Kubernetes container used about 248.5 GiB of host memory after
qualification on a 512 GiB server. That is an observed working set, not a
requirement for an additional 320 GiB reserve.

## Quality result

The complete 1,319-question GSM8K test set ran at concurrency 16, temperature
zero, and seed 42. The model answered 1,264 correctly: **95.83%**, with zero
request errors and zero 16,384-token response-cap hits.

Qwen did not emit GSM8K's literal `####` answer marker, so the pinned AIPerf
grader marked every response `unparsed` and used its documented last-number
fallback. The fallback extracted an answer for all 1,319 responses. See
[`BENCHMARKS.md`](BENCHMARKS.md) for the exact method, provenance, and hashes.

## Build and run

This repository does not contain model weights or a published GHCR image. Build
the exact source bundle locally:

```bash
docker build --platform linux/amd64 \
  --build-arg IMAGE_SOURCE=https://github.com/ormandj/sglang-qwen38-flash-next-sm120 \
  --build-arg IMAGE_SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t sglang-qwen38-flash-next-sm120:v0.1.0-rc.9 .
```

Then point the launcher at the downloaded FP8 snapshot and a persistent,
image-specific compilation cache:

```bash
export MODEL_DIR=/models/Qwen/Qwen3.8-Flash-Next-FP8
export CACHE_DIR=/srv/cache/sglang-qwen38-flash-next-sm120-v8
./examples/serve-qwen38-flash-next.sh
```

The launcher defaults to the tested MTP-3 profile. See [`RUN.md`](RUN.md) for
download, runtime, API, and diagnostic details.

## Reproducibility

[`stack.lock.json`](stack.lock.json) records the immutable CUDA 13 base,
SGLang main and effective trees, Qwen and SM120 integration heads, FlashInfer
source and cubin package, model revision, patch hash, and cache schema. Verify
the bundle with:

```bash
./scripts/validate-release.sh
./scripts/validate-docs.sh
./scripts/verify-patches.sh
```

The last command fetches the pinned public Git objects and reproduces both
effective source trees. The model weights remain subject to the Qwen Community
License 1.0; see [`NOTICE.md`](NOTICE.md).

## Scope limits

Only SM120 has been tested. This release does not claim SM121, arm64, NVFP4,
1M YaRN, or HiCache compatibility.
