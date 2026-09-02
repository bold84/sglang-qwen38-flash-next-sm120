# SGLang Qwen3.8-Flash-Next SM120

Reproducible SGLang container source for the ModelOpt NVFP4
[`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
checkpoint on two NVIDIA RTX PRO 6000 Blackwell GPUs (SM120).

This is an experimental `v0.1.0-rc.11` source release. The NVFP4 candidate
has passed dual-SM120 startup, text, reasoning, tool-call, decode,
cold-prefill through native 262K context, and GSM8K screening; its
interleaved same-night validation is recorded in [`BENCHMARKS.md`](BENCHMARKS.md).
C32 is explicitly unsupported by the tested token-pool capacity rather than
reported as a timing result. The five-run publication panel on this page
refers to the rc.9 FP8 profile; the NVFP4 publication panel is pending, so
there is no stable or cross-hardware performance claim yet.

## Tested configuration

- Linux/amd64 and 2x RTX PRO 6000 Blackwell (SM120).
- TP=2 and EP=2. TP2/EP1 is invalid for the checkpoint's 640-wide experts.
- ModelOpt NVFP4 experts served by the FlashInfer CUTLASS fused-MoE runner
  with a measured per-GPU tactic cache; dense projections remain BF16.
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

## Performance result

The frozen DSV4F publication workload completed five valid repetitions at each
supported decode concurrency on one unchanged server process. It used 16,384
requested input tokens, 4,096 forced output tokens, temperature zero, and fixed
seeds.

| C | Forward/s | Forced output tok/s | ITL | Output/forward/request |
|---:|---:|---:|---:|---:|
| 1 | 74.355 | 196.3 | 5.155 ms | 2.611 |
| 2 | 61.806 | 336.6 | 6.150 ms | 2.746 |
| 4 | 47.059 | 508.4 | 8.123 ms | 2.664 |
| 8 | 33.163 | 758.8 | 11.030 ms | 2.876 |
| 16 | 23.573 | 1,062.1 | 17.010 ms | 2.808 |

Five cache-busted cold-prefill requests per cell produced 9,428 prompt tok/s at
8K, 11,392 at 32K, 11,543 at 64K, and 11,252 at 128K. These are controlled
fixed-window engineering results, not expected application throughput. The
full method, DSV4F comparison, capacity disposition, and exact aggregate are in
[`BENCHMARKS.md`](BENCHMARKS.md) and
[`evidence/v0.1.0-rc.9/publication-summary.json`](evidence/v0.1.0-rc.9/publication-summary.json).

## Build and run

This repository does not contain model weights or a published GHCR image. Build
the exact source bundle locally:

```bash
docker build --platform linux/amd64 -f Containerfile \
  --build-arg IMAGE_SOURCE=https://github.com/ormandj/sglang-qwen38-flash-next-sm120 \
  --build-arg IMAGE_SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t sglang-qwen38-flash-next-sm120:v0.1.0-rc.11 .
```

Then point the launcher at the downloaded NVFP4 snapshot and a persistent,
image-specific compilation cache:

```bash
export MODEL_DIR=/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
export CACHE_DIR=/srv/cache/sglang-qwen38-flash-next-sm120-v9
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

Only SM120 has been tested. This release does not claim SM121, arm64, FP8,
1M YaRN, or HiCache compatibility.
