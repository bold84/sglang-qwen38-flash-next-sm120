# v0.1.0-rc.9 validation

## Immutable candidate

- Tested image digest:
  `sha256:ce30879b5d473967fe3f0f6947a63efe2ea971dc47fd086ac0db5c54fb0c8387`
- Model revision: `bcd9f01ddc9cff2316eb84281bebcd5b058bddce`
- SGLang main: `7f27bf470824f452a34e866d22ab5e332a23e26f`
- Published-image effective SGLang tree: `1cf4eb136f470e0dba5eed62d107d16e6bc3ed85`
  (the current source patch reproduces `2bf157ed74501598305c55f478025ea1f4d80ec8`;
  its native qualification is recorded below and the image must be rebuilt).
- FlashInfer main: `e4b7fa4b7c3ba5e17286d9c59f2bcf2ca07e0a6d`
- AIPerf: `6ed4823d127b3a6d12c63fb8c2ca5eff13f9ba23`
- Runtime: TP2/EP2, NEXTN 3/1/4, native 262,144 context, HiCache off.

## Functional smoke

The exact candidate passed health, model discovery, text generation, separated
reasoning, forced tool call, and image-input checks. It remained Ready with
zero restarts after the performance and quality screens.

## Full GSM8K

The pinned AIPerf grader ran all 1,319 GSM8K test questions once:

| Setting | Value |
|---|---:|
| Concurrency | 16 |
| Temperature | 0 |
| Seed | 42 |
| Response cap | 16,384 tokens |
| Duration | 609.57 seconds |
| Correct | 1,264 |
| Incorrect | 55 |
| Accuracy | 95.83% |
| Request errors | 0 |
| Response-cap hits | 0 |
| Output-token range | 35-11,398 |

Every response contained non-empty separated reasoning and final output. None
contained GSM8K's literal `####` answer marker. AIPerf therefore labeled all
1,319 records `unparsed` and used its documented last-number fallback; that
fallback extracted a non-empty answer for every response and produced the
1,264/1,319 score.

Raw artifact hashes:

| Artifact | SHA-256 |
|---|---|
| Full `SHA256SUMS` inventory | `3e86a84ef0ea2c8abe72e6c619b8e105fae536512803a63adf4bf9b07082c5e9` |
| Aggregate accuracy CSV | `87daa4d2d11646544926cdf8e000679a9e4c2e83fb898c12e4eb926ca3d60cba` |
| Per-question accuracy JSONL | `25ca13f00dacdb2bcfd12a6f34632149fbdc279dd5798d953d9eac7c9d464a91` |
| Outputs JSON | `03d841549528570bc14baa5297cf9a3a27862ff6cf068df841f0a4ddce0ee35b` |
| AIPerf profile summary | `9321f35acbe798728959c56a153d62210a22b84cc23e490b693516cf56c55739` |

## Performance status

### Five-run decode panel

The exact candidate ran the frozen DSV4F `publication` workload on one
unchanged server process: 16,384 requested input tokens, 4,096 forced output
tokens, temperature zero, the fixed five-seed panel, and five valid repetitions
at every supported concurrency. All 25 C1-C16 repetitions passed the occupancy,
queue, prefill-counter, context-window, duration, and sample-count controls with
zero request errors.

`Forward/s` is the arithmetic-neutral engine clock. `Forced output tok/s`
includes the model's path-dependent MTP acceptance, while ITL is the client
inter-token latency. These fixed-window values are engineering regression
signals rather than expected interactive or application throughput.

| C | Forward/s median | Forced output tok/s median | ITL median | Output/forward/request median |
|---:|---:|---:|---:|---:|
| 1 | 74.355 | 196.3 | 5.155 ms | 2.611 |
| 2 | 61.806 | 336.6 | 6.150 ms | 2.746 |
| 4 | 47.059 | 508.4 | 8.123 ms | 2.664 |
| 8 | 33.163 | 758.8 | 11.030 ms | 2.876 |
| 16 | 23.573 | 1,062.1 | 17.010 ms | 2.808 |

C1 had the largest output-rate variance: 6.60% sample CV. Its forward-rate CV
was only 0.40%, locating that variation in MTP acceptance rather than the
underlying engine clock.

### Cold prefill

The same process completed five cache-busted C1 requests at every frozen
prefill length. All 20 requests passed cold-cache, token-count, and compute
controls with zero errors.

| Target | Prompt tok/s | Median TTFT | Requests |
|---:|---:|---:|---:|
| 8K | 9,428.0 | 871.2 ms | 5 |
| 32K | 11,392.0 | 2,885.7 ms | 5 |
| 64K | 11,542.9 | 5,676.4 ms | 5 |
| 128K | 11,252.0 | 11,605.2 ms | 5 |

### DSV4F reference comparison

The table below compares absolute medians with the retained HiCache-disabled
DeepSeek V4 Flash `0.8.1-rc10` five-run panel from the same frozen AIPerf
revision. This is not a paired geometric comparison.

| C | Qwen forward/s change | Qwen forced-output change | Qwen ITL change |
|---:|---:|---:|---:|
| 1 | +14.2% | -39.4% | +61.8% |
| 2 | +25.2% | -29.3% | +41.7% |
| 4 | +41.0% | -22.4% | +26.3% |
| 8 | +42.4% | -18.4% | +13.4% |
| 16 | +30.9% | -24.6% | +20.9% |

Qwen's forward clock was faster at every supported cell, but its MTP-3
acceptance produced only 2.61-2.88 output tokens per forward per request, so
acceptance-weighted output throughput was lower. Cold-prefill rate was 19.3%,
29.3%, 36.1%, and 41.2% higher at 8K, 32K, 64K, and 128K respectively.

### C32 capacity disposition

C32 is unsupported for the fixed test shape under this runtime envelope. The
client completed 32/32 requests with zero errors, but the requests required
657,006 prompt-plus-output tokens while the server exposed a 580,032-token
pool. At most 26 requests ran concurrently and six normally remained queued.
The canonical analyzer therefore rejected the attempt with
`no samples matched the average-context window`; it is retained as capacity
evidence and is not counted as a performance repetition.

### Evidence

The complete raw bundle's `SHA256SUMS` manifest hashes to
`0797bb93a90b915d29ccacb068f1ca10bd8a0e62c1c87216a1bd889860019f53`.
The exact machine-readable supported-panel aggregate is published at
[`evidence/v0.1.0-rc.9/publication-summary.json`](evidence/v0.1.0-rc.9/publication-summary.json)
and hashes to
`3772dddaf9b0caf6027d09ca084df2862daedbb3c0961c3b8015c19a3b47205c`.

For the published image, near-native-context, non-speculative, AgentX, and
human release-review gates remain pending. It is therefore an experimental
release, not a stable or cross-hardware performance claim.

## Post-image SM120 source optimization

The current archived SGLang patch applies to the same pinned `7f27bf4708` base
and reproduces tree `2bf157ed74501598305c55f478025ea1f4d80ec8`. The source was qualified
natively with the same model revision, TP2/EP2 topology, FlashInfer linear
attention, and NEXTN 3/1/4 envelope. These are single-run engineering results,
not replacements for the five-run published-image panel above.

The optimized path fuses QSA index preparation and compression, keeps QSA
prefill on FlashInfer, fuses GDN projection/convolution, removes PLE,
hyperconnection, and MoE gate host/allocation overhead, and selects measured
SM120 MoE kernel configurations. Replay-based SSM graph optimization remains
disabled because it did not preserve exact GDN state semantics. CUDA-graph
FlashAttention scratch storage is retained per captured capacity so a wider
capture cannot invalidate pointers held by an earlier graph.

| Workload and metric | Original patch | Optimized source | Change |
|---|---:|---:|---:|
| C1 output throughput | 180.74 tok/s | 200.83 tok/s | +11.1% |
| C1 mean TPOT | 5.236 ms | 4.693 ms | -10.4% |
| C1 mean TTFT | 1,202.5 ms | 1,162.8 ms | -3.3% |
| C16 output throughput | 757.19 tok/s | 859.60 tok/s | +13.5% |
| C16 mean TPOT | 17.561 ms | 15.690 ms | -10.7% |
| C16 mean TTFT | 10,640.9 ms | 10,459.0 ms | -1.7% |

Both C1 runs used exactly 16,332 input and 4,096 output tokens. The C16 runs
used 262,140 and 261,308 input tokens respectively (a 0.32% difference) and
65,536 output tokens each. NEXTN acceptance also improved by 7.3%; dividing
output throughput by acceptance length leaves a +3.6% C1 and +5.8% C16
wall-clock forward-rate proxy, so the output-rate gain is not attributed solely
to kernel speed.

The final source passed 224 selected kernel, graph, model, and memory-cache
tests with no failures. A fixed 32-token greedy request produced byte-identical
text and token IDs before and after the optimization. The full-context smoke
generated eight tokens from a 262,000-token prompt in 21.85 seconds, and two
sequential 128K graph-backed requests completed without an illegal-address or
server restart. The launcher now bounds capture at batch size 32; the state
cache still caps this exact runtime to 27 active requests.

## v0.1.0-rc.10 NVFP4 candidate

The rc.10 candidate switches the bundle to the ModelOpt NVFP4 checkpoint
(`RadixArk/Qwen3.8-Flash-Next-NVFP4@7b719225`) on refreshed SGLang main
`99b9109553`, resolves `modelopt_fp4` to the `flashinfer_cutlass` MoE runner
on SM120, tunes the FlashInfer CUTLASS MoE tactics through the 16,384-token
extend ceiling, seeds those measured per-GPU tactics into the image, pins
NCCL to Simple/16-channel/256-thread in the launcher, and pins the tactic
cache across restarts with a rank-agnostic cache digest.

Attribution protocol: earlier same-session arms (baseline at 21:22, then
tree edits, then each env change) were single-run and straddled several hours
of machine state, so absolute cross-era deltas were confounded. The retained
numbers below are interleaved same-night A/B runs on one host: the pre-edit
tree at `687d2f4ab` with default NCCL versus the final candidate tree; the
final-candidate column spans the tactic draws before and after cache pinning.

| Workload and metric | Pre-edit config | rc.10 candidate | Change |
|---|---:|---:|---:|
| Cold TTFT 8K (3 reqs, C1) | 538.5 ms | 496.9-503.7 ms | -6.5 to -7.7% |
| Cold TTFT 32K (3 reqs, C1) | 2,147.0 ms | 1,976.5-2,028.3 ms | -5.5 to -7.9% |
| Cold TTFT 131K (3 reqs, C1) | 9,135.2 ms | 8,457.5-8,638.1 ms | -5.4 to -7.4% |
| Cold TTFT 262K (3 reqs, C1) | 21,305.1 ms | 19,918.6-20,371.3 ms | -4.4 to -6.5% |
| C16 output throughput (16x16K in, 4K out) | 906.1-943.8 tok/s | 911.5-979.3 tok/s | up to +8.1%, draw-dependent |
| C1 verify-step time | 12.93 ms/step | 12.90-12.97 ms/step | neutral |

C1 mean TPOT differences across arms tracked NEXTN acceptance-length sampling
noise (2.87-3.10 across runs) rather than engine time; step time is the
honest C1 signal. Within the candidate, the isolated NCCL pin effect
(identical tree and cache) measured -3.5 to -5.3% prefill TTFT and +2.9% C16
output; the extend-bucket tactic cache contributes the remainder of the
prefill gain. C16 varies with the MoE tactic draw (see the accuracy section):
different valid draws differ by up to ~6% at C16 while remaining
accuracy-equivalent.

### Accuracy gate (controlled, 5x256, frozen harness)

The first accuracy pass used single 128-question runs with an extraction
harness that broke when the model drifted out of the few-shot CoT register,
and servers whose FlashInfer tactic caches were silently re-tuned from
scratch on every start (the cross-rank digest compared raw per-rank files,
whose keys embed the rank index, so it could never agree). Those runs scored
anywhere from 0.28 to 0.94 and are retained only as the incident record.

The gate was then re-run under control: the rank-agnostic digest fix pins the
tactic cache across restarts (cache-file mtime verified unchanged through
every arm), the extraction harness was frozen, and each arm ran five
256-question repetitions:

| Arm | 5-run scores (n=256) | Mean |
|---|---|---:|
| A: pre-edit tree `687d2f4ab` (instance 1) | .8906 .8867 .8906 .8828 .8906 | 0.8883 |
| A2: pre-edit tree (instance 2, same code/env/cache) | .8867 .8789 .8750 .8828 .8867 | 0.8820 |
| B: rc.10 tree, HC fusion off | .8828 .8828 .8789 .8789 .8867 | 0.8820 |
| C: rc.10 tree, HC fusion on | .8477 .8320 .8477 .8438 .8477 | 0.8437 |

A second instance of the identical pre-edit code (A2) differs from the first
(A) by 0.63 points - the same size as the A-vs-B gap - so the server-instance
lottery (~0.5 points) explains B, and the shipped stack is accuracy-neutral
(B equals A2 exactly). The fused HC-mix up-GEMM epilogue (C) is below the
baseline in 5/5 paired runs (-3.8 points, far beyond the lottery) and was
**rejected and removed** despite a measured 1.5x standalone kernel win and a
-2.7 to -3.7% prefill TTFT improvement; the numerics class of its fused
sigmoid/multiply/mean epilogue does not survive contact with this model.

### Rejected candidates

- HC-mix up-GEMM epilogue fusion: 1.5x standalone kernel, -3% prefill TTFT,
  but -3.8 points GSM8K 5/5 - rejected on accuracy (see above).
- Adaptive speculative decoding: the default ladder's 7-step tier violates
  the QSA compress-ratio-4 draft cap, and a QSA-safe ladder crashes with a
  CUDA illegal memory access while building adaptive runtime states - the
  hybrid QSA/mamba state machinery does not survive multi-width graph
  rebuilds. Rejected.
- DP-attention with the FlashInfer all-to-all dispatcher (`--dp 2
  --enable-dp-attention --moe-a2a-backend flashinfer`) regressed C1 TPOT
  +32% and C16 output -5.4% on this two-GPU PCIe host; rejected.
- Retuning the persistent hyper-connection mix kernel's CTA/tile geometry:
  every variant within 6% of the shipped 188-CTA configuration across row
  counts 1-16; ceiling under 1% of step time, rejected as churn.
- Porting the SM100 tcgen05 BF16 split-K GEMM family (FlashInfer PR #4266):
  requires SM100/SM103 tcgen05 MMA that SM120 does not implement; not
  portable without a rewrite.
- Enabling the GDN fused decode projection+convolution path: dormant under
  NEXTN because target-verify never takes the decode forward mode; no ITL
  win available without spec-decode redesign.
- NCCL `allreduce:^LL` scoped protocol versus the global Simple pin:
  byte-identical latency on the 41.9 MB all-reduce (1.2950 vs 1.2961 ms in a
  7x100 microbenchmark); the global pin is kept for simplicity.

These are single-night engineering signals on one host, not the five-run
publication panel; the rc.9 FP8 panel remains the published reference and the
NVFP4 image has not yet been rebuilt for the record.
