# Changelog

## v0.1.0-rc.12

- Weight-only FP8 clone of the shared lm_head for the NEXTN draft
  (`SGLANG_DRAFT_FP8_HEAD=1`, wired in the launcher): the draft's 2-3
  full-vocab GEMV passes per decode step drop from 445us to 219us each
  (e4m3 per-row scales, exact e4m3->bf16 conversion, fp32 accumulate,
  single bf16 rounding at the logits store). Decode step time -5.2%
  (C1 step-proxy 12.31 vs 12.99 ms, 4/4 reps, no overlap), accept length
  unchanged (2.83 vs 2.83). The target verify keeps the exact bf16 head,
  so output quality is structurally untouched; gated 5x256 GSM8K at/above
  the same-window control (0.9508 vs 0.9512).
- Patch: sha256 ed26c335..., applies onto upstream main 99b9109553,
  reproduces effective tree f2c0965077.

## v0.1.0-rc.11

- Rebase the release onto the retry branch: sync-free QSA speculative row
  mapping (last stream sync in the spec metadata path) and the HC-mix fused
  prefill up-projection epilogue with the numerics contract fixed (model-
  dtype logits rounding at the F.linear boundary, Inductor-order branch
  reduction); prefill TTFT -3% (8K) / -4.3% (32K) over rc.10's tree on the
  same tactic draw, gated 5x256 GSM8K (above control 5/5).
- Patch: sha256 955f8af1..., applies onto upstream main 99b9109553,
  reproduces effective tree eaa4682c9.

## v0.1.0-rc.10

- Switch the bundle to the Qwen3.8-Flash-Next NVFP4 (ModelOpt FP4) checkpoint
  at `RadixArk/Qwen3.8-Flash-Next-NVFP4@7b719225`.
- Rebase the integration onto refreshed SGLang main `99b9109553` and carry
  the QSA sparse-decode upstream chain plus the ported SM120 optimizations.
- Resolve `modelopt_fp4` to the `flashinfer_cutlass` MoE runner on SM120
  directly from the model config, route its standard prefill dispatch through
  the CUTLASS experts, and add extend-bucket FlashInfer autotuning.
- Ship the measured per-GPU FlashInfer CUTLASS MoE tactic cache (the NVFP4
  analogue of the FP8 Triton tile JSONs) inside the image and seed it into
  `CACHE_DIR` on first start.
- Pin the FlashInfer tactic cache across restarts: the cross-rank agreement
  digest now compares the rank-agnostic tactic multiset instead of raw
  per-rank files (whose keys embed the rank), so a seeded or previously
  tuned cache is no longer silently discarded and re-drawn on every start.
- Record the controlled 5x256 GSM8K gate: the shipped stack is
  accuracy-neutral versus the pre-edit tree (two identical pre-edit server
  instances bracket the candidate within the ~0.5-point server lottery),
  and the fused HC-mix up-GEMM epilogue was rejected on a 5/5 accuracy
  regression despite its 1.5x kernel win.
- Reject adaptive speculative decoding (QSA draft cap and adaptive-state
  crashes) and record the remaining candidate dispositions.

## v0.1.0-rc.9

- Publish the first reproducible dual-SM120 Qwen3.8-Flash-Next FP8 source
  bundle.
- Pin the exact SGLang main, Qwen support, SM120 QSA, MTP compatibility,
  FlashInfer 0.6.18 source/cubins, model revision, and integration patch.
- Select TP2/EP2 and NEXTN MTP-3 as the default launcher profile.
- Record passing text, reasoning, tool-call, image-input, matched decode,
  cold-prefill, and full GSM8K validation.
- Publish the five-run supported-concurrency decode and cold-prefill panel,
  retaining C32 as an analyzer-rejected capacity result.
- Keep HiCache, 1M YaRN, NVFP4, SM121, and arm64 outside v0.1 scope.
