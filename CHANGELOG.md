# Changelog

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
