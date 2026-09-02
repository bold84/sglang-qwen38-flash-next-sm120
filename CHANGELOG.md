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
- Pin NCCL to Simple/16-channel/256-thread in the launcher for the dual-GPU
  PCIe all-reduce path.
- Record the NVFP4 interleaved same-night validation: prefill TTFT -6.5 to
  -7.9% across 8K-262K, C16 output +3.8 to +8.1%, C1 step-time neutral, and
  GSM8K unchanged within sampling noise.


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
