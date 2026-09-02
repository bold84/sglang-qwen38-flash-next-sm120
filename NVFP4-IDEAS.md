# NVFP4 on SM120 — ideas, results, and what to do differently

Status: the NVFP4 work is **parked, not deleted**. This branch
(`qwen38-sm120-clean`) starts from the last pre-NVFP4 commit — the FP8
rc.9 state — and carries only this document. The full NVFP4 effort remains
available on the preserved branches listed under "Key references". This
document records every approach we tried, what the evidence said, and what a
retry should do differently.

Hardware: 2× RTX PRO 6000 Blackwell (SM120, **no NVLink**, cross-NUMA PCIe,
peer DMA ≈ 26.9 GB/s/direction). Model: `RadixArk/Qwen3.8-Flash-Next-NVFP4`
(rev `7b719225`), experts-only NVFP4 W4A4 (ModelOpt `nvfp4_experts_only`
recipe), everything else BF16. The checkpoint's own qualification: GSM8K
97.27 (BF16 band 97.12–97.50 → quantization is ≈ free), AIME26 98.75/100.
**Note the card says "validated on GB300 and B300" — SM120 was never a
validated platform; nobody has published an accuracy number for this quant
outside the converter's own record.**

## Ideas that worked (validated, keep for the retry)

1. **NCCL protocol/channel pins** — `NCCL_PROTO=Simple`,
   `NCCL_MIN/MAX_NCHANNELS=16`, `NCCL_NTHREADS=256`. Root cause: NCCL prices
   the SYS path at 0.1 GB/s in its cost model, picks RING_**LL** for the
   41.9 MB per-layer all-reduce, and LL doubles wire bytes on a
   bandwidth-bound link. Isolated effect (identical tree+cache):
   **−3.5…−5.3% prefill TTFT, +2.9% C16**. Scoped `NCCL_PROTO=allreduce:^LL`
   measured identical on the big AR (1.2961 vs 1.2950 ms) but leaves small
   collectives on LL — slightly safer, zero cost. Microbench harness was a
   2-rank torchrun all-reduce at the exact [8192,2560] bf16 shape.
2. **FlashInfer CUTLASS MoE tactic cache = the NVFP4 analogue of the FP8
   device-name Triton tile JSONs.** The FP8 profile
   (`E=256,N=640,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition,...json`)
   is unreachable under NVFP4: SM120+modelopt_fp4 force-selects
   `flashinfer_cutlass`, and `ModelOptNvFp4FusedMoEMethod.apply` raises for
   the Triton runner. The equivalent per-GPU artifact is the autotune cache
   under `SGLANG_CACHE_DIR/flashinfer/autotune/<fi-ver>/sm120/<hash>/rank_tp*.json`.
   Extend-bucket tuning (`SGLANG_FLASHINFER_AUTOTUNE_EXTEND=1`, tune buckets
   to the 16 384-token ceiling during warmup) + shipping the frozen seed in
   the image gave the remaining **−2…−3% prefill TTFT**. Cache key = sha256
   over model_path|dtype|quant|moe_backend|tp|pp|attn_dp|ep|hf_config|skip_ops
   — validate a replication against a known host key before trusting it.
3. **The rank-agnostic cache-digest fix (real bug, upstream-worthy).**
   `_autotune_cache_digest` hashed each rank's raw `rank_tp{N}` file whose
   *keys embed the rank index*, so tp0-vs-tp1 digests could never agree:
   **every warmup silently discarded the cache and re-tuned from scratch**.
   Consequences: per-restart tactic lottery (±6% C16 throughput, and GSM8K
   draws from 0.28 to 0.94 on the brittle harness), and a shipped seed cache
   that never actually loaded. Fix: digest the `_metadata` stamp + the sorted
   multiset of tactic values (skip `_`-prefixed bookkeeping). Without this,
   ANY seeded/tuned-cache strategy is a no-op.
4. **Methodology that finally produced trustworthy numbers:**
   - Interleaved same-night A/B on one host (machine drift fooled every
     cross-era comparison — a 21:22 baseline was 5–7% unreproducible at 01:00).
   - C1 decode must be compared as **step time (TPOT × accept length)**;
     accept-length sampling noise (2.87–3.10) otherwise masquerades as ±4%
     TPOT swings.
   - Accuracy gate: **5×n≥256 reps, frozen harness, pinned tactics, sign
     test** ("consistently below ⇒ reject"). Characterized noise floor:
     ~0.3–0.5pt within-server, ~0.5pt between identical server instances.
     Two instances of the *same* pre-edit code differed by 0.63pt — never
     conclude from single 128-question runs.

## Ideas that failed (with evidence — do not redo blindly)

- **HC-mix up-GEMM epilogue fusion** (fuse sigmoid·mul·branch-mean into the
  up projection, branch-interleaved weights, Triton M64/N64/K32/w4/s3).
  Standalone kernel was **1.50×** vs cuBLAS+inductor tail (0.304 vs 0.463 ms
  at rows=8192), gave −2.7…−3.7% TTFT — but lost **−3.8pt GSM8K, 5/5 runs**
  vs fusion-off on otherwise identical pinned servers. Rejected under the
  sign-test rule. Lesson: kernel-level bf16 reassociation in a 48-layer
  hyper-connection path can be quality-relevant even when unit tests match
  the fp64 reference at 2e-2. If retried, gate with 5×256 **before**
  believing perf.
- **Adaptive speculative decoding** (`--speculative-adaptive`): default
  ladder [1,3,7]@bs1 violates the QSA compress-ratio-4 draft cap
  (`speculative_num_draft_tokens <= 4` → NotImplementedError); a QSA-safe
  custom ladder (`{"1":[3],"8":[3,1],"32":[1,0],"64":[0]}`) builds runtime
  states then dies with **CUDA illegal memory access** — the hybrid
  QSA/mamba state machinery does not survive multi-width graph rebuilds.
  Dead until upstream fixes adaptive×QSA.
- **DP-attention + flashinfer a2a** (`--dp 2 --enable-dp-attention
  --moe-a2a-backend flashinfer`): +32% C1 TPOT, −5.4% C16. Wrong topology
  for 2-GPU PCIe.
- **HC persistent-mix retune** (CTA/BLOCK sweep, rows 1–16): every variant
  within ~6% of shipped 188-CTA config; ceiling <1% of step time.
- **tcgen05/SM100 kernel ports** (flashinfer PR4266 split-K bf16 GEMM,
  cutedsl bf16 backend): require SM100/SM103 MMA that SM120 lacks. The
  dense bf16 GEMMs run SM80-era cuBLAS fallbacks and are still ~1.26 ms/step
  addressable — but only via an SM120-native kernel (see untested ideas).
- **Custom all-reduce v2 / torch-symm-mem / mscclpp / flashinfer AR fusion /
  quant-comms**: all arch-gated off on SM120 (capability-12 missing from
  size tables, NVLink-only Lamport paths, ROCm/NPU-only). Verified in source;
  don't spend runs.
- **GDN fused decode proj+conv** (`SGLANG_ENABLE_GDN_DECODE_FUSED_PROJ_CONV`):
  dormant under NEXTN — target-verify never takes the decode forward mode.
- **Prefill CUDA graphs (breakable)**: measured 1.5–1.9× *slower* (17.6 s vs
  11.6 s at 128K) on this arch. QSA builds host-side sparse metadata per
  forward; deliberately excluded upstream.
- **NCCL channel/thread variants**: 8ch 1.322 / 16ch 1.294 / 24ch 1.296 /
  32ch 1.370 ms on the 41.9 MB AR; nthreads 256 ≈ 128 < 512. 16/256 is the
  optimum; nothing further there.

## What we likely got wrong (honest assessment)

1. **Accuracy attribution was late.** Single n=128 runs with a harness whose
   stop-string (`\n\nQ:`) interacted with a style-flip mode (model exits the
   few-shot CoT register → instant EOS → empty completion) produced a
   0.28–0.94 scatter that we initially chased as code regressions. The real
   culprit chain was tactic-lottery + brittle harness. Freeze the harness
   (persist every completion!) and pin tactics from run one.
2. **Official-protocol gap never explained.** Full 1319-question run under
   the converter's protocol (chat template, t0.6/p0.95, 8 192 cap, 16
   threads) scored **93.56%** vs the official **97.27%** on our stack.
   Binomial noise is ±0.65pt, so the −3.7pt is real, split unknown between
   (a) grader differences (their sgl-eval vs our `####`-else-last-number;
   several failures were classic extraction-trap questions) and (b) SM120
   serving-stack numerics (unvalidated platform). **A retry must persist all
   responses and split extraction-vs-model misses on day one.**
3. **Shipped-too-early release attempt**: cutting rc.10 with an accuracy
   story that was still moving. Get the official-protocol
   gap to ≤1pt and explained before shipping anything.

## Untested ideas (ranked, for the retry)

1. **`TORCH_BLAS_PREFER_CUBLASLT=1` (+ `CUBLASLT_WORKSPACE_SIZE`)** — the
   decode profile's #1 candidate, never run: cuBLAS picks SM80-wmma
   fallbacks for every skinny bf16 GEMM; measured 3.38 ms/step vs a
   2.12 ms weight-traffic floor ⇒ **up to +9% C1 decode** for an env var.
   Test first.
2. **QSA draft-extend host sync removal** —
   `qwen_sparse_attn_backend._speculative_row_to_request` does
   `int(repeats.sum().item())` per draft-extend: 76–182 ms full-stream sync
   per 8K chunk, currently hidden behind queued work. Latent but real;
   `extend_seq_lens_cpu` already exists as the CPU-mirror idiom.
3. **QSA chunk-prefill BLOCK_N table** — non-H20 devices inherit a
   Hopper-era `_L20_CONFIGS` table; (16,1,2) at 8 192 tokens; sweep
   (32,4,2)/(32,4,3)/(64,2,2). Ceiling ~4–6 ms/chunk (~1%).
4. **NCCL_P2P_LEVEL=SYS + NCCL_CUMEM_ENABLE=1** (must be pre-set in the
   launching shell — SGLang clobbers it otherwise): flips transport from
   host-staged SHM to P2P/CUMEM. Big-message upside ≈0 (both at the wire
   floor) but removes proxy hops for small collectives. Low priority.
5. **Tactic-draw selection for accuracy, not just speed.** Evidence that
   different valid tactic draws correlate with different greedy behavior
   (style-flip modes). If the official-protocol gap turns out
   stack-numerics-related, try pinning tactics to the draw that reproduces
   the official GSM8K band, or disable MoE autotune entirely
   (`--disable-flashinfer-autotune`, heuristic tactic 0) and re-measure.

## Key references for the retry

- Model: `/root/autodl-tmp/models/Qwen3.8-Flash-Next-NVFP4` (kept; 137 GB),
  revision `7b719225242aacd3dbd3f9407468c2ee9a9d2594`; qualification notes
  and raw metric JSONs ship inside the model dir.
- GSM8K test set: `hf download openai/gsm8k --config main --repo-type
  dataset` (test split, 1 319 rows; a local copy lives under
  `cache/sglang-qwen38-flash-next-sm120/datasets/gsm8k/`).
- Parked sglang work: branch `qwen38-sm120-optimizations-nvfp4`, tip
  `c8306b7305` (base `99b9109553` = upstream main; 13 commits: Qwen/QSA
  cherry-picks, SM120 perf port, NVFP4 enablement `0ae7c39bf3`, autotune
  digest fix). The two commits worth resurrecting first are the **digest
  fix** and the **EXTEND autotune** feature — see
  `python/sglang/srt/model_executor/runner/flashinfer_autotune.py`,
  `arg_groups/overrides.py`, `moe_runner/flashinfer_cutlass.py`.
- Parked release attempt: release repo branch
  `qwen38-sm120-optimizations-nvfp4`, tip `b23c71f` (commits `f73a24b` +
  `b23c71f`); the launcher NCCL env pins, seeded-cache layout, and
  container pins live in that diff (`examples/serve-qwen38-flash-next.sh`,
  `Containerfile`, `stack.lock.json`, `assets/flashinfer-autotune/`).
- Profile findings (decode): step anatomy 13.3 ms (VERIFY 79.5%), dense bf16
  GEMM 3.38 ms/step, lm_head 1.75 ms (82–86% of peak), MoE fp4 91% of peak,
  graph coverage already 96% — launch count is not a lever.
- FP8 precedent: per-GPU Triton tile JSONs under
  `sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/`;
  drop-in selector `SGLANG_MOE_CONFIG_DIR` (FP8 only).
