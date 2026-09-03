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
  the fp64 reference at 2e-2. Root cause found later the same day by
  forensic recovery of the deleted kernel: not a reassociation accident but
  a single missing bf16 logits rounding before the sigmoid — see the
  Addendum; the idea is one cast line from being retryable. If retried,
  gate with 5×256 **before** believing perf.
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
- **GDN fused decode at draft-extend** (tested 09-03, rejected): widened the
  `is_decode()` gate to include `is_draft_extend_v2()`. Kernel bit-exact,
  1.65× standalone (37 vs 62 µs at 64 rows) — **zero end-to-end**
  (flush-cache C4/128: 21.27/21.82 vs 19.01/21.09 ms). Draft-extend is too
  thin a slice; the graph already overlaps unpack+conv. Reverted. Lesson:
  standalone kernel speedups don't transfer when the block isn't on the
  critical path — profile the tail, not the kernel.
- **Draft-extend LM-head prune** (tested 09-03, parked broken): pruning the
  draft LM head 64→16 rows via `select_index` in the draft-extend graph
  runner collapses accept 2.9→1.0 (correct outputs via target-only fallback,
  but all speculation lost). Stashed as WIP in the sglang retry branch;
  needs forensics on why the pruned forward poisons draft logits (select
  rows match the worker formula — the corruption is elsewhere).
  11.6 s at 128K) on this arch. QSA builds host-side sparse metadata per
  forward; deliberately excluded upstream.
- **NCCL channel/thread variants**: 8ch 1.322 / 16ch 1.294 / 24ch 1.296 /
  32ch 1.370 ms on the 41.9 MB AR; nthreads 256 ≈ 128 < 512. 16/256 is the
  optimum; nothing further there.

- **`TORCH_BLAS_PREFER_CUBLASLT=1` + `CUBLASLT_WORKSPACE_SIZE=32768`**
  (tested 09-03, rejected): interleaved same-window A/B on identical
  tree/tactics/cache — C16 step time 47.2–48.5 ms (LT, n=4) vs 47.8–48.3
  (control, n=3): **neutral**. Prefill 8K/32K: 502.5/2 023.7 vs
  503.7/2 028.3 ms: **neutral**. C1 step: 13.90/13.97 (LT) vs 13.61/13.70
  (control): **possible ~2% cost**. The dispatch switch itself works
  (torch 2.13 `preferred_blas_library` flips Cublas→Cublaslt) — cuBLASLt
  simply has no better skinny-M bf16 kernels for SM120. The profile's
  3.38-vs-2.12 ms/step gap in the dense bf16 GEMMs remains real but is not
  reachable via this knob; it needs an SM120-native kernel.

### Environment gotchas on this GPUhub box (09-03)

- **cgroup memory cap = 220 GiB** (host shows 1 TB — `free` lies). Both
  models' page caches resident (173 GB FP8 + 137 GB NVFP4) OOM-kill rank 0
  (SIGKILL, exit −9) at launch. `drop_caches` is permission-denied; evict
  the *other* model's pages with `posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)`
  over its files (172.8 GiB evicted → launch fine). Check
  `/sys/fs/cgroup/memory.current` before big loads.
- **Slow tenant-correlated drift is real**: C1 step time 12.94 ms (03:35) →
  13.66 ms (06:53) across ALL arms (+5.5%) on identical config+cache.
  Same-window interleaving is mandatory on this machine; cross-hour
  comparisons are confounded. Fast run-to-run C16 scatter is NOT drift —
  it is the accept-length lottery (step-time spread ±1.3% when four raw-tput
  replicates span ±4%).

## What we likely got wrong (honest assessment)

1. **Accuracy attribution was late.** Single n=128 runs with a harness whose
   stop-string (`\n\nQ:`) interacted with a style-flip mode (model exits the
   few-shot CoT register → instant EOS → empty completion) produced a
   0.28–0.94 scatter that we initially chased as code regressions. The real
   culprit chain was tactic-lottery + brittle harness. Freeze the harness
   (persist every completion!) and pin tactics from run one.
2. **Official-protocol gap never explained — now localized to the grader,
    not the model.** Full 1319-question runs under the converter's protocol
    (chat template, t0.6/p0.95, 8 192 cap, seed 0, 16 threads) on this
    machine: **NVFP4 93.56% vs FP8 93.63% — one question apart, near-identical
    failure lists** (the same trap questions: robe, Carla, Claire, Shiela,
    Adrien, Billy). The NVFP4 quantization is accuracy-free on our stack.
    The shared −3.7pt vs the official 97.27% sits in the grader/protocol
    layer: the same FP8 checkpoint scores 95.83% (AIPerf, temp 0, rc.9),
    93.63% (our harness, t0.6), 97.27% (sgl-eval, t0.6) — a 3.6pt spread
    across graders on identical weights. A retry should reproduce sgl-eval's
    extraction exactly and persist all responses; expect most of the gap to
    be systematic extraction misses, not model misses. Side result: NVFP4
    served this workload 31% faster than FP8 (967 vs 739 tok/s) at identical
    quality.
3. **Shipped-too-early release attempt**: cutting rc.10 with an accuracy
   story that was still moving. Get the official-protocol
   gap to ≤1pt and explained before shipping anything.

## Untested ideas (ranked, for the retry)

1. ~~`TORCH_BLAS_PREFER_CUBLASLT=1`~~ — **tested 09-03, rejected** (see
   Failed ideas). ~~QSA draft-extend host sync removal~~ — **landed 09-03**
   (`perf(qsa)` @ `3fa2e31651`): wall-neutral at 8K chunks (stall hides
   behind queued work); kept as the last-stream-sync removal.
   ~~HC-fusion one-line fix~~ — **landed 09-03** (`feat(hc-mix)` @
   `4fe45c3dba`): the up-GEMM epilogue is back with the numerics contract
   fixed (fp32 accumulator rounded to the model dtype at the reference
   F.linear boundary before the fp32 sigmoid/mul/mean; HC=4 branch
   reduction pinned to Inductor's sequential order via tl.split). New
   `matches_compiled_chain_semantics` unit test guards the contract and
   self-checks that the unrounded variant fails it. Prefill vs same-draw
   control: 8K 486.7–497.8 vs 504.9–505.9 ms (**−3%**), 32K 1932 vs 2020
   (**−4.3%**). 5×256 GSM8K gate vs fresh-instance control (pinned
   tactics, frozen harness): fused-on 0.9375–0.9492 (mean 0.9453) vs
   fused-off 0.8750–0.8984 (mean 0.8812) — **above control 5/5**; the
   rejection rule (consistently below) does not trigger. The +6.4pt is
   more plausibly register stabilization than kernel gain: the fused
   kernel is deterministic where the torch.compile path it replaces does
   instance-varying inductor autotuning; accepted on the unit-level
   semantics guarantee + the sign test, with the lottery caveat recorded.
2. ~~QSA chunk-prefill BLOCK_N table~~ — **tested 09-03, rejected.** All
   five alternatives [(32,4,2), (32,4,3), (64,4,2), (64,2,2), (32,8,2),
   (64,8,2)] SLOWER than the inherited (16,1,2) at 8K rows (0.60–0.85 vs
   0.59 ms) AND numerically different (maxdiff ~1.6 — BLOCK_N changes
   per-row token coverage, not a free knob). Table stands.
3. **chunked-prefill-size 16384** — **landed 09-03 (evening), after two
   false verdicts.** First pass vs archived baselines showed −27%/−43%
   TTFT (stale-baseline drift); the first same-window control reversed it
   — because those runs lacked `--flush-cache`, so radix replays diluted
   and boundary-shifted the signal. The decisive A/B/A/B with flush-cache
   (raw prefill): 16K wins uniformly — 8K 478/478 vs 490/479, 32K
   1874/1862 vs 1932/1936, 128K 8016/8026 vs 8316/8321 ms (**−3.5%** at
   32K–128K), decode step-proxy unchanged (12.32 vs 12.31). Quality: the
   5×256 gate looked −0.7pt low (0.9437 vs pooled 0.9508) but the full
   1319-question official protocol says neutral (0.9333 vs 0.9318) —
   256-question gates carry ±0.7pt subsample noise on this harness.
   Lessons: (a) prefill A/Bs MUST flush cache or radix replay pollutes
   TTFT; (b) borderline 256-question gates resolve with the 1319 official
   run before rejecting. Pinned tactic cache already covers the 16384
   bucket (EXTEND tuning ceiling), no retune needed.
4. ~~Triton skinny split-K GEMM for decode dense projections~~ — **tested
   09-03, rejected.** GPU-only profiling showed the in-server cuBLAS picks
   (wmma 16x16/32x32) already stream the fused in_proj at ~1.54 TB/s and a
   tuned Triton split-K table lost everywhere (0.3–0.99×) on the exact
   decode shapes; F.linear hits 1.5–1.8 TB/s = bandwidth roofline. The old
   "3.38 vs 2.12 ms/step" gap estimate did not survive contact with shapes.
   Decode dense GEMM is weight-bandwidth-bound at peak; only weight
   quantization moves it (checkpoint territory, out of scope).
5. ~~HC persistent-mix tile retune (rows≤16)~~ — **tested 09-03,
   rejected.** An idle+interleaved rerun showed base ≈ variant at rows
   4/8/12/16 (12.3–13.6 µs); the apparent 27% win was server-traffic
   contamination of the microbench. Microbenches MUST run with the server
   idle (or GPU-isolated); one contaminated sweep produced a phantom.
6. **FP8 draft lm_head (weight-only, draft only)** — **landed 09-03**,
   `SGLANG_DRAFT_FP8_HEAD` (default off). The NEXTN draft's shared lm_head
   (2 draft GEMVs + draft-extend per step, 445 µs each at bf16, M=1–64)
   gets an e4m3 per-row-scaled clone; e4m3→bf16 conversion is exact, fp32
   accumulate, one bf16 rounding at the logits store (same boundary as the
   reference). 219 µs in-graph (2.04×), step-proxy C1 12.31 vs 12.99 ms
   (**−5.2% decode**, 4/4 reps, no overlap), accept length identical
   (2.83 vs 2.83). Target verify keeps the exact bf16 head, so output
   quality is structurally untouched. GSM8K 5×256 gate: 0.9492/0.9531/
   0.9375/0.9570/0.9570 (mean 0.9508) vs same-window bf16 control
   0.9453/0.9570 — at/above band, accepted.
   Files: `srt/layers/draft_fp8_head.py`, override in
   `models/qwen4_exp_mtp.py::set_lm_head_from_target`.
7. ~~Decode-shape MoE tactic retune~~ — **tested 09-03, rejected.**
   Standalone `cutlass_fused_moe` sweep with `profile_ids` override
   (TP1 proxy, EP-shaped weights, Swiglu) found [17,56] 7% faster than
   pinned [19,57] at rows=16, and [17,57] 4% faster than the rows=128
   bucket's [17,49] at rows=80. In-server A/B (flush-cache C4/128):
   retuned 23.60/23.34 vs control 19.01/21.09 ms — **proxy ranking did
   not transfer** (EP2 routing, real weight distribution, and graph
   capture change the picture). Pinned cache restored from backup. Lesson:
   in-situ autotune beats proxy benchmarks; don't hand-edit the tactic
   JSON without a serving gate.
8. **NCCL_P2P_LEVEL=SYS + NCCL_CUMEM_ENABLE=1** (must be pre-set in the
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

---

## Addendum — HC-fusion failure root-caused (2026-09-03, read-only forensic recovery; refines the failed-ideas entry)

A forensics pass recovered the deleted epilogue kernel from the Triton
compile cache after source recovery failed (`.pyc` files were regenerated
at restoration; a read-only dangling-blob scan found no copy). Recovery:
`cache/sglang-qwen38-flash-next-sm120/nvfp4-prefill-tuned/triton/FLL6DBYQ…/_hc_mix_up_epilogue_kernel.ttir`
(compiled 02:13:48) plus a second specialization (rows-divisibility
variant, 02:08:55), with test intent recovered from
`work/sglang/test/.pytest_cache/v/cache/nodeids` and the wrapper signature
from `/tmp/bench_hc_mix_up.py`.

1. **No logic bug.** Formula, branch packing (`gemm_n = j*4+b`,
   pre-permuted up-weights `[j,b,k]`, X remap `(n%4)*HS + n//4`), row/K
   masking, fp32 accumulation (`tt.dot` → 64×256 f32, no atomics), and the
   final bf16 store all match the reference structure. Tests had covered
   rows 256/2048/8192/8256 (fp16+bf16) and a small-row gate — no boundary
   defect.
2. **Single primary divergence: a missing bf16 logits rounding.** The
   compiled reference materializes up-projection logits to bf16
   (`F.linear` global write) before the fp32 sigmoid·mul·mean epilogue;
   the fused kernel fed the raw fp32 accumulator straight into `sigmoid`.
   It was *more precise than the reference* — and therefore a numerically
   different model. The lesson upgrades from "bf16 reassociation is risky"
   to: **dtype-boundary fidelity to the validated runtime is the property
   that matters, not accuracy.** `no_less_accurate_than_eager` passes by
   construction; only a `matches_compiled_semantics` test at prefill sizes
   would have caught it — that test existed for the decode path only
   (rows 17/128).
3. **Fix spec if retried (one line, register-only):** after the K loop,
   `logits = acc.to(tl.bfloat16).to(tl.float32)` before the sigmoid.
   Keep layout, fp32 epilogue, bf16 store unchanged; optionally make the
   HC=4 reduction branch-sequential `((b0+b1)+b2)+b3` to match Inductor's
   order. Expected to retain the −2.7…−3.7% TTFT. Gate unchanged:
   5×256, pinned tactics, sign test.
4. **CPU quantification of the skipped cast** (seed-fixed; HC=4, HS=64,
   lowrank=16, rows=8192; 2.1 M gates): 99.94% of gate values change
   (mean |Δ| 1.9e-4, max 8.2e-4); after final bf16 store, 16.8% of mixed
   outputs flip, mostly by 1 ulp. Compounded over ~48 mix sites per token
   into greedy argmax, plausibly the 9–11/256 flips observed (the
   −3.7/−3.8pt A/B). Attribution caveat: every launch in that window
   logged "per-rank caches disagree … tuning from scratch", so tactic
   lottery leaves an unquantified non-cast residual.
