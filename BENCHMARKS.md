# v0.1.0-rc.9 validation

## Immutable candidate

- Tested image digest:
  `sha256:ce30879b5d473967fe3f0f6947a63efe2ea971dc47fd086ac0db5c54fb0c8387`
- Model revision: `bcd9f01ddc9cff2316eb84281bebcd5b058bddce`
- SGLang main: `7f27bf470824f452a34e866d22ab5e332a23e26f`
- Effective SGLang tree: `1cf4eb136f470e0dba5eed62d107d16e6bc3ed85`
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

Matched exploratory decode and cold-prefill screens passed without request or
server-metric validation errors. They are deliberately not published as a
release performance table because the uniform five-repetition publication
panel has not been run.

Near-native-context, non-speculative, capacity, AgentX, and uniform publication
performance gates remain pending. This repository is therefore an experimental
source release rather than a stable performance or compatibility claim.
