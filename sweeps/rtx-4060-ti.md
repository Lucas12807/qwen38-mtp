# RTX 4060 Ti 16GB — contributor sweeps and studies

Contributor-authored deep dives for Qwen3.8-27B MTP on the RTX 4060 Ti 16GB (Ada, SM 89, 288 GB/s).

### RTX 4060 Ti 16GB: the slowest single card in the table — draft depth plateaus at n-max 2-3
*by [@CeIest2](https://github.com/CeIest2)*

First single-RTX-4060-Ti row, and the lowest-bandwidth single card in the table (288 GB/s —
the 5060 Ti row sits at 448 GB/s). Runs **quimmedes/Qwen3.8-27B-XYZ Q4-XYZ** (v1, 15,194,248,512 B
= 14.15 GiB, sha256 58f6975a5b5707ee…; ~120 MiB larger than the Q4-XYZ-v2 file in the 5060 Ti row) at
**32K context** with a **q4_0 KV cache**.

Host: Ubuntu 24.04.4, driver 580.173.02 (CUDA 13.0), 31 GB RAM. llama.cpp master `9a286ac`
(2026-08-21) built from source with CUDA 12.8, `-DCMAKE_CUDA_ARCHITECTURES=89`. Benched with the
desktop nearly idle (246 MiB GPU footprint, no compositor spill — rule 7).

| n-max | p-min | tok/s (median) | tok/s (mean) | acceptance | VRAM (MiB) |
|---|---:|---:|---:|---:|---:|
| off | — | 17.8 | 17.6 | — | 14,405 |
| 2 | 0.00 | **34.8** | 32.4 | 0.50–0.93 | 15,150 |
| 3 | 0.00 | 34.9 | 32.8 | 0.34–0.88 | 15,284 |
| 4 | 0.00 | 34.4 | 34.5 | 0.31–0.87 | 15,471 |
| 4 | 0.70 | 33.2 | 33.3 | 0.57–0.95 | 15,455 |

Per-prompt medians across the depth sweep (code / prose / bash):

| n-max | code | prose | bash |
|---|---:|---:|---:|
| off | 17.7 | 17.8 | 17.8 |
| 2 | 36.3 | 27.4 | 34.8 |
| 3 | 40.6 | 25.1 | 34.9 |
| 4 | 45.1 | 24.0 | 34.4 |

**Findings:**

1. **Draft depth plateaus where the 5060 Ti keeps climbing.** The 5060 Ti (448 GB/s) gained at
   every step to n-max 4 (50.0 → 53.6 → 59.3); this card is flat from n-max 2 (34.8 → 34.9 →
   34.4, all within noise). The per-prompt split explains it: code keeps paying (36.3 → 40.6 →
   45.1) while prose loses symmetrically (27.4 → 25.1 → 24.0), and bash is flat. At 288 GB/s the
   extra verify cost of deep drafts exactly cancels the code gain. n-max 2 is the sweet spot here,
   matching rule 1's card-dependence.
2. **p-min gating hurts here too, only milder.** The 0.70 gate at n-max 4 raised acceptance
   (0.31–0.87 → 0.57–0.95) but cost 3.5% throughput (34.4 → 33.2) — same direction as the 5060
   Ti's −11.5%, smaller magnitude. Rule 2's "helps starved cards" does not materialize even at
   288 GB/s on this arch; the inversion point sits somewhere below this card, not at it.
3. **n-max 4 fits at 94.5% VRAM on this file** (15,471 / 16,380 MiB) — no OOM, unlike the
   IQ4_XS file on the 5060 Ti. The ~1 GiB gap between this 14.15 GiB file and unsloth's Q4_K_M
   is what buys the whole MTP context; the file fitting is still the load-bearing choice for
   16 GB owners (finding 4 of the 5060 Ti sweep holds).
4. **The baseline scales with bandwidth.** 17.8 tok/s spec-off vs the 5060 Ti's 26.3 on the
   sibling quant — a 0.68 ratio against a 0.64 bandwidth ratio; the baseline is fully
   bandwidth-bound, and the +95% flag gain is the durable part.

**Method:** unchanged `probe.py` at commit `c7bc415`, three runs x three prompts (python merge,
mmap-vs-read, bash watcher), 400 tokens, thinking off, warmup discarded. Both arms `--parallel 1`;
only the spec flags differ (same ngl 999, `-fa 1`, 32K ctx, q4_0 K/V). VRAM via `nvidia-smi`
during serve. Acceptance from llama-server `draft acceptance` log lines (range across the nine
tasks, warmup excluded).

**Honest caveats:**
- **32K context**, not the repo-standard 131K — a Q4-tier file cannot hold 131K on one 16 GB card.
- This is the Q4-XYZ **v1** file, not the v2 of the 5060 Ti row — the two rows are not
  file-identical, so cross-card deltas are bandwidth inferences, not controlled A/Bs.
- Single 3-run passes per arm (probe.py's own medians-of-3); the n2/n3/n4 overall medians sit
  within ~1.5% of each other, i.e. the plateau is real but the ordering of the three arms is noise.
