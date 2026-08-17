# Qwen3.8-27B MTP: the flag was free the whole time

One llama.cpp flag unlocks +33% to +145% decode speed for Qwen3.8-27B on consumer GPUs, depending on the card, and the knobs the community mapped push further still. No new files, no conversion, no custom build. The MTP head already ships inside the GGUF you downloaded on launch night.

Opened hours after the Aug 14 2026 release. Within three days the community grew it into a living record: **39 configurations, 28 contributors, a decade of silicon from 2016 Pascal to Blackwell workstation cards, three quant makers, and seven tuning rules nobody knew at launch**, see [Community numbers](#community-numbers).

## The numbers

The two founding rows, measured the night of the drop. The full living table (10 machines, all vendors) lives in [Community numbers](#community-numbers).

| Card | Baseline | With the flag | Gain | Acceptance |
|---|---|---|---|---|
| RTX 3090 24GB | 31.0 tok/s | **41.3 tok/s** | **+33%** | 0.76-0.80 |
| RTX 5090 mobile 24GB | 36.7 tok/s | **50.9 tok/s** | **+39%** | 0.76-0.82 |

Paired A/B: same card, same GGUF, same config both sides. Live llama-server with a streaming client, every generated token clocked, warmup discarded, medians of 3 runs x 3 prompts, thinking off, 131K context resident, q4_0 KV cache, unsloth Q4_K_M. These are serve measurements, not llama-bench numbers.

## The flag

```
--spec-type draft-mtp --spec-draft-n-max 2 --parallel 1
```

## Full launch command

```bash
llama-server -m Qwen3.8-27B-Q4_K_M.gguf \
  -c 131072 -ngl 999 -fa 1 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --spec-type draft-mtp --spec-draft-n-max 2 --parallel 1
```

The KV cache flags matter on their own: without them, context creation fails past roughly 90K next to 17GB of weights. With them, the full 262K window fits a 24GB card at 22.2GB (drop `-c` to 262144 and remove the spec flags if you want maximum window instead of maximum speed).

Weights: [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF). Official model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

## Tuning n-max

Swept on the 5090 mobile, same method:

| n-max | Overall median | Code prompts | Prose prompts | Acceptance |
|---|---|---|---|---|
| **2** | **50.9** | 56.4 | 42.5 | 0.76-0.82 |
| 3 | 48.3 | 59.4 | 37.9 | 0.68 |
| 4 | 47.3 | 60.2 | 33.4 | 0.65 |

Acceptance decays as you draft deeper and prose pays for it first. Run 2 as your daily, 3 if your session is pure code.

## The seven rules the community found

Discovered by contributors in the table across the first three days, detailed in the sections and footnotes below:

1. **The n-max sweet spot is card- and topology-dependent.** 24GB cards peak at n-max 2, bigger or faster cards at 3-4, and switching split mode moves it too, so re-sweep after any config change ([@lingster](https://github.com/lingster), [@Jackwwg83](https://github.com/Jackwwg83)).
2. **`--spec-draft-p-min` helps starved cards and hurts fast ones.** The ~0.60 confidence gate makes deep drafting nearly free on bandwidth-poor rigs ([@tomertec](https://github.com/tomertec)), and inverts on desktop Blackwell, where three independent RTX 5090s ran fastest ungated. Acceptance is a vanity metric there: gating raised it and lowered throughput ([@taco-devs](https://github.com/taco-devs), [@paulomcg](https://github.com/paulomcg), [@jcr211](https://github.com/jcr211)). Sweep it, don't adopt it.
3. **The gain scales with generation length where overhead dominates.** Short generations can pay more than they win ([@Spadav](https://github.com/Spadav)); on rigs whose baseline is already bandwidth-bound the full gain shows at 400 tokens and length adds nothing ([@Jackwwg83](https://github.com/Jackwwg83)).
4. **On multi-GPU boxes, fix the split mode before touching spec flags.** The default `--split-mode layer` serializes decode; `tensor` was +68% on its own on a 5060 Ti pair, and the two levers stack to 3.14x ([@Jackwwg83](https://github.com/Jackwwg83)).
5. **Speculative decode is a single-stream optimization.** The advantage is gone by `--parallel 4`, and a `--parallel 2` BASELINE reads ~20% low, which inflates your gain claim, so measure both arms at `--parallel 1` ([@Jackwwg83](https://github.com/Jackwwg83), [@paulomcg](https://github.com/paulomcg)).
6. **Rebuild llama.cpp before you tune anything, and re-derive whole flag stacks per card class.** A config that pays on a bandwidth-starved APU inverted below baseline on a 960 GB/s card ([@Splizard](https://github.com/Splizard)).  Upstream is optimizing this arch weekly: a current build was +10-15% on every quant before any flag ([@taco-devs](https://github.com/taco-devs)), and a fresh 3090 baseline now equals the day-one with-flag number ([@hauntedhost](https://github.com/hauntedhost)).
7. **A shared desktop halves everything, silently.** A live compositor and browser spilled 3.5 GB of weights to host RAM over PCIe while `/health` stayed green; decode halved with no error anywhere. Bench headless, or verify the weights are actually resident (`mem_info_gtt_used` or your vendor's equivalent) before trusting a number ([@Splizard](https://github.com/Splizard)).

## How it works

Qwen trained multi-token-prediction (nextn) layers into Qwen3.8. The quantizers kept them: unsloth's GGUFs carry the `blk.*.nextn.*` tensors, which llama.cpp loads and, without the flag, ignores. llama.cpp added draft-mtp speculative decoding in [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673) (July 2026): the server drafts tokens with the built-in head and verifies them with the main model, so accepted drafts cost a fraction of a full forward pass. Everything was in place on release night. The flag connects it.

## Caveats

- `--parallel 1` for measurement, always: some hosts do serve with higher parallel, but the spec advantage is gone by 4 concurrent streams, and a parallel-2 baseline corrupts the A/B (see rule 5)
- prompt processing takes a small hit from device-to-host embedding transfers
- these are day-one llama.cpp speeds through the qwen3_5 code path, the hybrid attention kernels are young and the floor should rise with upstream work
- your absolute numbers will differ with hardware, drivers, and thermals, the deltas are the durable part

## Measure it yourself

`probe.py` is the streaming client behind every number here. It clocks every generated token (reasoning and content deltas both) against a live server and prints per-prompt medians.

```bash
python3 probe.py                 # defaults to http://127.0.0.1:8080
python3 probe.py http://127.0.0.1:8090
```

Run it once against a baseline serve and once with the flag, same everything otherwise. That pairing is the whole method.

## Community numbers

Ran the A/B on your card? Open a PR and add a row.

| Card | Baseline | With flag | n-max | Acceptance | Contributor |
|---|---|---|---|---|---|
| RTX 3090 24GB | 31.0 | 41.3 | 2 | 0.78 | [@sudoingX](https://x.com/sudoingX) |
| RTX 5090 mobile 24GB | 36.7 | 50.9 | 2 | 0.79 | [@sudoingX](https://x.com/sudoingX) |
| RTX 4090 24GB | 47.7 | 76.3 | 2 | 0.56 | [@Spadav_](https://x.com/Spadav_) |
| RTX A6000 48GB (Ada) | 26.7 | 52.5 | 2 | 0.54-0.98 | [@lingster](https://github.com/lingster) |
| RX 7900 XTX 24GB | 30.7 | 43.9 | 2 | 0.60-0.95 | [@Jqianggu](https://x.com/Jqianggu) |
| 2× RTX 3090 + 3090 Ti 24GB (TP) | 49.1 | 81.1 | 2 | 0.52-0.96 | [@guilhermedemelocabral](https://github.com/guilhermedemelocabral) |
| RTX 4090 24GB (UD-Q4_K_XL) | 36.1 | 74.8 | 2 | 0.56-0.94 | [@rkvhtd](https://github.com/rkvhtd) |
| 2x RX 9070 16GB (Vulkan) | 22.1 | 41.6 | 2 | 0.73 | [@tomertec](https://github.com/tomertec) |
| AMD Radeon AI PRO R9700 32GB | 27.0 | 43.3 | 2 | 0.60-0.94 | [@ajnytebot](https://github.com/ajnytebot) |
| Ryzen AI Max+ 395 / Radeon 8060S | 11.5 | 23.7 | 2 | 0.52-0.94 | [@shiwuxiu](https://github.com/shiwuxiu) |
| AMD Radeon 890M iGPU (Strix Point) 48GB UMA | 2.7 | 5.7 | 2 | 0.59-0.91 | [@davidglogan](https://github.com/davidglogan) |
| RTX PRO 6000 Blackwell 96GB | 63.8 | 91.5 | 2 | 0.72-0.81 | [@commdata2338](https://github.com/commdata2338) |
| RTX 5090 32GB (desktop) | 61.4 | 135.0 | 4 | 0.38-0.87 | [@taco-devs](https://github.com/taco-devs) |
| RTX 5090 32GB (Q6_K, 128K) | 61.9 | 130.0 | 2 | 0.52-0.95 | [@hypertectonic](https://x.com/hypertectonic) |
| RTX 5090 32GB (Q6_K, 256K) | 62.0 | 121.7 | 2 | 0.50-0.95 | [@hypertectonic](https://x.com/hypertectonic) |
| RTX 3090 Ti 24GB (Q4_K_M, 128K) | 42.0 | 60.9 | 2 | 0.47-0.93 | [@hypertectonic](https://x.com/hypertectonic) |
| RTX 3090 Ti 24GB (Q4_K_M, 256K) | 41.2 | 61.4 | 2 | 0.55-0.94 | [@hypertectonic](https://x.com/hypertectonic) |
| RTX 5090 32GB desktop (UD-Q5_K_XL) | 66.3 | 144.2 | 4 | 0.32-0.89 | [@TrickRiggin](https://github.com/TrickRiggin) |
| 3× RTX 3060 12GB (layer split) | 17.3 | 24.5 | 8 | 0.87 | [@EamonMcKiernan05](https://github.com/EamonMcKiernan05) |
| AMD Radeon 9060 XT 16 GB | 15.2 | 28.7 | 2 | 0.62-0.93 | [@kdrapel](https://github.com/kdrapel) |
| RTX 3090 24GB (turboquant, n-max 6) | 39.8 | 61.5 | 6 | 0.61-0.90 | [@NicholaiVogel](https://github.com/NicholaiVogel) |
| RTX 3090 24GB (b10450) | 41.3 | 63.5 | 2 | 0.69-0.87 | [@hauntedhost](https://github.com/hauntedhost) |
| GMK EVO-X2, Ryzen AI Max+ 395 (64GB unified, ROCm/HIP) | 10.5-11.1 | 21.4-22.2 | 12 | 0.95-1.0 | [@KyaniteLabs](https://github.com/KyaniteLabs/qwen38-27b-strix-halo) |
| 2× RTX 5060 Ti 16GB (PP, default `-sm layer`) | 22.1 | 42.8 | 2 | 0.53-0.94 | [@Jackwwg83](https://github.com/Jackwwg83) |
| 2× RTX 5060 Ti 16GB (TP, `-sm tensor`) | 37.1 | 65.9 | 2 | 0.51-0.88 | [@Jackwwg83](https://github.com/Jackwwg83) |
| RTX 5090 32GB (desktop) | 62.7 | **108.7** | 3 | 0.72 | [@jcr211](https://github.com/jcr211) |
| RTX 5090 32GB (desktop) | 69.3 | 129.1 | 4 | 0.55 | [@paulomcg](https://github.com/paulomcg) |
| RX 7900 GRE 16GB (Vulkan, packed) | 28.7 * | **47.8 avg (36.6–53.8)** | 3 | 0.87–0.96 (avg ~0.93) | [@lsunay](https://x.com/lsunay1) (Hermes agent on PC-12) |
| RTX 5090 32GB (Q4_K_M, 131K) | 76.9 | 155.5 | 2 | 0.50-0.95 | [@anstaendig](https://github.com/anstaendig) |
| RTX 5090 32GB (UD_Q4_K_XL, 262K) | 74.3 | 179.7 | 4 | 0.35–0.91 | [@anstaendig](https://github.com/anstaendig) |
| RTX PRO 6000 Blackwell Max-Q 96GB | 45.7 | 97.1 | 2 | 0.52-0.95 | [@awilliamson](https://github.com/awilliamson) |
| 2× Tesla P40 24GB (tensor split) | 11.7 | 22.6 | 4 | 0.68-0.87 | [@lyesrock](https://github.com/lyesrock) |
| RTX 5090 32GB (UD-Q5_K_XL, 262K) | 69.6 | 147.5 | 6 | 0.69-0.87 | [@lyesrock](https://github.com/lyesrock) |
| 2× RTX 5060 Ti 16GB (TP, Q4_K_M) | 38.3 | 76.0 | 3 | 0.43-0.93 | [@mgoswick](https://github.com/mgoswick) |
| RTX 5090 32GB (UD-Q4_K_XL, 262K, q4_0 KV) | 76.3 | 171.7 | 4 | 0.69-0.87 | [@lyesrock](https://github.com/lyesrock) |
| 2× Tesla P40 24GB (UD-Q4_K_XL, 262K, q4_0 KV) | 13.2 | 23.4 | 4 | 0.82 | [@lyesrock](https://github.com/lyesrock) |
| RX 7900 XTX 24GB (Vulkan/RADV) | 28.8 | 70.7 | 3 | 0.43-0.95 | [@Splizard](https://github.com/Splizard) |
| RTX 3090 24GB (UD-Q2_K_XL, OC) | 52.4 | 85.6 | 2 | 0.76 | [@dcrey7](https://github.com/dcrey7) |
| RTX 3090 24GB (UD-Q4_K_XL, OC) | 43.9 | 79.6 | 2 | 0.78 | [@dcrey7](https://github.com/dcrey7) |

\* A6000 row: unsloth Q8_K_XL, 256K context, q8_0 KV cache — 40.0 GB VRAM baseline, 41.4 GB with spec (rows above: Q4_K_M, 131K, q4_0 KV).
\* RX 7900 XTX row: unsloth Q4_K_M, 131K context, q4_0 KV cache — 18.9 GB VRAM baseline, 19.7 GB with spec.
\* 3×24GB TP row: two RTX 3090 + one 3090 Ti, Unsloth UD-Q6_K_XL, tensor-parallel `--split-mode tensor`, `--parallel 4`, 500K unified KV pool, q8_0 KV, f16 draft KV, mmproj Q8, temp 1.0. VRAM ~16.6 GB/GPU baseline, ~18.7 GB/GPU with spec (tightest card). Method: stock `probe.py`. `--parallel 1` was not required on this host.
\* RTX 4090 row: unsloth UD-Q4_K_XL, 160K context, q4_0 KV cache, llama.cpp b10360, Windows/CUDA, 275W power limit — 20.6 GB VRAM baseline, 21.9 GB with spec.
\* RX 9070 row: two 16GB cards on one 31.84 GiB pool, Vulkan build b10426, unsloth UD-Q4_K_XL, **262K context**, q8_0 KV cache — 28.0 GiB across the pool with spec. Method differs from the rows above and is spelled out under the sweep below.
\* R9700 row: unsloth UD-Q4_K_XL, 262K context, q4_0 KV cache, llama.cpp b10433, Vulkan/RADV — 22.53 GB VRAM baseline, 24.55 GB with n-max 2. Method: unchanged `probe.py` at commit `67c20536`, three runs x three prompts, thinking off.
\* Ryzen AI Max+ 395 row: 64GB unified memory, unsloth UD-Q4_K_XL, 32K context, q8_0 KV cache, llama.cpp b10437, Windows build 26200, Vulkan with AMD driver 32.0.31035.1003. Method: unchanged `probe.py` at commit `67c2053`, three runs x three prompts, thinking off.
\* RTX 4090 Spadav_ row: unsloth Q4_K_M, 200K context, q4_0 KV cache, q8_0 draft KV, mmproj loaded (888MB on GPU) — method: stock probe.py + 4096-token curl (MTP crossover: overhead dominates at ≤400 tokens, +60% at 4096 tokens).
\* Radeon 890M row: Ryzen AI 9 HX 370 (Strix Point, gfx1150), 48 GB UMA carve of 96 GB DDR5, unsloth UD-Q4_K_XL, **131K context**, q4_0 KV cache, llama.cpp Vulkan/RADV, Ubuntu 26.04 — 19.15 GB VRAM baseline, 20.13 GB with n-max 2. Method: unchanged `probe.py`, three runs x three prompts, thinking off. Note this is a *different* APU class from the Ryzen AI Max+ 395 row above: Strix Point 890M is 16 CUs on a 128-bit bus, Strix Halo 8060S is 40 CUs on 256-bit, and this row holds 131K context resident against that row's 32K — both differences push this baseline down.
\* RTX PRO 6000 row: unsloth Q4_K_M, 131K context, q4_0 KV cache, llama.cpp b10335, CUDA — method: stock `probe.py`, thinking off, `--parallel 1` both sides. n-max 4 on this card: 85.7 overall (code 105.7 up, prose 58.8 down, acceptance 0.65-0.77) — overall peaks at n-max 2 here, same code-up/prose-down shape as the A6000 and 3×3090 sweeps. Cross-engine bonus on the same card: vLLM 0.27.1 with unsloth's NVFP4 build and MTP (`--speculative-config '{"method":"mtp","num_speculative_tokens":N}'`) gives 63.3 -> 96.1 at n=2 and 116.3 (+84%) at n=4 — vLLM keeps climbing where llama.cpp has peaked. (An earlier revision carried a streamed-word-drop caveat for vLLM MTP; it was traced to the benchmark harness's own SSE parsing — a grep regex truncating multi-token deltas at escaped quotes — not to vLLM. Retracted with verification at [vllm-project/vllm#52469](https://github.com/vllm-project/vllm/issues/52469), closed; streamed output is byte-identical to non-streamed with MTP on.)
\* RTX 5090 desktop row: unsloth UD-Q4_K_XL, 192K context, q8_0 KV cache, llama-server self-built from the PR #26704 branch (master-equivalent for this path, CUDA arch 120), Windows 11 native — ~26.3 GB VRAM baseline, ~28.2 GB at n-max 4. Method differs from `probe.py`: server `timings.predicted_per_second` over 900-token generations, 2 runs x the same 3 prompts, thinking off, warmup discarded. Acceptance range is per-request (prose low end, Python high end), 0.60 aggregate.
\* RTX 5090 32GB rows: unsloth Q6_K, 128K/256K context, q4_0 KV cache, llama.cpp `62bf73d`, Windows/CUDA. Loaded VRAM was 24,441/27,385 MiB baseline and 25,797/29,381 MiB with spec. MTP gained 110.0% at 128K and 96.3% at 256K.
\* RTX 3090 Ti rows: Q4_K_M, 128K/256K context, q4_0 KV cache, llama.cpp `62bf73d`, Linux/CUDA. Loaded VRAM was 18,842/21,786 MiB baseline and 20,134/23,718 MiB with spec. MTP gained 45.0% at 128K and 49.0% at 256K. The 256K MTP arm had 846 MiB free at load.
\* RTX 5090 32GB and RTX 3090 Ti method: unchanged `probe.py` at commit `b299c0f`, three runs x three prompts, thinking off, warmup discarded. Baseline and MTP used the same model and serving config at each context. Only the MTP arm added `--spec-type draft-mtp --spec-draft-n-max 2`.
\* RTX 5090 desktop UD-Q5_K_XL row: 131K context, q4_0 KV cache, llama.cpp b10448 (`ad1de39e0`), Windows/CUDA 13.3. Loaded VRAM was 22,449 MiB baseline and 24,083 MiB at n-max 4. Method: unchanged `probe.py` at commit `b299c0f`, three runs x three prompts, thinking off, warmup discarded. N-max 2 reached 129.4 tok/s with 80.9% aggregate acceptance; n-max 4 reached 144.2 tok/s with 63.8% aggregate acceptance, a 117.5% gain over baseline. Only the MTP flags changed within each pair.
\* 3×3060 row: unsloth UD-Q4_K_XL, 262K context, q8_0 KV cache, layer split `--split-mode layer --tensor-split 35,37,28` (no NVLink), llama.cpp b10068, `--parallel 1`, `--spec-draft-n-max 8 --spec-draft-p-min 0.85`. Method differs from the rows above: **serve-timing averages from llama-server logs, not `probe.py`** — baseline = weighted mean of 126K decoded tokens with MTP off (17.3 t/s); with flag = weighted mean over 23.4M decoded tokens of live traffic (24.5 t/s, short bursts 28-39 t/s, long-context drops to 14-16 t/s). Acceptance = aggregate 0.87 (176,446/202,620 drafted tokens, per-task median 0.884, range 0.70-1.00) from `draft acceptance` log lines.
\* Radeon 9060 XT 16 row: AtomicChat AD-IQ3_S_IQ3_XXS, 128k context, q4_0 KV cache — method: stock probe.py. Windows, ROCM
\* RTX 3090 b10450 row: unsloth Q4_K_M, 131K context, q8_0 KV cache, q8_0 draft KV, llama.cpp b10450 (master `ece963f41`), CUDA/Linux (CachyOS), froggeric fixed chat template, `--reasoning-format deepseek` — method: unchanged `probe.py`, three runs x three prompts, thinking off. Worth noting: the b10450 *baseline* (41.3) equals the original day-one with-flag number for this card — the young hybrid-attention kernels caught up upstream, and the flag now stacks on top of that (+54%).
\* GMK EVO-X2 row: Ryzen AI Max+ 395 (Strix Halo), 64GB unified memory, Linux, ROCm/HIP llama.cpp (gfx1151, ROCm 7.2.4), unsloth UD-Q4_K_XL, 96K context, f16 KV cache, `--parallel 1`, thinking off — 10.5-11.1 tok/s spec-off baseline; 21.4-22.2 tok/s with `--spec-draft-n-max 12` at 0.95-1.0 acceptance on the bench prompt (novel-traffic acceptance 0.345); stacking `--spec-type draft-mtp,ngram-mod --spec-ngram-mod-n-min 24` takes the streamed count bench to 59.7-64.0 cold and 148-163 warm on back-to-back repeats — the warm figure is an ngram repetition artifact on that prompt, not a general speedup (real novel traffic: prose 11-24, code 30-40 tok/s); production metric is time-per-task, 7.6-14.3 s per correct task across the thermal band. Method: streamed HTTP bench against a live llama-server (not `probe.py`), medians of 3+ runs; one-command reproducer: [bench.sh](https://github.com/KyaniteLabs/qwen38-27b-strix-halo/blob/main/bench.sh), full writeup: [one week with Qwen3.8-27B on Strix Halo](https://kyanitelabs.tech/blog/qwen-27b-strix-halo-one-week-local).
\* 2× RTX 5060 Ti rows: unsloth UD-Q4_K_XL (sha256 `bee238bb…1372`), 131K context, q4_0 KV cache, llama.cpp built from source at commit `ece963f4` with `-DCMAKE_CUDA_ARCHITECTURES=120`, CUDA 13.0 / driver 580.173.02, PCIe 3.0 x8, `PHB` topology (no P2P). Method: unchanged `probe.py`, three runs x three prompts, thinking off. VRAM: PP 21.7 GB baseline / 23.0 GB with spec; TP 20.9 GB / 22.1 GB. The 16.68 GiB of weights do not fit one 16GB card, so two cards is the floor on this box — the split-mode choice is not optional here, which is what makes the two rows worth reading side by side. Both rows verified at `n_ctx_slot = 131072`. Details under the sweep below.
\* RTX 5090 desktop row: unsloth Qwen3.8 Dynamic NVFP4 (FP8-as-Q8 unified-mtp), 163,840 context, q8_0 KV cache, llama.cpp b10430, Windows/CUDA, driver 610.88 — first NVFP4 quant in the table. Full n-max sweep: 2 → 98.4, **3 → 108.7 (+73%)**, 4 + p-min 0.60 → 103.9, 8 → 92.6 — deep-draft optimum consistent with the A6000 48GB pattern; n-max 8 confirmed worst spec setting. Method: unchanged `probe.py`, three runs x three prompts, thinking at template default (xhigh). Acceptance 0.721 aggregate (1845/2558 from server logs).
\* RTX 5090 32GB row: unsloth UD-Q4_K_XL, **192K context**, q8_0 KV cache, mmproj loaded (vision, `--image-min-tokens 1024`), llama-swap `unified-cuda-2026-08-14`, Linux/CUDA — 26.5 GB VRAM baseline, 28.5 GB with spec. **Ungated** — see the p-min A/B below. Both arms at `--parallel 1` so only the spec flags differ. Method: stock `probe.py`, medians of three runs x three prompts, thinking off. n-max sweep at p-min 0.60 below.
\* RTX 3090 turboquant row: unsloth Q4_K_M, 131K context, q4_0 KV cache, custom turboquant llama.cpp at commit `95b18c0`, NVIDIA driver 610.43.03, `--parallel 1`, all layers on GPU, thinking off. Method: unchanged `probe.py`, medians of three runs x three prompts, same setup both arms. MTP at n-max 6 with p-min 0.75.
\* RX 7900 GRE row: the only packed-16GB row in the table (96% VRAM with MTP; 86% spec-off). spec-off baseline back-filled 2026-08-16 on the same card (MTP removed as the only variable, live agent traffic to ~45K) — 28.5–28.8 tok/s at 35–45K context. custom AtomicChat IQ3_XXS quant (not unsloth Q4_K_M), 90K context, turbo3/turboquant KV cache, llama.cpp 1655 (2168b0cd8) in a custom llama-cpp-turboquant Docker image, Debian 13 trixie, kernel 6.12.101, Vulkan/AMD Navi 31, --parallel 1, --reasoning-budget 512, flash-attn on. Both arms live Hermes agent traffic (not probe.py), context bands differ by ~1.5K tokens. Full study under the section below.
\* RTX 3090 turbo3 section: same host (RTX 3090 24GB, Debian 12, Ryzen 7 9700X, CPB disabled), unsloth Q4_K_M, **both arms MTP-on** (`--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75 --spec-default`, `-np 1`) — this row compares two KV cache configurations, not spec-off vs spec-on. Baseline arm: q4_0 KV, 185K context, llama27b-mtp:cuda image (mtp-clean fork), live Hermes agent traffic (14 completed turns, 100–2700 output tokens, avg 37.2 tok/s, acceptance 0.57). Flag arm: **turbo3 KV, 200K (204800) context** (262K exceeded 24 GB at model load), llama-cpp-turboquant:cuda-latest (v10465, fca3093c9), clean probe 3×400 tokens, medians of 3, avg 54.2 tok/s (48.06–58.34), acceptance 0.64–0.84. VRAM 23.0/23.3 GB of 24.6 GB (~94–95%), 52–63°C. Confound disclosed: the two arms run different llama.cpp builds (mtp-clean vs turboquant v10465), so part of the delta is the newer build, not only the KV cache. Method differs from the rows above and is spelled out below.
\* RTX 5090 32GB (Q4_K_M, 131K): Unsloth Q4_K_M, 131K context, q4_0 KV cache - 19.54GB baseline, 20.38GB with spec.
\* RTX 5090 32GB (UD_Q4_K_XL, 262K): Unsloth Q4_K_XL, 262K context, q4_0 KV cache - 23.27GB baseline, 24.75GB with spec.
\* RTX PRO 6000 Blackwell Max-Q row: lmstudio-community/Qwen3.8-27B-GGUF Q8_0, 131K context, q4_0 K/V cache, llama.cpp 0.1.0-dev build 10454 (`4df29be4f`), built with GNU 16.1.1 for Linux x86_64 / CUDA 13.3.1-1, RTX PRO 6000 Blackwell Max-Q Workstation Edition 96GB at 300W, `--parallel 1`. VRAM: 32.072 GiB baseline / 33.540 GiB with spec (+1468 MiB). Method: unchanged `probe.py`, three runs x three prompts, thinking off; qwen38-mtp (28527fc55) commit used for `probe.py`.
\* 2× Tesla P40 row: unsloth UD-Q5_K_XL, 131K context, q4_0 KV cache, llama.cpp b10453 (`3cb7ffb1a`), Ubuntu 25.04 / CUDA 12.9, driver 580.178.04, `--tensor-split 1,1` with `numactl --interleave=all`, P2P disabled (cards on separate NUMA nodes, SYS path), `--parallel 1`, thinking off. ~11 GB/GPU baseline, ~12 GB/GPU with spec. Method: `probe.py` unchanged, three runs x three prompts, both arms otherwise identical. MTP arm: `--spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.75`; prefill 354 → 295 tok/s (20K tokens). Full n-max sweep and a multi-model companion sweep below.
\* RTX 5090 32GB UD-Q5_K_XL 262K row: unsloth UD-Q5_K_XL, 262K context, q4_0 KV cache, llama.cpp b10453 (`3cb7ffb1a`), Linux/CUDA, driver 610.57.04. Loaded VRAM ~32.1 GB. Method: unchanged `probe.py`, three runs x three prompts, thinking off, warmup discarded. Both arms at `--parallel 1`. Baseline 69.6 tok/s; with `--spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-p-min 0.75` 147.5 tok/s (+112%). Acceptance 0.69-0.87 (avg 0.78). N-max 4 ungated reached 155.8 tok/s (0.31-0.87 acceptance) — ungated is faster but the gated config is the safer daily bet.
\* 2× RTX 5060 Ti Q4_K_M row: unsloth Q4_K_M, 131K context, q4_0 KV cache, `--split-mode tensor`, llama.cpp b10450 (`ece963f`), Windows 11 / CUDA 13.2 native `sm_120a`, driver 596.36, `--parallel 1`. VRAM per GPU: 9,808 MiB baseline / 10,684 MiB at n-max 3. Method: unchanged `probe.py` at commit `b299c0f`, three runs x three prompts, thinking off, warmup discarded. Only the MTP arm added `--spec-type draft-mtp --spec-draft-n-max 3`.
\* RTX 5090 32GB UD-Q4_K_XL 262K q4_0 KV row: unsloth UD-Q4_K_XL, 262K context, q4_0 KV cache (both K and V), llama.cpp b10453 (`3cb7ffb1a`), Linux/CUDA, driver 610.57.04. Loaded VRAM ~25.8 GB. Method: unchanged `probe.py`, three runs x three prompts, thinking off, warmup discarded. Both arms at `--parallel 1`. Baseline 76.3 tok/s; with `--spec-type draft-mtp --spec-draft-n-max 4` (ungated, no p-min) 171.7 tok/s (+125%). Acceptance 0.69-0.87 (avg 0.69). N-max 6 gated (p-min 0.75) reached 154.5 tok/s (0.77-0.87 acceptance) — ungated is faster, gated is the safer daily bet.
\* 2× Tesla P40 UD-Q4_K_XL 262K q4_0 KV row: unsloth UD-Q4_K_XL, 262K context, q4_0 KV cache (both K and V), llama.cpp b10453 (`3cb7ffb1a`), Linux/CUDA, `--tensor-split 1,1` (no NVLink), `--parallel 1`. Method: unchanged `probe.py`, three runs x three prompts, thinking off, warmup discarded. Baseline 13.2 tok/s; with `--spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.75` 23.4 tok/s (+77%). Acceptance 0.82 avg. N-max 4 ungated reached 23.6 tok/s (0.68 acceptance) — only +0.8% faster (noise) with much lower acceptance; gated is the clear winner on bandwidth-starved Pascal. N-max 6 gated: 22.5 tok/s (0.80 acceptance) — n-max 4 is the sweet spot. KV f16 was within 2% of q4_0 on all arms (13.4 / 23.6 / 22.4 / 22.6) and also fits 262K ctx on 2×24GB with this quant — q4_0 KV buys headroom, not speed; f16 is the default choice when VRAM allows.
\* RX 7900 XTX Vulkan row: unsloth Q4_K_M, 131K context, q4_0 KV cache, llama.cpp master `4695f00` (2026-08-17), Vulkan backend (RADV, Mesa 26.1.5) on Void Linux musl, kernel 7.0.14, ReBAR enabled, Ryzen 9 5900XT host — no ROCm installed; this is mainline Mesa only. 0.6 GB VRAM before serve, 18.6 GB serving baseline, 19.9 GB with spec. Method: unchanged `probe.py` at `dc18736`, three runs x three prompts, thinking off, both arms `--parallel 1`. Row is the n-max 3 arm (aggregate acceptance 0.72); n-max 2 on the same config measured 57.9 at 0.80 aggregate (1652/2055), the range's low end is the prose prompt in both arms. Full sweep and gate A/B in the section below. Relative to the earlier XTX row: this baseline reads lower (28.8 vs 30.7) while the flag arms read far higher — different backend and a 2026-08-17 build (see rule 6); the deltas are the durable part.
\* RTX 3090 UD-Q2_K_XL row: unsloth UD-Q2_K_XL (10.7 GB), 32K context, f16 KV cache, llama.cpp master source snapshot of 2026-08-14 (built from archive, so no commit hash; it contains the `ssm_scan` state-rollback that makes deep MTP drafts work on the DeltaNet layers), CUDA 13 self-built, CachyOS/Linux, driver 610.43.03. VRAM 13.9 GB baseline, 14.7 GB with spec. **The card is overclocked**: memory +1200, core +180, power limit 430 W against a 390 W default. Method: unmodified `probe.py` at `a4c3028da1`, three full passes per arm, the table shows the median of the three pass medians; passes were 52.5/52.4/52.3 and 85.6/85.1/87.2. Both arms `--parallel 1`, thinking off.
\* RTX 3090 UD-Q4_K_XL row: same machine, same build, same method, same overclock; unsloth UD-Q4_K_XL (17.9 GB), 32K context, f16 KV. VRAM 20.5 GB baseline, 21.4 GB with spec. Passes were 43.9/43.9/43.9 and 79.8/79.6/78.2. This row is the one to compare against the other 3090 rows, since those use ~17 GB Q4 files; the UD-Q2_K_XL row above is faster mainly because it reads 38% fewer weight bytes per token, not because of the overclock. See the section below.

### A6000 48GB: n-max sweep

Same A6000, same config as the row above, `--spec-draft-n-max` swept 2-6. Overall and per-prompt probe medians (tok/s), draft acceptance from the server log:

| n-max | Overall | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Acceptance |
|---|---|---|---|---|---|
| 2 | 52.5 | 57.0 | 43.1 | 52.5 | 0.54-0.98 |
| 3 | 60.7 | 67.6 | 44.1 | 60.7 | 0.42-0.91 |
| **4** | **64.1** | 77.1 | 41.6 | 64.1 | 0.32-0.93 |
| 5 | 62.8 | 80.5 | 40.8 | 62.8 | 0.29-0.84 |
| 6 | 58.6 | 84.3 | 37.4 | 58.6 | 0.23-0.84 |

The overall peak is n-max 4 here, not 2 — the card has enough headroom to absorb the cost of deeper verification before the acceptance decay eats the win. Same shape as the 5090 sweep: the code prompts keep rising all the way up (84.3 at n-max 6), the prose prompt falls from the start (43.1 -> 37.4), and acceptance decays monotonically. Daily mixed use: 4, pure code sessions: 5-6, prose-heavy: 2.

### 3× RTX 3090 / 3090 Ti 24GB (TP): n-max sweep

Same 3×24GB host, same config as the row above, `--spec-draft-n-max` swept 2-6. Overall and per-prompt probe medians (tok/s), draft acceptance from the server log:

| n-max | Overall | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Acceptance |
|---|---|---|---|---|---|
| 2 | 81.1 | 96.2 | 68.8 | 81.1 | 0.52-0.96 |
| **3** | **95.6** | 108.1 | 71.1 | 95.6 | 0.38-0.92 |
| 4 | 94.9 | 114.7 | 67.2 | 94.9 | 0.37-0.92 |
| 5 | 93.3 | 115.9 | 55.7 | 93.3 | 0.29-0.84 |
| 6 | 88.7 | 112.9 | 54.1 | 88.7 | 0.25-0.80 |

Overall peaks at n-max 3. Code keeps climbing through n-max 5 (115.9); prose falls after n-max 3 (71.1 → 54.1). Same shape as the 5090 and A6000 sweeps. Daily mixed use: 2–3; pure code: 4–5; prose-heavy: 2. A separate greedy code-completion hash gate (not `probe.py`) matched n-max 2 and diverged at n-max 1/3/4; chaining `ngram-mod` made n-max 2 unstable on that gate, so this host still ships n-max 2 without ngram.

### 2x RX 9070 16GB: n-max sweep, and the second knob nobody is turning

Same pair of cards, same config as the RX 9070 row above, at the full 262K window. Decode medians (tok/s) at three prompt depths, draft acceptance from the server log:

| config | ~0 ctx | 16K | 65K | Acceptance |
|---|---|---|---|---|
| spec off | 22.1 | 21.0 | — | — |
| n-max 2, no gating | 41.6 | 37.2 | 28.0 | 0.73 |
| **n-max 4, `--spec-draft-p-min 0.60`** | **42.9** | **38.5** | 27.4 | 0.86 |
| n-max 4, `--spec-draft-p-min 0.75` | 38.3 | 36.6 | **28.9** | 0.91 |
| n-max 4, `--spec-draft-p-min 0.80` | 40.7 | 35.5 | — | 0.90 |
| n-max 8, any p-min | 30.0-36.7 | 31.8-33.9 | — | 0.74-0.84 |

**`--spec-draft-n-max` alone is only half the tuning.** `--spec-draft-p-min` is a confidence gate: the head stops drafting once its own probability falls below the threshold, so a deeper n-max costs nothing on the rounds where the draft was going to be rejected anyway. That is what makes n-max 4 usable here — ungated n-max 4 was worse than n-max 2, and gated at 0.60 it drafts ~3.3 tokens per round at 0.86 acceptance instead of exactly 2.0 at 0.73. On a copy-heavy prompt (rename-and-echo over a 4K-token file) the gated arm is **+21%** over the deployed n-max 2, well above the +3% it shows on mixed work. n-max 8 loses at every gate value.

Two things this rig says that the NVIDIA rows do not:

- **The delta is much larger and the ceiling is the same.** +88% at n-max 2, +94% at the tuned setting, against +33-39% on the 24GB NVIDIA cards — because the *baseline* is unusually bad here, not because the flag is better. Vulkan on RDNA4 at batch-1 decodes across two cards serially, so 22.1 tok/s unassisted; with the head engaged it lands at 41.6-42.9, the same absolute band as everyone else. If your card is bandwidth-poor at batch 1, this flag is worth more to you than to a 3090 owner.
- **N=1 crowned the wrong arm.** A single-run screen put p-min 0.75 on top; medians of 3 flipped the winner to 0.60. The noise band here is around +/-2 tok/s, which is wide enough to invert a ranking. Do reps before you deploy a config.

Method note, since it is not the repo's: numbers come from a local greedy streaming harness (max 1200 tokens) rather than `probe.py`, all in one session against one server instance with only the spec flags varying. The n-max 2 and p-min 0.60 arms are medians of 3; the spec-off, p-min 0.75/0.80 and n-max 8 arms are single screening runs. Treat the sweep as a shape, and the two headline arms as measurements.

### AMD Radeon AI PRO R9700 32GB: n-max 2 versus 4

Same R9700, same unsloth UD-Q4_K_XL model and serving config as the row above, at 262K context with q4_0 KV. Upstream `probe.py` unchanged: medians of three runs per prompt, thinking off. Draft acceptance is from the server log.

| n-max | Overall | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Acceptance |
|---|---|---|---|---|---|
| 2 | 43.3 | 45.3 | **36.3** | 43.3 | 0.60-0.94 |
| **4** | **47.4** | **75.5** | 33.3 | **47.4** | 0.28-0.90 |

N-max 4 lifts the overall median another 9.5% over n-max 2, driven by a 66.7% jump on the Python prompt. Prose falls 8.3% as aggregate acceptance drops from 82.3% to 60.5%; prose-only acceptance is 29.4% at n-max 4. The speed shape supports n-max 4 as a code-specialized arm, while n-max 2 remains the mixed-workload default. Loaded VRAM: 22.53 GB spec-off, 24.55 GB at n-max 2, and 24.87 GB at n-max 4.

### Ryzen AI Max+ 395 / Radeon 8060S: n-max 2 versus 4

Same 64GB Strix Halo system, same unsloth UD-Q4_K_XL model and serving config as the row above: Windows Vulkan, 32K context, q8_0 KV, Flash Attention, batch 2048/512, and `--parallel 1`. Upstream `probe.py` was unchanged; all values are medians of three runs per prompt with thinking off. Acceptance is from the server log, excluding warmup.

| n-max | Overall | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Acceptance |
|---|---|---|---|---|---|
| **2** | **23.7** | 25.2 | **19.1** | **23.7** | 0.52-0.94 (78.2% aggregate) |
| 4 | 23.5 | **29.9** | 16.5 | 23.5 | 0.35-0.91 (65.8% aggregate) |

MTP n-max 2 doubles the overall median versus the 11.5 tok/s spec-off control (+106%). N-max 4 is effectively flat overall (-0.8% versus n-max 2), but the workload split is sharp: Python rises 18.7% while prose falls 13.6%. This system keeps n-max 2 for mixed use and treats n-max 4 as a code-specialized option.

### Radeon 890M iGPU (Strix Point): n-max sweep — biggest gain here, and an unstable peak

Same 890M, same config as the row above, `--spec-draft-n-max` swept with a no-flag baseline.
**Two independent runs, both with unchanged `probe.py`, identical flags, nothing else on the GPU.**
Overall probe medians (tok/s):

| n-max | Run A (11:10) | Run B (14:04/15:14) | Acceptance (Run B) |
|---|---|---|---|
| baseline (no flag) | 2.6 | 2.7 | - |
| 2 | 5.9 | 5.7 | 0.59-0.91 |
| 3 | 7.5 | 7.0 | 0.47-0.73 |
| 4 | 7.9 | **7.4** | 0.35-0.69 |
| 5 | **8.5** | 6.7 | 0.27-0.61 |
| 6 | 7.2 | not run | - |

🛑 **The two runs disagree on where the peak is — Run A says n-max 5, Run B says n-max 4 — so I am
not claiming a peak for this card.** Baseline and n-max 2 reproduce to within 3% (2.6/2.7 and
5.9/5.7), and the divergence widens with n, reaching **27% at n-max 5**. That is worth knowing on
its own: on a bandwidth-shared iGPU whose clocks and memory contention move between runs, three
runs x three prompts is not enough to resolve a peak, even though it is plenty for the n-max 2
headline. Anyone sweeping on an APU should repeat the sweep before trusting an optimum.

**The table row above uses n-max 2, where both runs agree.** At that setting this is **+111%**;
at whichever of 4/5 is the true optimum it is **+174% to +227%** — the largest gain posted here
either way, on the slowest hardware in the table.

Why an iGPU tops the gain column while sitting last on absolute throughput:
**MTP pays in proportion to how bandwidth-starved decode already is.** A dense 27B on a 128-bit
LPDDR5 bus is the most starved configuration in this table, so amortising the weight read across
accepted drafts buys the most. A large multiple of a small number is still a small number - this
does not make the 890M a good card for a dense 27B, it makes MTP the difference between unusable
and marginal on one.

Daily mixed use: 4 is the safe pick (top-2 in both runs). n-max 6 fell away in the run that tested it.

### RTX 5090 32GB desktop: n-max sweep, and where the p-min rule inverts

Same config as the row above, `--spec-draft-n-max` swept off/1-6. Per-prompt means of 2 runs (tok/s), acceptance per-request from server timings:

| n-max | Overall | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Acceptance |
|---|---|---|---|---|---|
| off | 61.4 | 61.6 | 61.3 | 61.5 | — |
| 1 | 94.2 | 98.1 | 86.2 | 98.3 | 0.68-0.94 |
| 2 | 118.9 | 130.1 | 99.0 | 127.7 | 0.56-0.92 |
| 3 | 127.9 | 150.8 | 97.3 | 135.6 | 0.45-0.90 |
| **4** | **135.0** | 162.6 | 95.3 | **147.0** | 0.38-0.87 |
| 5 | 129.0 | 158.0 | 86.3 | 142.6 | 0.30-0.78 |
| 6 | 127.5 | 160.1 | 78.5 | 143.9 | 0.27-0.76 |

Same shape as every sweep above — code keeps climbing, prose peaks early, acceptance decays monotonically — but the desktop 5090 stretches rule 1 to its endpoint: the overall optimum is 4, the net gain is **+120%** over spec-off (the largest single-GPU delta in the table), and even n-max 6 still beats n-max 2. Verification on this card is cheap enough that deep drafts keep paying long after acceptance has collapsed.

The same headroom **inverts rule 2**. `--spec-draft-p-min` swept at n-max 4, same method:

| p-min | Overall | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Aggregate acceptance |
|---|---|---|---|---|---|
| **0.0** | **129.8** | 157.6 | **92.9** | 138.8 | 0.60 |
| 0.3 | 131.3 | 164.0 | 85.5 | 144.5 | 0.62 |
| 0.5 | 111.8 | 140.1 | 69.4 | 125.9 | 0.69 |
| 0.7 | 95.9 | 118.9 | 58.9 | 109.9 | 0.80 |
| 0.85 | 96.1 | 125.1 | 59.8 | 103.4 | 0.89 |

Gating raises acceptance exactly as rule 2 says (0.60 → 0.89) and costs throughput the whole way past 0.3 — the 0.60-0.70 band that rescues n-max 4 on the RX 9070 pair loses 26% here, and prose loses the most. The two rules are one rule: p-min converts skipped drafts into plain decode rounds, which is a win only when your plain decode is bandwidth-starved and your verify batches hurt. On a card where verification is nearly free, every draft is worth attempting and acceptance is a vanity metric. Bandwidth-poor rigs: gate. Desktop Blackwell: leave p-min at 0 and draft deep.

One datapoint for the caveats section's "the floor should rise with upstream work": before settling on the build above, the same card ran the same configs on a launch-week binary (b10435-era). Moving to current master was worth **+10-15% decode across every quant tested**, MTP on or off, before touching any flag — the qwen3_5 hybrid-attention path is still being optimized upstream. If your row was measured on a day-one build, rebuilding may be the cheapest speedup in this README.

### AMD Radeon 9060 XT 16 GB
Single card. LLama compiled with ROCM 7.1 support (Vulkan slower with dense models). Model is AtomicChat IQ3_XXS (https://huggingface.co/AtomicChat/Qwen3.8-27B-GGUF). Upstream `probe.py` unchanged.

| n-max | Overall | Acceptance |
|---|---|---|
| 2 | 28.7 | 0.62-0.93 |
| 3 | 30.7 |  0.41-0.91 |
| 4 | 30.9 |  0.29-0.93 |

nb. made another sweeping test on MTP n-max with another benchmark and more complex task and n-max = 2 was the clear winner, performance was impacted with larger n-max, 2 is the sweet general spot

### 2× RTX 5060 Ti 16GB: on a multi-GPU box the split mode is worth more than the flag

Same box, same GGUF, same serve config throughout — only `--split-mode` and the spec flags change. Stock `probe.py`, medians of three runs x three prompts, thinking off. Acceptance from the server log.

| split mode | spec | Overall | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Acceptance |
|---|---|---|---|---|---|---|
| `layer` (default) | off | 22.1 | 22.0 | 22.1 | 22.0 | — |
| `layer` | n-max 2 | 42.8 | 47.3 | 34.4 | 42.8 | 0.53-0.94 |
| `layer` | n-max 3 | **47.5** | 54.0 | 35.4 | 47.5 | 0.48-0.75 |
| `layer` | n-max 4 | 45.2 | 60.1 | 32.1 | 45.2 | 0.39-0.72 |
| `tensor` | off | **37.1** | 37.1 | 37.3 | 37.0 | — |
| `tensor` | n-max 2 | 65.9 | 70.6 | 51.4 | 65.9 | 0.51-0.88 |
| `tensor` | n-max 3 | 69.3 | 78.7 | 53.6 | 69.3 | 0.38-0.74 |
| `tensor` | n-max 4 | **71.3** | 83.4 | 47.6 | 71.3 | 0.35-0.70 |
| `tensor` | n-max 5 | 67.1 | | | | |
| `tensor` | n-max 6 | 56.2 | | | | |
| `tensor` | n-max 8 | 59.7 | | | | |
| `tensor` | n-max 4, `p-min 0.60` | 46.2 | 71.6 | 23.3 | 46.2 | 0.41-0.82 |
| `tensor` | n-max 6, `p-min 0.60` | 45.4 | 65.8 | 24.3 | 45.4 | 0.56-0.85 |
| `row` | off | fails to load | | | | |

**The default split mode costs more than the flag gains.** `--split-mode layer` splits layers across cards, so at batch 1 there is nothing to pipeline: GPU0 computes while GPU1 idles and the pair decodes at single-card bandwidth. `--split-mode tensor` splits each matmul instead, both cards read weights at once, and the baseline goes 22.1 -> 37.1 (**+68%**) before any spec flag is involved. That is a larger free win than MTP gives on a 24GB single card.

The two levers stack cleanly: **22.1 -> 69.3, a 3.14x end-to-end gain from two flags, neither of which costs anything.** Worth spelling out because `layer` is the default, so a two-card owner who follows the launch command verbatim measures 42.8 and stops there.

The all-reduce is not free but it is cheaper than expected here: these are GeForce cards with P2P disabled, on PCIe 3.0 x8 with a `PHB` topology, so every reduction crosses host memory. Ideal scaling would put TP at 44.2 (2x the 22.1 single-card-equivalent); the measured 37.1 says the reduction costs about 16%. A host with working P2P should keep more of it.

Two things worth knowing before you try this:

- `--split-mode row` refuses to load — `device CUDA0 does not support split buffers`. `tensor` is the mode that works.
- `--split-mode tensor` skips the auto-fit pass. `llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort` is a warning, not an error, and the server starts anyway with whatever `-c` you passed. Check `n_ctx_slot` in the log before comparing against a `layer` run, or you may be comparing two different context sizes. Both arms above were verified at `n_ctx_slot = 131072`.

**The n-max sweet spot moves with the split mode, not just the VRAM pool.** Same box, same 32GB, same everything else: `layer` peaks at n-max 3 (47.5, falling to 45.2 at 4) while `tensor` peaks at n-max 4 (71.3), holds 67.1 at 5 and only collapses at 6. Rule 1 says the sweet spot is card-dependent — this says it is also *topology*-dependent, because tensor-split makes each verification step cheaper, so the pool can absorb a deeper draft before acceptance decay eats the win. If you switch split mode, re-sweep n-max; the old optimum does not carry over.

Everything else matches the other sweeps: code prompts keep climbing (83.4 at n-max 4 under `tensor`), prose falls from the start (53.6 at n-max 3 down to 47.6 at 4), acceptance decays monotonically.

**`--spec-draft-p-min` is a large net loss here**, which is the opposite of what it does on the RX 9070 pair. Gating n-max 4 at 0.60 drops the overall median from 71.3 to 46.2 — and it does so *while raising* acceptance from 0.35-0.70 to 0.41-0.82. The gate is refusing drafts that would have been accepted; the prose prompt takes the worst of it (53.6 → 23.3). Both rigs are 2x16GB, so the knob clearly does not generalize across backends — worth sweeping rather than adopting.

**The gain does not scale with output length on this box.** Running the same two arms at 4096 tokens instead of 400 (probe.py with only `MAX_TOKENS` changed): baseline 37.1, MTP n-max 3 68.7 — a 1.85x ratio against 1.87x at 400 tokens. Rule 3 holds where spec overhead dominates short generations; here the tensor-split baseline is bandwidth-bound and stable, so the full gain is already present at 400 tokens and there is nothing left to recover at 4096.

`llama-bench` on the same box as a cross-check, `-fa 1 -ctk q4_0 -ctv q4_0`:

| split mode | pp512 | tg128 | pp4096+tg128 |
|---|---|---|---|
| `layer` | 926.96 ± 18.81 | 22.58 ± 0.02 | 482.17 ± 0.22 |
| `tensor` | 1028.04 ± 2.69 | **38.37 ± 0.26** | 550.86 ± 0.77 |

`llama-bench` puts the decode gain at +70%, `probe.py` at +68% — two different tools, two different prompt sets, same answer. Prefill moves only +11%, which is the expected shape: prompt processing is compute-bound and already batched, so layer-split leaves both cards reasonably busy; decode at batch 1 is where the serialization actually bites.

#### Decode against prompt depth

`probe.py` runs at near-zero context, which is the easiest case for any spec-decode setup. Both arms re-measured under `-sm tensor` with a filler prompt sized against the server's own `/tokenize` endpoint, so the depths are token-exact rather than a chars-per-token guess. TTFT excluded from the decode rate, as in `probe.py`.

| prompt tokens | spec off: TTFT | prefill t/s | decode t/s | MTP n-3: TTFT | prefill t/s | decode t/s | ratio |
|---|---|---|---|---|---|---|---|
| 4,075 | 2.59 s | 1,572 | 36.24 | 2.80 s | 1,454 | 56.05 | 1.55x |
| 16,255 | 7.57 s | 2,148 | 34.53 | 8.25 s | 1,970 | 53.27 | 1.54x |
| 32,467 | 10.51 s | **3,089** | 32.38 | 11.53 s | 2,817 | 50.79 | 1.57x |
| 64,891 | 22.35 s | 2,904 | 28.73 | 24.50 s | 2,649 | 56.43 | 1.96x |
| 129,781 | 52.33 s | 2,480 | 23.70 | 58.11 s | 2,233 | 46.34 | 1.96x |

**The flag matters more the deeper you go, not less.** The unassisted arm loses 35% of its decode rate between 4K and 128K (36.24 → 23.70); the MTP arm loses 17% (56.05 → 46.34). At 128K of resident context this pair still generates at 46 tok/s.

Two caveats on this table. The 64K and 128K rows generated only ~42 tokens before the model stopped, so those decode figures rest on a much smaller sample than the shallower rows — treat the 1.96x as indicative, not settled. And prefill peaking mid-sweep (3,089 t/s at 32K, lower at both 4K and 128K) is the usual batching-vs-attention-cost curve, not a measurement error.

#### Concurrency, and whether MTP survives `--parallel 4`

Upstream says `--parallel 1` is required for spec-decode; the 3x3090 row reports it was not required on that host. On this one **the server starts and serves correctly with `--parallel 4` and MTP together**. Whether it *helps* is a different question. 16K prompts, `-sm tensor`, per-stream is the median stream, aggregate is total generated tokens over wall clock including prefill:

| N | spec off: per-stream | aggregate | MTP n-3: per-stream | aggregate |
|---|---|---|---|---|
| 1 | 34.19 | 6.91 | **57.99** | 10.72 |
| 2 | 19.32 | 18.53 | 20.94 | 16.34 |
| 4 | 19.13 | **29.48** | 19.21 | **29.24** |

**Speculative decoding is a single-stream optimisation.** At N=1 it is worth 1.70x. By N=2 the advantage is inside the noise, and at N=4 the two arms are indistinguishable (29.48 vs 29.24) — once the batch is full the GPU has real work queued and there is no idle verification capacity for the draft head to exploit. If you serve concurrent users, tune `--parallel` and skip the spec flags; if you are one person at a terminal, the flags are most of your speed.

One side effect worth a line: across every arm the MTP runs drew *less* power than their spec-off controls while producing more tokens — combined peak 268 W vs 293 W on the depth sweep, 267 W vs 305 W under concurrency. The draft head is not buying speed with watts.

### RTX 5090 32GB (desktop): the p-min gate is inverted here too

Gate A/B at n-max 4. All three arms at 192K, q8_0 KV, `--parallel 1` — only the
spec flags vary, so these are directly comparable to the row's baseline.
`probe.py` medians of three runs x three prompts, thinking off:

| config | Overall median | Overall mean | P1 code (py) | P2 prose (mmap) | Acceptance |
|---|---|---|---|---|---|
| spec off | 69.3 | 69.3 | 69.3 | 69.2 | — |
| n-max 4, `--spec-draft-p-min 0.60` | 123.9 | 125.4 | 150.7 | 98.0 | 0.73 |
| **n-max 4, ungated** | **129.1** | 131.6 | 157.4 | 109.1 | 0.55 |

Ungated wins on every prompt: +4.2% overall median, +4.4% on code, +11.3% on
prose. Smaller than the 26% loss @taco-devs measured in the 0.60-0.70 band, but
the same direction — and @jcr211's NVFP4 rig lands the same way, with gated
n-max 4 (103.9) below ungated n-max 3 (108.7).

That is three independent RTX 5090s, on three different quants, where **rule 2
inverts**. The gate helps the bandwidth-starved cards it was found on; where
verification is close to free it converts skipped drafts into plain decode
rounds. Worth treating as card-class-dependent rather than a default.

**Acceptance is a vanity metric on this class of card.** Gating moved it from
0.55 to 0.73 while throughput went *down*. Tuning for acceptance picks the
slower configuration.

Depth optimum here is 4. A separate ungated sweep at 224K with two slots put
n-max 6 at 113.0 median against 126.7 for n-max 4, with prose collapsing from
100.3 to 79.3.

**A `--parallel` warning worth its own line.** The README says `--parallel 1` is
required, and it is — but the cost of getting it wrong is not just an
unsupported config, it is a corrupted baseline. This box measured 55.4 tok/s
unassisted at `--parallel 2`; at `--parallel 1`, same everything else, it is
69.3. **`--parallel 2` alone costs ~20% of single-stream decode.** Pairing an
MTP arm against a `--parallel 2` baseline reads as +133% when the honest figure
is +86%. Measure both arms at `--parallel 1`.

### RX 7900 GRE 16GB: packed-16GB live-traffic study, spec-off baseline added 2026-08-16

Data: [@lsunay](https://x.com/lsunay1), Hermes agent on PC-12.

Same host, same model, same serve, only the spec flags varying. The card runs the custom AtomicChat IQ3_XXS quant at 90K context with turbo3/turboquant KV — VRAM sits at 15.35 GB of 16 GB (~96%), so this is the tightest 16 GB rig in the table, and the config was tuned for a live Hermes agent session, not a clean probe run.

Live serve, one slot (`-np 1`), `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75`. Decode speeds and draft acceptance from the server log over eight consecutive real agent turns (context growing 29.2K → 37.0K tokens between turns):

| Task | Output | Decode | Acceptance | Mean draft len |
|---|---|---|---|---|
| 1436 | 700 tok | 51.8 tok/s | 0.930 | 3.46 |
| 1671 | 310 tok | 47.7 tok/s | 0.932 | 3.13 |
| 1795 | 109 tok | 44.8 tok/s | 0.903 | 3.32 |
| 1844 | 778 tok | 53.8 tok/s | 0.964 | 3.71 |
| 2090 | 482 tok | 50.2 tok/s | 0.940 | 3.46 |
| 2260 | 150 tok | 36.6 tok/s | 0.872 | 2.55 |
| 2347 | 751 tok | 52.6 tok/s | 0.962 | 3.70 |
| 2577 | 472 tok | 44.5 tok/s | 0.905 | 3.25 |

Average decode **47.8 tok/s** (best 53.8, worst 36.6 — the 150-token turn, where the spec overhead dominates exactly as the length rule predicts). Acceptance band **0.87–0.96, average ~0.93**; mean draft length 2.55–3.71, i.e. n-max 3 was almost fully consumed. Prompt processing held 239–314 tok/s (3.3–4.2 ms/token) across the 29K–37K context growth. Vulkan compute-graph reuse climbed 757 → 1420 over the session.

**Spec-off baseline, measured 2026-08-16** (the card was deployed spec-on from the start, so this was back-filled): same host, same model, same turbo3 KV, same 90K context — the only variable removed is MTP (no `--spec-type/--spec-draft-*` flags), run in a fresh `--rm` container on port 8095 under the same live Hermes agent session (context grew to ~45K). Server-log `print_timing` over the final turns:

| Task | Output | Decode | Prompt eval |
|---|---|---|---|
| 8 | 251 tok | 34.08 tok/s | 300 tok @ 246.8 tok/s |
| 10 | 150 tok | 30.94 tok/s | 20,265 tok @ 460.7 tok/s (cold) |
| 425 | 455 tok | 30.58 tok/s | 3,753 tok @ 370.8 tok/s |
| 885 | 84 tok | 30.94 tok/s | 356 tok @ 261.7 tok/s |
| 2059 | 116 tok | 29.42 tok/s | 132 tok @ 205.1 tok/s |
| 2177 | 1,290 tok | 28.96 tok/s | 1,852 tok @ 301.9 tok/s |
| 3470 | 93 tok | 29.40 tok/s | 946 tok @ 265.1 tok/s |
| 4379 | 610 tok | 28.71 tok/s | 829 tok @ 251.0 tok/s |
| 4993 | 186 tok | 28.80 tok/s | 303 tok @ 223.4 tok/s |
| 5181 | 305 tok | 28.70 tok/s | 328 tok @ 232.7 tok/s |
| 5488 | 351 tok | 28.51 tok/s | 272 tok @ 208.3 tok/s |
| 5841 | 200 tok | 28.53 tok/s | 292 tok @ 222.1 tok/s |
| 6043 | 152 tok | 33.13 tok/s | 2,565 tok @ 497.1 tok/s |
| 6199 | 232 tok | 28.49 tok/s | 320 tok @ 235.2 tok/s |

Decode settled at **28.5–28.8 tok/s (avg ~28.7)** by 35–45K context; the early 34 tok/s figures are small-context turns. VRAM: **13.77 GB of 16 GB** (15.35 GB with MTP — the draft context costs ~1.58 GB on top of the packed weights + KV).

**Honest read of the A/B:** the baseline band (28.5–28.8 at 35–45K) sits ~1.5K tokens *above* the MTP band's top (37K), where this card's spec-on decode is already at its floor (44.5 tok/s). MTP-on therefore gained at least **+54%** over the same card's unassisted decode at adjacent context — the true figure at equal context is probably a few points higher, but we do not claim more than the measured numbers show. Two structural notes: (1) without MTP the 90K slot fits with ~2.2 GB to spare, so the spec-on deployment was *not* forced by VRAM — MTP bought speed, not context; (2) the baseline's cold prefill (460–497 tok/s) is faster than the MTP serve's (239–314 tok/s) — the draft head's overhead lands on prompt processing, not just decode.

**p-min 0.60 vs 0.75, A/B on the same serve** (container restarted between arms, same 400-token probe plus live agent traffic):

| setting | Acceptance | Decode | Verdict |
|---|---|---|---|
| **p-min 0.75 (n-max 3)** | 0.87–0.96, avg ~0.93 | ~47–48 tok/s | **shipped** |
| p-min 0.60 (n-max 3) | 0.67–0.77 | ~50.5 tok/s idle, +7% | rejected |

At 0.60 the head drafts past its confidence gate and acceptance drops ~0.15–0.25; the speed gain is a flat ~7% idle and the long-turn acceptance collapses to 0.67 under real traffic. This card is the counterpoint to the RX 9070 pair's finding that 0.60 "makes deeper n-max nearly free": with 16 GB at 96% VRAM and n-max 3, the gate at 0.75 already keeps rejection cheap, and loosening it just buys noise. The 0.60 number is also a single-run screen (the 9070 section's own warning applies: medians of 3 before deploying), but the acceptance delta is too large to ignore.

What this section adds:

- **Spec-off baseline back-filled 2026-08-16:** 28.5–28.8 tok/s unassisted at 35–45K context versus 44.5–53.8 with MTP — at least **+54%** at adjacent context on the same card (context bands differ by ~1.5K tokens; equal-context figure would be a few points higher, not claimed).
- **The deployment was speed-driven, not VRAM-forced:** without MTP the same 90K slot fits with ~2.2 GB to spare (13.77 GB of 16 GB). MTP cost ~1.58 GB of draft context and bought the speed.
- **n-max 3 holds as a daily driver at 0.93 average acceptance on RDNA3 16 GB** — the 5090 sweep's "run 2 as your daily" did not apply here; at 0.75 the third draft slot was accepted almost all the time, so there was nothing to give back.
- **VRAM headroom is a hidden variable.** Every row in the main table runs with 40%+ of the card's VRAM free; this one runs at 96%. The acceptance and speed bands above are what MTP does on a fully packed 16 GB card, not a comfortable 24 GB one.
- **The p-min knob is workload- and VRAM-dependent**, not just card-size-dependent: 0.60 won on the 2×9070 pool, 0.75 wins here.

### RTX 3090 24GB: the KV cache is a third tuning knob (q4_0 vs turbo3, both MTP-on)

Data: [@lsunay](https://x.com/lsunay1), Hermes agent on PC-18.

Same host, same quant (unsloth Q4_K_M), same spec flags on both arms: `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75 --spec-default`, `-np 1`, `-fa on`. Only the KV cache and the llama.cpp build differ — this is a cache A/B, not a spec-on/off A/B, and both arms are MTP-on.

| arm | KV cache | Context | llama.cpp build | Method | Decode | Acceptance |
|---|---|---|---|---|---|---|
| baseline | q4_0 | 185K | mtp-clean fork (llama27b-mtp:cuda) | live agent traffic, 14 turns (100–2700 tok) | **37.2 avg (29.3–45.4)** | 0.57 avg (0.47–0.69) |
| flag | **turbo3** | **200K (204800)** | turboquant v10465 (fca3093c9) | clean probe, 3×400 tok, medians of 3 | **54.2 avg (48.06–58.34)** | 0.64–0.84 (avg ~0.76) |

**+47% decode** on the same card with MTP active on both sides (37.2 → 54.2 avg; 38.2 → 56.2 median), and the acceptance band actually *rises* (0.57 → 0.64–0.84) — the faster verification pass leaves less draft-verify pressure per round. VRAM: 23.0 GB (q4_0, 185K) vs 23.3 GB (turbo3, 200K (204800)) of 24.6 GB.

Honest caveats, in order of size:

1. **Confounded build.** The two arms run different llama.cpp builds (an older `mtp-clean` fork vs turboquant v10465, ~2 months newer upstream). Part of the +47% is almost certainly the newer build's improved CUDA kernels, not the KV cache alone. A clean test would run both cache types on the same build; that was not possible here because the production `mtp-clean` image predates `--spec-default` and the turboquant image is the only one with turbo3.
2. **262K does not fit.** The turbo3 arm was planned at 262K (the repo's standard window) but the container failed to load the model at that size on 24 GB; 200K (204800) was the working ceiling. The q4_0 arm's 185K is its own working ceiling at `--cache-ram 10384`.
3. **Arm methods differ.** The baseline arm is live Hermes agent traffic (thinking on, real tool-call turns, 100–2700 output tokens) — realistic but noisier. The flag arm is a clean 3×400-token probe — controlled but short, which per the length rule underestimates the flag arm's advantage (spec scales with generation length).

What this section adds:

- **The KV cache type is a hidden third knob.** The community rules so far cover n-max and p-min; on a fully packed 24 GB card, q4_0 → turbo3 at n-max 3 is another large delta with the spec head already engaged.
- **Turbo3 + MTP + 204K fits a 24 GB card at ~95% VRAM** — the A6000 row shows 256K q8_0 at 40 GB; this is the 24 GB equivalent.
- **Acceptance rising under a faster serve is plausible**: verification gets cheaper, so the gate (p-min 0.75) passes more drafts. Treat as an observation, not a law.

### RTX 5090 32GB: n-max sweep

I've ran it in both configs, but n-max comparison table only for the second config that I actually run.
Config 1: Unsloth Q4_K_M, Context 131K, Q4_0 KV Cache, VRAM: 20.38GB
Config 2: Unsloth UD_Q4_K_XL, Context 262K, Q4_0 KV Cache, MMPROJ loaded, VRAM: 24.75GB
| n-max | Overall | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Acceptance |
|---|---|---|---|---|---|
| 2 | 151.1 | 167.0 | **129.4** | 151.1 | 0.58–0.95 |
| 3 | 163.2 | 192.2 | 126.4 | 163.2 | 0.44–0.91 |
| **4** | **177.6** | **207.2** | 113.1 | **177.6** | 0.34–0.86 |
| 5 | 156.6 | 215.8 | 100.9 | 156.6 | 0.28–0.87 |
| 6 | 164.3 | 211.3 | 95.4 | 164.3 | 0.24–0.86 |

**n-max 4** is the sweet spot (Interestingly, with 131K it wasn't any different, but with 262K it was). 

### RTX PRO 6000 Blackwell Max-Q 96GB: n-max and p-min tuning

Same RTX PRO 6000 Blackwell Max-Q host and serving configuration as the
community row above: lmstudio-community/Qwen3.8-27B-GGUF Q8_0, 131K context,
q4_0 K/V cache, llama.cpp 0.1.0-dev build 10454 (`4df29be4f`), Linux/CUDA,
`--parallel 1`. Upstream `probe.py` was unchanged; all reported prompt values
are medians of three runs with thinking off.

The main-table A/B is spec-off versus n-max 2. The sweep below is additional
tuning and does not replace that comparable n-max 2 row.

#### Ungated n-max sweep

| n-max | Overall | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Acceptance |
|---|---:|---:|---:|---:|---:|
| spec off | 45.7 | 46.0 | 46.0 | 45.0 | — |
| 1 | 73.9 | 76.7 | 67.0 | 73.9 | 0.69-0.97 |
| **2** | **97.1** | 101.8 | 75.5 | 97.1 | **0.52-0.95** |
| 3 | 101.0 | 121.3 | **78.8** | 101.0 | 0.44-0.96 |
| **4** | **108.1** | 129.6 | 71.3 | **108.1** | 0.35-0.89 |
| 5 | 103.5 | **141.5** | 60.7 | 103.5 | 0.24-0.86 |
| 6 | 104.9 | 140.4 | 65.8 | 104.9 | 0.27-0.81 |

The comparable n-max 2 arm more than doubles the baseline, from 45.7 to
97.1 tok/s (**+112.5%**). Ungated throughput continues rising beyond n-max 2
on this card and peaks overall at n-max 4, 108.1 tok/s (**+136.5%** versus
spec-off). The workload split is pronounced: Python continues to n-max 5 at
141.5 tok/s, while prose peaks at n-max 3 and falls sharply as the draft
window deepens.

Acceptance follows the same split. At n-max 4 the code prompt still accepts
roughly 0.82-0.89 of drafted tokens, while the prose prompt is only about
0.35-0.38. At n-max 5 prose falls to roughly 0.24-0.29 while Python remains
above 0.80. This made the host a useful case for testing the second knob,
`--spec-draft-p-min`.

#### p-min sweep at n-max 5

N-max was fixed at 5, where ungated Python throughput was highest, and
`--spec-draft-p-min` was swept from 0.20 through 0.80.

| n-max | p-min | Overall | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Aggregate acceptance |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.00 | 103.5 | 141.5 | 60.7 | 103.5 | 54.3% |
| 5 | 0.20 | 105.5 | 137.5 | 68.8 | 105.5 | 58.2% |
| 5 | **0.30** | **115.5** | **147.4** | **73.6** | **115.5** | 64.6% |
| 5 | 0.40 | 110.9 | 140.9 | 67.6 | 110.9 | 64.8% |
| 5 | 0.50 | 106.3 | 145.5 | 66.0 | 106.3 | 70.7% |
| 5 | 0.60 | 110.9 | 142.6 | 71.5 | 110.9 | 77.1% |
| 5 | 0.70 | 106.8 | 132.7 | 68.9 | 106.8 | 82.1% |
| 5 | 0.80 | 99.2 | 124.4 | 63.2 | 99.2 | 85.8% |

P-min 0.30 is the best tested gate at n-max 5. It raises overall throughput
from 103.5 tok/s ungated to 115.5 tok/s (**+11.6%**), while improving all
three prompts: Python 141.5 -> 147.4, prose 60.7 -> 73.6, and Bash
103.5 -> 115.5 tok/s. Against spec-off, the tuned result is **+152.7%**.

Higher p-min values increase aggregate acceptance in this sweep, but not
throughput. By p-min 0.80 acceptance reaches 85.8% while overall throughput
falls to 99.2 tok/s. Acceptance alone is therefore not the tuning target; the
useful point balances speculative depth, confidence gating, and verification
cost.

#### Gated depth check at p-min 0.30

The best p-min was then held fixed while n-max was varied around and above the
winning depth.

| n-max | p-min | Overall | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Aggregate acceptance |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.30 | 98.5 | 130.7 | **78.0** | 98.5 | 66.1% |
| **5** | **0.30** | **115.5** | 147.4 | 73.6 | **115.5** | 64.6% |
| 6 | 0.30 | 110.1 | 147.1 | 71.0 | 110.1 | 56.4% |
| 7 | 0.30 | 110.0 | **149.9** | 69.2 | 110.0 | 51.7% |
| 8 | 0.30 | 111.6 | 144.2 | 65.0 | 111.6 | 47.1% |

The gate does not improve every depth: n-max 4 with p-min 0.30 falls to
98.5 tok/s, below the 108.1 tok/s ungated n-max 4 result. The two knobs
interact rather than contributing independently.

The best tested overall `probe.py` setting on this host is therefore
n-max 5 / p-min 0.30 at 115.5 tok/s. Going deeper does not improve mixed
throughput. Python reaches 149.9 tok/s at n-max 7, but that is only 1.7% above
n-max 5 and comes with lower prose throughput, lower aggregate acceptance,
and greater run-to-run variance.

## 2× Tesla P40 24GB (tensor split): n-max sweep on Pascal

Same box as the row above — two P40s (sm_61, 732 GB/s each, no P2P, separate NUMA nodes), unsloth UD-Q5_K_XL, 131K context, q4_0 KV, `--tensor-split 1,1` under `numactl --interleave=all`, `--parallel 1`, thinking off. `probe.py` unchanged, medians of three runs x three prompts, only the spec flags changing between arms. Draft acceptance from the server log:

| config | Overall median | Overall mean | Prefill 20K | Acceptance |
|---|---|---|---|---|
| spec off | 11.7 | 11.8 | 354 | — |
| n-max 2, p-min 0.75 | 21.1 | 20.0 | 300 | 0.73-0.94 |
| **n-max 4, p-min 0.75** | **22.6** | **21.6** | 295 | 0.68-0.87 |
| n-max 6, p-min 0.75 | 21.1 | 20.5 | 290 | 0.63-0.81 |
| n-max 6, p-min 0.60 | 20.1 | 19.7 | 290 | 0.46-0.74 |

**+92% at the peak (22.6 vs 11.7) — the largest multi-GPU delta in the table, on the oldest architecture here.** Pascal has no tensor cores and the dense 27B decode is bandwidth-starved exactly like the 890M and RX 9070 rows, so the rule-1/rule-2 shape holds: amortising the weight read across accepted drafts is the whole game.

Two specifics that differ from the newer cards:

- **n-max 4 beats n-max 6, not the other way around.** The 3×3090 and 5060 Ti tensor rows peak at 3-4 too, but here n-max 6 loses ground (acceptance 0.63-0.81 vs 0.68-0.87) and n-max 8 was not run because the trend was already falling. The verification batch on sm_61 is expensive enough that the deeper draft stops paying.
- **p-min 0.75 helps, p-min 0.60 hurts** — the RX 9070 direction, not the 5090-desktop inversion. Gating at 0.60 collapsed acceptance to 0.46-0.74 and cost ~2 tok/s. On this class of card the gate rescues the deep draft; it is not a vanity knob.

The prefill tax is the usual ~15-17% (354 → 295 tok/s), the standard cost of device-to-host embedding transfers noted in the caveats section.

Production follow-up on the same box, **f16 KV instead of q4_0** (the cards have room; q4_0 only matters when the 131K-262K context must fit a tight pool): n-max 4 at p-min 0.75 still measures 22.3 mean / 24.0 median over the same probe — the q4_0 in the table row was a fit decision, not a speed one, and f16 KV reproduces it. Daily mixed use: 4.

### Companion: the MTP gain and the n-max optimum are per-model

Same box, same build (b10453), same method, each model at its production context. KV cache: q4_0 for the Qwen3.8-27B arm (the paired A/B above), f16 for the rest — the production follow-up verified f16 reproduces the 3.8 peak, so the table stays comparable. Baseline / with-flag decode medians (tok/s), `probe.py` unchanged:

| Model (UD-Q5_K_XL) | Baseline | Best spec arm | Peak | Gain | Acceptance |
|---|---|---|---|---|---|
| Qwen3.8-27B dense | 11.8 | n-max 4, p-min 0.75 | 22.6 | +92% | 0.68-0.87 |
| Qwen3.6-27B dense | 12.0 | n-max 6, p-min 0.75 | 22.2 (median 24.3) | +85% | 0.78-1.00 |
| Qwen3.6-35B-A3B MoE | 49.6 | n-max 4, p-min 0.75 | 67.1 | +35% | 0.83-0.99 |
| Qwen3.5-9B | 34.2 | n-max 4, p-min 0.75 | 46.8 | +37% | 0.68-0.95 |
| Qwen3.5-4B | 53.8 | n-max 2, p-min 0.75 | 67.5 | +25% | 0.82-0.99 |

Three things the sweep adds to rule 1:

- **The optimum n-max moves between models of the same size and class.** Qwen3.8-27B (dense) peaks at n-max 4; Qwen3.6-27B (dense, same quant, same cards) peaks at n-max 6. Same hardware, same method, different winner — re-sweep when the model changes, not just when the GPU does.
- **MoE collapses the gain.** The 35B-A3B is 4× faster than the dense 27B with the flag *off*, and the flag only adds +25-35% on top of that instead of +85-92%. The active-3B parameters are not bandwidth-bound at batch 1, so there is less to amortise — rule 3 in its purest form, and probably the single most useful row for anyone picking between the two.
- **Smaller models want shorter drafts.** The 4B peaks at n-max 2 (67.5) and n-max 4 is already behind (66.3); the 9B at n-max 4. With the model's own weights fitting in a small fraction of the pool, the verify cost per drafted token dominates faster.

p-min 0.75 won on every arm here; the MoE arms kept 0.83-0.99 acceptance at n-max 4, so gating is nearly free where acceptance was healthy to begin with.

#### Independent Q4_K_M replication and production follow-up

A second Windows dual-5060-Ti host independently repeated the tensor-split sweep using Unsloth Q4_K_M rather than UD-Q4_K_XL. Same GGUF and serving configuration within every A/B; only the spec flags changed. Upstream `probe.py` was unchanged, with three runs x three prompts and thinking off.

| config | Overall median | Overall mean | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Aggregate acceptance |
|---|---:|---:|---:|---:|---:|---:|
| spec off | 38.3 | 38.3 | 38.3 | 38.5 | 38.3 | — |
| n-max 2 | 72.0 | 69.5 | 79.2 | 56.3 | 72.0 | 78.1% |
| **n-max 3** | **76.0** | **73.8** | 86.6 | **57.5** | **76.0** | 71.4% |
| n-max 4 | **76.0** | 72.8 | **94.4** | 51.2 | **76.0** | 64.8% |

The result reproduces the main finding above with a different quant and host: tensor split lifts the spec-off baseline from 23.1 tok/s under layer split to 38.3 (+66%), before MTP. Tensor n-max 3 reaches 76.0 tok/s, 3.29x the original layer-split baseline. N-max 3 is the mixed-workload choice because it has the highest mean and prose result; n-max 4 remains the code-specialized arm.

A separate production-layout follow-up loaded the BF16 vision projector and used q8_0 for both main and draft KV. At 65,543 active prompt tokens, tensor n-max 3 decoded at 58.75 tok/s; at 129,007 tokens it decoded at 44.26 tok/s after 707.4 tok/s prefill. Both depths returned three hidden needle codes exactly. Two deterministic 1,200-token agent tasks decoded at 62.9 tok/s for business operations and 83.4 tok/s for Python, and their streamed responses matched buffered output byte-for-byte.

One Windows-specific memory result: `--load-mode none` reduced physical RAM added after load from 16.33 GiB to 1.85 GiB versus default `auto`/mmap. The same two 1,200-token outputs remained byte-identical, with effectively unchanged decode (63.18/83.58 versus 63.06/83.95 tok/s). Executable worker gates also passed literal CRUD (5/5), deterministic scoring (5/5), and structured-agent JSON validation. These are supplemental production checks, not part of the main-table paired A/B.

### RX 7900 XTX (Vulkan): n-max sweep, gate A/B, the desktop tax, and flag stacks don't travel

Same card, same config as the row, unchanged `probe.py`, medians of three runs x three prompts. Acceptance from server logs (per-request range, aggregate in parentheses):

| n-max | Overall median | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Acceptance |
|---|---|---|---|---|---|
| off | 28.8 | 28.6 | 28.8 | 28.9 | — |
| 2 | 57.9 | 61.8 | 47.2 | 57.9 | 0.51-0.95 (0.80) |
| **3** | **70.7** | 78.8 | **50.4** | **70.7** | 0.43-0.95 (0.72) |
| 4 | 70.2 | **86.3** | 49.7 | 70.2 | 0.32-1.00 (0.63) |
| 4, `p-min 0.60` | 60.7 | 79.9 | 39.6 | 60.7 | 0.53-0.94 (0.78) |

Same shape as every sweep above: code climbs through n-max 4, prose peaks at 3, acceptance decays monotonically. Overall optimum is 3 (+145% over spec-off, +22% over the launch-command n-max 2), with 4 a statistical tie on the strength of code alone — daily mixed use 3, pure code 4, prose-heavy 2. And rule 2 inverts here exactly as on the three desktop 5090s: the 0.60 gate raised aggregate acceptance from 0.63 to 0.78 while dropping the overall median 13% and prose by 20%. Vulkan verification on this card is cheap enough that every draft is worth attempting — another card class where acceptance is a vanity metric.

Two further screens from the same card, both from a custom streaming harness (not `probe.py`), labeled as screens per the contributing rules:

- **A shared desktop halves everything, silently.** Serving IQ4_XS at 16K next to a live compositor and browser holding 7.3 GB, 3.5 GB of weights spilled to GTT (host RAM over PCIe) and decode read 16.8 baseline / 21.4-29.7 with n-max 2 — prompt processing fell from ~740 to ~60-200 tok/s. The server starts fine and `/health` is green; nothing tells you the weights aren't resident. Check `mem_info_gtt_used` (or your vendor's equivalent) before trusting any number measured on a desk machine.
- **Rule 1 extends to whole flag stacks.** Importing the Strix Halo config from the issues verbatim (`draft-mtp,ngram-mod --spec-draft-n-max 12 --spec-ngram-mod-n-min 24`) measured 7.7 tok/s on prose — *below the unassisted baseline* — and 17.5 on code, against 21.4/29.7 for plain n-max 2 on the same degraded setup. Deep drafts plus ngram chaining that pay on a bandwidth-starved 256-bit APU invert on a 960 GB/s card. Re-derive the stack on your own hardware class, not just n-max.

### RTX 3090 24GB: two quants on one card, and two things that do nothing

Both rows above are the same machine, same build, same afternoon, so the pair
isolates the quant. The overclock (memory +1200, core +180, 430 W) is on for
every number here.

| quant | size | baseline | MTP n=2 | gain | acceptance |
|---|---:|---:|---:|---:|---:|
| unsloth UD-Q2_K_XL | 10.7 GB | 52.4 | 85.6 | +63% | 0.76 |
| unsloth UD-Q4_K_XL | 17.9 GB | 43.9 | 79.6 | +81% | 0.78 |

**The quant moves the baseline more than the flag choice does.** Going from the
17.9 GB file to the 10.7 GB one lifts the unassisted floor 43.9 -> 52.4, which
is most of the distance between this card and the other 3090 rows here. Worth
knowing before comparing baselines across rows: a row's quant size sets its
floor, so `Q4_K_M` rows and `UD-Q2_K_XL` rows are not measuring the same thing.

**The overclock is worth about +6.5%, not more.** On the matched Q4 file this
card baselines 43.9 against 41.3 on @hauntedhost's stock b10450 3090. I mention
the size because I nearly reported the overclock as a +27% effect by comparing
my Q2 baseline to their Q4 baseline, which is not a like-for-like comparison.

**`-fa 1` is already on, so adding it changes nothing.** `--flash-attn`
defaults to `auto` and auto resolves to on for this path. The proof is VRAM,
not speed: baseline and `-fa on` load 16338 and 16340 MiB, while `-fa off`
loads 23782 MiB, 7.4 GB more, because the unfused path materialises the
attention buffers. Anyone adding the flag expecting a gain will measure noise.

**`--spec-type draft-mtp,ngram-mod` segfaults here.** It dies mid-generation in
`server_context_impl::update_slots()`, reproducibly, on the third prompt. The
GMK EVO-X2 row calls that combination unstable; on this box it is a crash. The
same run also shows the repetition artifact that row warns about: with
ngram-mod the code prompt's three passes were 111.1 cold, then 124.4 and 122.5
warm, against 112.7 for MTP alone. `probe.py` sends each prompt three times, so
n-gram scores its own cache on runs two and three. `ngram-cache` and
`ngram-map-k` both ran clean and both lost (81.9 and 81.3 overall against 90.9
for MTP alone at n-max 4).

**Nothing else moved it.** Tested and rejected on this card, all against a
90 +/- 1 overall baseline at n-max 4: depths 3, 5 and 6; p-min 0.0 and 0.4;
KV cache q8_0 and q4_0; draft-model KV q8_0 (best code number of anything I
ran, 117.5, but worse overall); ubatch 256 and 1024. Power sweep from 250 W to
480 W: 390 W and 430 W are within noise of each other overall, 480 W is +6%,
and efficiency peaks around 300 W.

**A note on your overall column.** It is the median of nine runs, which on this
rig lands on the bash prompt, and that prompt moves about ±8 tok/s between
otherwise identical passes. I had a prose figure shift 5 tok/s and an
acceptance figure shift 0.04 between single passes, enough to reverse a
conclusion I had already written down. Every number in my rows is the median of
three complete probe passes rather than one.

## License

Apache-2.0. The numbers and verdicts are real, the conclusions are mine.
