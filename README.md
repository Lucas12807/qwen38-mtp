# Qwen3.8-27B MTP: the flag was free the whole time

One llama.cpp flag unlocks +33-39% decode speed for Qwen3.8-27B on consumer GPUs. No new files, no conversion, no custom build. The MTP head already ships inside the GGUF you downloaded on launch night.

Measured hours after the Aug 14 2026 release, published so every 24GB card owner gets the speed on day one.

## The numbers

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

## How it works

Qwen trained multi-token-prediction (nextn) layers into Qwen3.8. The quantizers kept them: unsloth's GGUFs carry the `blk.*.nextn.*` tensors, which llama.cpp loads and, without the flag, ignores. llama.cpp added draft-mtp speculative decoding in [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673) (July 2026): the server drafts tokens with the built-in head and verifies them with the main model, so accepted drafts cost a fraction of a full forward pass. Everything was in place on release night. The flag connects it.

## Caveats

- `--parallel 1` is required for now, single slot only
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
| RTX A6000 48GB (Ada) | 26.7 | 52.5 | 2 | n/a | [@lingster](https://github.com/lingster) |

\* A6000 row: unsloth Q8_K_XL, 256K context, q8_0 KV cache — 40.0 GB VRAM baseline, 41.4 GB with spec (rows above: Q4_K_M, 131K, q4_0 KV).

## License

Apache-2.0. The numbers and verdicts are real, the conclusions are mine.
