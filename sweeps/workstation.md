# Workstation cards — contributor sweeps and studies

Contributor-authored deep dives for Qwen3.8-27B MTP, moved verbatim from the main README. Each section carries its author and original PR. Add yours via a PR to this file, row and footnote go in the main [community table](../README.md#community-numbers).

### A6000 48GB: n-max sweep
*by [@lingster](https://github.com/lingster), PR #2*

Same A6000, same config as the row above, `--spec-draft-n-max` swept 2-6. Overall and per-prompt probe medians (tok/s), draft acceptance from the server log:

| n-max | Overall | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Acceptance |
|---|---|---|---|---|---|
| 2 | 52.5 | 57.0 | 43.1 | 52.5 | 0.54-0.98 |
| 3 | 60.7 | 67.6 | 44.1 | 60.7 | 0.42-0.91 |
| **4** | **64.1** | 77.1 | 41.6 | 64.1 | 0.32-0.93 |
| 5 | 62.8 | 80.5 | 40.8 | 62.8 | 0.29-0.84 |
| 6 | 58.6 | 84.3 | 37.4 | 58.6 | 0.23-0.84 |

The overall peak is n-max 4 here, not 2 — the card has enough headroom to absorb the cost of deeper verification before the acceptance decay eats the win. Same shape as the 5090 sweep: the code prompts keep rising all the way up (84.3 at n-max 6), the prose prompt falls from the start (43.1 -> 37.4), and acceptance decays monotonically. Daily mixed use: 4, pure code sessions: 5-6, prose-heavy: 2.


### RTX PRO 6000 Blackwell Max-Q 96GB: n-max and p-min tuning
*by [@awilliamson](https://github.com/awilliamson), PR #28*

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

