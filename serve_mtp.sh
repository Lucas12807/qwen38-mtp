#!/bin/bash
# Qwen3.8-27B with the MTP flag on a 24GB card.
# Adjust the model path, everything else is the measured config.
MODEL="${1:-$HOME/models/qwen3.8-27b-dense/Qwen3.8-27B-Q4_K_M.gguf}"

llama-server -m "$MODEL" \
  -c 131072 -ngl 999 -fa 1 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --spec-type draft-mtp --spec-draft-n-max 2 --parallel 1 \
  --host 127.0.0.1 --port 8080
