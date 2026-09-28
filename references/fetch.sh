#!/usr/bin/env bash
# Download arXiv versions of references that may not be redistributed here.
# They go into references/local/, which is git-ignored.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p local
for id in 1306.2164 1008.3477 1708.09213 quant-ph/0511069 1805.01450 2005.06787 \
          quant-ph/0301063 2002.07730 math-ph/0609050 quant-ph/0406196; do
    out="local/arXiv-${id//\//_}.pdf"
    [ -f "$out" ] || { curl -sSL -A "Mozilla/5.0" -o "$out" "https://arxiv.org/pdf/$id"; sleep 3; }
    echo "$out"
done
