#!/usr/bin/env bash
# Same prompt as run.sh, run by Claude Code headless (claude -p) instead of Antigravity.
# The agent works in a fresh empty directory outside the repository (no git history, no
# topics/03-slicing reference answer, no MCP servers, no user settings), auto-approving its own
# tool calls like the agy runs. Claude Code's bundled skills stay available (the 2026-09-29 run
# used the dataviz skill). Its files are then copied into <folder>/.
# Usage: ./run-claude.sh [model] [effort] [folder]     (WORK=<dir> to choose the scratch dir)
set -uo pipefail
cd "$(dirname "$0")"

PROMPT="$(sed -n '/^````text$/,/^````$/p' PROMPT.md | sed '1d;$d')"
MODEL="${1:-claude-sonnet-5-5}"
EFFORT="${2:-xhigh}"
OUT="${3:-claude-sonnet-5.5-xhigh}"
WORK="${WORK:-$(mktemp -d)}"
mkdir -p "$WORK/agent" "$OUT"
echo "model=$MODEL effort=$EFFORT work=$WORK/agent out=$OUT"

( cd "$WORK/agent" && timeout 90m claude -p "$PROMPT" --model "$MODEL" --effort "$EFFORT" \
      --dangerously-skip-permissions --strict-mcp-config --setting-sources "" \
      --no-session-persistence --output-format stream-json --verbose \
      > "$WORK/stream.jsonl" 2> "$WORK/stderr.log"; echo "exit=$?" > "$WORK/exit" )

# Final message -> transcript.md; exit code and run statistics -> agent.log.
python - "$WORK" "$OUT" <<'EOF'
import json, sys
work, out = sys.argv[1], sys.argv[2]
result = {}
for line in open(f"{work}/stream.jsonl", encoding="utf-8"):
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        continue
    if msg.get("type") == "result":
        result = msg
open(f"{out}/transcript.md", "w", encoding="utf-8").write(result.get("result", "") + "\n")
stats = {k: result.get(k) for k in ("subtype", "num_turns", "duration_ms", "total_cost_usd")}
with open(f"{out}/agent.log", "w", encoding="utf-8") as f:
    f.write(open(f"{work}/exit").read())
    f.write(json.dumps(stats) + "\n")
EOF
( cd "$WORK/agent" && find . -type f -not -path '*/__pycache__/*' -not -path '*/.pytest_cache/*' \
      -exec cp --parents {} "$OLDPWD/$OUT/" \; )
cat "$OUT/agent.log"
