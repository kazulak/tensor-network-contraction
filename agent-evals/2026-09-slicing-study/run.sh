#!/usr/bin/env bash
# Launch one Antigravity agent per Google model on the same prompt, each in its own folder.
# The agents auto-approve their own tool calls, so each one is confined to its folder only
# by the prompt. Blocks until all agents finish.
# Progress: tail -f <model>/agent.log ; final answer: <model>/transcript.md
set -uo pipefail
cd "$(dirname "$0")"

PROMPT="$(sed -n '/^````text$/,/^````$/p' PROMPT.md | sed '1d;$d')"
MODELS="gemini-3.8-flash-high gemini-3.7-flash-high gemini-3.6-flash-high gemini-3.1-pro-high"

for m in $MODELS; do
    mkdir -p "$m"
    ( cd "$m" && agy -p "$PROMPT" --model "$m" --effort high --new-project \
          --dangerously-skip-permissions --print-timeout 90m \
          > transcript.md 2> agent.log; echo "exit=$?" >> agent.log ) &
    echo "started $m (pid $!)"
done
wait
echo "all agents finished"
