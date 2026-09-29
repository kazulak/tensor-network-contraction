#!/usr/bin/env bash
# Panel review of topics/: three Antigravity agents in different roles review, discuss, and the
# chair writes a joint report. Each agent works in its own throwaway copy of the repository; they
# talk only through the files in <out>/round-*/ which are passed to the next round as ./panel/.
# The agents auto-approve their own tool calls, so each is confined to its copy only by the prompt.
#
# Usage: reviews/panel/panel.sh <out-dir>        e.g. reviews/2026-09-topics
# Progress: tail -f <out-dir>/work/*/agent.log
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
[ $# -eq 1 ] || { echo "usage: $0 <out-dir>"; exit 1; }
mkdir -p "$1" && OUT="$(cd "$1" && pwd)"

MODEL="${MODEL:-gemini-3.8-flash-high}"
ROLES="physicist engineer mathematician"
CHAIR="${CHAIR:-physicist}"
TIMEOUT="${TIMEOUT:-60m}"

# run_agent <step> <role> <prompt-file>: fresh workspace, earlier rounds in ./panel/, keep OUTPUT.md
run_agent() {
    local step=$1 role=$2 ws="$OUT/work/$1-$2"
    rm -rf "$ws" && mkdir -p "$ws/panel"
    cp -r "$REPO"/{topics,references,README.md,GUIDELINES.md,PLAN.md,requirements.txt,pytest.ini} "$ws/"
    find "$ws" \( -name __pycache__ -o -name .pytest_cache \) -prune -exec rm -rf {} +
    for f in "$OUT"/round-*/*.md; do
        [ -e "$f" ] && cp "$f" "$ws/panel/$(basename "$(dirname "$f")")-$(basename "$f")"
    done
    local prompt; prompt="$(cat "$HERE/roles/$role.md"; echo; cat "$HERE/$3")"
    ( cd "$ws" && agy -p "$prompt" --model "$MODEL" --effort high --new-project \
          --dangerously-skip-permissions --print-timeout "$TIMEOUT" \
          > transcript.md 2> agent.log; echo "exit=$?" >> agent.log )
    if [ -s "$ws/OUTPUT.md" ]; then
        mkdir -p "$OUT/$step" && cp "$ws/OUTPUT.md" "$OUT/$step/$role.md"
        echo "done $step $role"
    else
        echo "FAILED $step $role: no OUTPUT.md (see $ws/agent.log)"
    fi
}

for round in 1 2 3; do
    for r in $ROLES; do run_agent "round-$round" "$r" "round-$round.md" & done
    wait
    for r in $ROLES; do
        [ -s "$OUT/round-$round/$r.md" ] || { echo "stopping: round $round incomplete"; exit 1; }
    done
done

run_agent report "$CHAIR" report.md
[ -s "$OUT/report/$CHAIR.md" ] && mv "$OUT/report/$CHAIR.md" "$OUT/REPORT.md" && rmdir "$OUT/report"
echo "panel finished: $OUT/REPORT.md"
