#!/usr/bin/env bash
# Blind audit: copy the four studies under anonymous names into audit/work/, then let one
# auditor agent (Gemini 3.8 Flash, High) review all of them. Blocks until the auditor finishes.
set -uo pipefail
cd "$(dirname "$0")"

MODEL="gemini-3.8-flash-high"
LETTERS=(A B C D)
STUDIES=(gemini-3.7-flash-high gemini-3.1-pro-high gemini-3.8-flash-high gemini-3.6-flash-high)  # shuffled

rm -rf work && mkdir -p work
echo "# Key (not shown to the auditor)" > KEY.md
for i in 0 1 2 3; do
    cp -r "../${STUDIES[$i]}" "work/study-${LETTERS[$i]}"
    rm -rf "work/study-${LETTERS[$i]}/__pycache__" "work/study-${LETTERS[$i]}/agent.log" \
           "work/study-${LETTERS[$i]}/transcript.md"
    echo "- study-${LETTERS[$i]}: ${STUDIES[$i]}" >> KEY.md
done
sed -n '/^````text$/,/^````$/p' ../PROMPT.md | sed '1d;$d' > RESEARCH_PROMPT.md

PROMPT="$(sed -n '/^````text$/,/^````$/p' AUDIT_PROMPT.md | sed '1d;$d')"
agy -p "$PROMPT" --model "$MODEL" --effort high --new-project \
    --dangerously-skip-permissions --print-timeout 120m > transcript.md 2> agent.log
echo "exit=$?" >> agent.log
echo "auditor finished"
