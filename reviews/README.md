# Reviews

Agent reviews of [topics/](../topics/). They follow the review loop in
[GUIDELINES.md](../GUIDELINES.md): agents review first, Claude Code checks their key claims, and
I read the result.

## Panel

[panel/panel.sh](panel/panel.sh) runs three Antigravity agents (Gemini 3.8 Flash, High), one per
role:

| Role | Lens | IDs |
|---|---|---|
| [Lead physicist](panel/roles/physicist.md) | Physics, conventions, sources, intuition | P1, P2, ... |
| [Senior software engineer](panel/roles/engineer.md) | Code matches README, real tests, simplicity | E1, E2, ... |
| [Mathematician](panel/roles/mathematician.md) | Precise claims, cited theorems, clean formulations | M1, M2, ... |

They review the topics not only for errors and gaps, but for what could be **simplified** and
what a **newcomer** needs explained so the topics read as one path.

1. [Round 1](panel/round-1.md): each reviews independently.
2. [Round 2](panel/round-2.md): each reads the other two reviews and answers every finding
   (agree, disagree or refine), then adds what all three missed.
3. [Round 3](panel/round-3.md): each answers the objections to its own findings and gives a final
   ranked list, labelled consensus, majority or disputed.
4. [Report](panel/report.md): the physicist chairs and writes `REPORT.md`.

Agents in a round run in parallel, each in its own throwaway copy of the repository. They see each
other only through the previous rounds' outputs. The prompt given to an agent is its role file
followed by the round file, verbatim. The round outputs are the agents' untouched text. The only
change is that absolute local paths in links were removed, leaving paths relative to the copy of
the repository the agent reviewed.

```bash
reviews/panel/panel.sh reviews/YYYY-MM-topics      # MODEL, CHAIR, TIMEOUT can be overridden
```

## Runs

| Run | Topics reviewed | Result |
|---|---|---|
| [2026-09-topics](2026-09-topics/) | 00–03 | [REPORT.md](2026-09-topics/REPORT.md): 14 changes. [CHECK.md](2026-09-topics/CHECK.md): all 6 code findings confirmed; 1 physics error and several unsourced claims in the prose. [APPLIED.md](2026-09-topics/APPLIED.md): what changed |
