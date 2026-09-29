# Tensor networks: topics and AI agent evals

A personal side project on tensor networks for quantum circuit simulation, organised in two
tracks:

| Track | What it is | How seriously to take it |
|---|---|---|
| [**topics/**](topics/) | One idea per folder, taken from a named textbook or paper, implemented minimally and **checked against an exact reference** | Meant to be correct. Small on purpose. |
| [**agent-evals/**](agent-evals/) | Evaluations of AI agents: one fixed setup and prompt, several models, outputs scored and compared | **For fun.** The point is the evaluation, not the agents' research. |

> **AI disclosure.** AI does most of the writing here, and it's labelled everywhere.
> - **Agent evals:** the agents' outputs are entirely AI-written (Gemini models in Google Antigravity).
> - **Topics:** code is written with AI assistance (Claude Code). I choose the source, the
>   claim and the checks, and each topic README says who wrote it and whether I have reviewed it.
>
> Nothing here is peer-reviewed.

My main research is not here. My Master's thesis was on tensor network contraction for quantum
circuit simulation on processing-in-memory architectures, with a proof of concept. Reproductions
of specific papers will get their own standalone repositories.

## Topics so far

| # | Topic | Established and checked |
|---|---|---|
| 00 | [Tensor network basics](topics/00-tensor-network-basics/) | The order of pairwise contractions decides the memory and time needed; a network can count graph colourings |
| 01 | [Circuits as tensor networks](topics/01-circuits-as-tensor-networks/) | Closed/open circuit networks reproduce the state vector exactly (GHZ, QFT, random U(4)) |
| 02 | [Contraction order and cost](topics/02-contraction-order-and-cost/) | Best contraction width is constant on a ring and grows like L on an L×L grid |
| 03 | [Slicing](topics/03-slicing/) | Slicing trades memory for work: 8× less memory costs ~14% more, 256× less costs ~1300× more |

Next: parallel contraction, MPS and SVD truncation, MPS circuit simulation (TEBD), DMRG.
See [PLAN.md](PLAN.md).

## Running

```bash
pip install -r requirements.txt
pytest                      # runs topics/ only
python topics/02-contraction-order-and-cost/order.py
```

## Layout

```
topics/         checked, minimal implementations (Python; other languages welcome side by side)
agent-evals/    agent evaluations: prompt, launcher, outputs and review, one dated folder per run
reviews/        agent panel reviews of topics/ (physicist, engineer, mathematician)
GUIDELINES.md   rules for both tracks
references/     sources (redistributable PDFs included, others via fetch.sh)
PLAN.md         roadmap
```

MIT licence. To cite, see [CITATION.cff](CITATION.cff).
