# Multi-Agent Quantum & Algorithmic Research Repository

This repository serves as a generalized research environment for evaluating Large Language Model (LLM) agents on complex scientific and algorithmic problems. Multiple autonomous agents are deployed in isolated environments to design, implement, and benchmark solutions to selected research questions.

---

## Active Research Projects

### 1. Parallel Tensor Network Contraction Research
This project studies the acceleration of tensor-network contractions in Python via parallel processing, focusing on memory constraints and execution speedup limits.

* **Main Workspace & Synthesized Findings**: [parallel_contraction_research/proper_research/](parallel_contraction_research/proper_research/)
  * Includes the synthesized research paper, dependencies, and smoke tests.
* **Autonomous Agent Workspaces**:
  * **Google Gemini 3.5 Flash (High)**: [parallel_contraction_research/gemini_3.5_flash/](parallel_contraction_research/gemini_3.5_flash/)
    * Workspace containing paper, speedup plots, and implementation.
  * **Gemini Pro 3.1 (High)**: [parallel_contraction_research/gemini_pro_3.1/](parallel_contraction_research/gemini_pro_3.1/)
    * Workspace containing paper, speedup plots, and implementation.
  * **GPT OSS 120b (Medium)**: [parallel_contraction_research/gpt_oss_120b/](parallel_contraction_research/gpt_oss_120b/)
    * Workspace (no files produced).

---

## Repository Structure

* **[parallel_contraction_research/](parallel_contraction_research/)**: Workspace root for the tensor network contraction project.
  * **[proper_research/](parallel_contraction_research/proper_research/)**: Target folder containing consolidated research, requirements, and smoke tests.
  * **[gemini_3.5_flash/](parallel_contraction_research/gemini_3.5_flash/)**: Google Gemini 3.5 Flash results.
  * **[gemini_pro_3.1/](parallel_contraction_research/gemini_pro_3.1/)**: Gemini Pro 3.1 results.
  * **[gpt_oss_120b/](parallel_contraction_research/gpt_oss_120b/)**: GPT OSS 120b workspace.
* **[.github/workflows/](.github/workflows/)**: Contains continuous integration configurations.
* **[LICENSE](LICENSE)**: Project license (MIT License).
* **[README.md](README.md)**: Main entry point for the repository (this file).

---

## Continuous Integration & Testing

The repository runs a validation pipeline via GitHub Actions to ensure that research workspaces are correctly set up and basic contractions execute error-free. The CI workflow installs all pinned python dependencies and executes a validation benchmark:

* Workflow definition: [.github/workflows/smoke_benchmark.yml](.github/workflows/smoke_benchmark.yml)
* Target test script: [parallel_contraction_research/proper_research/smoke_benchmark.py](parallel_contraction_research/proper_research/smoke_benchmark.py)
