# Multi-Agent Quantum & Algorithmic Research Repository

This repository serves as a generalized research environment for evaluating Large Language Model (LLM) agents on complex scientific and algorithmic problems. Multiple autonomous agents are deployed in isolated environments to design, implement, and benchmark solutions to selected research questions.

---

## Active Research Projects

### 1. Parallel Tensor Network Contraction Research
This project studies the acceleration of tensor-network contractions in Python and Julia via parallel processing, focusing on memory constraints, slicing, and execution speedup limits.

* **Master Reproduction Script**: [reproduce.sh](reproduce.sh)
  * Runs the entire reproduction pipeline: sets up the environment, verifies Julia, installs Python dependencies, and runs the comparative multi-topology scaling sweep on "hot" daemon infrastructure with a single command.
* **Main Workspace & Synthesized Findings**: [parallel_contraction_research/proper_research/](parallel_contraction_research/proper_research/)
  * Includes the final scientific paper, dependencies, and execution logs.
* **Hybrid Architecture Design**: [parallel_contraction_research/hybrid_python_julia_design.md](parallel_contraction_research/hybrid_python_julia_design.md)
  * Design blueprint combining Python (planning/slicing) with Julia (GIL-free parallel execution).
* **Autonomous Agent Workspaces**:
  * **Google Gemini 3.5 Flash (High)**: [parallel_contraction_research/gemini_3.5_flash/](parallel_contraction_research/gemini_3.5_flash/)
  * **Gemini Pro 3.1 (High)**: [parallel_contraction_research/gemini_pro_3.1/](parallel_contraction_research/gemini_pro_3.1/)
  * **GPT OSS 120b (Medium)**: [parallel_contraction_research/gpt_oss_120b/](parallel_contraction_research/gpt_oss_120b/)

---

## Repository Structure

* **[reproduce.sh](reproduce.sh)**: Master one-command execution script to reproduce the entire hybrid Python-Julia scaling benchmark.
* **[parallel_contraction_research/](parallel_contraction_research/)**: Workspace root for the tensor network contraction project.
  * **[proper_research/](parallel_contraction_research/proper_research/)**: Target folder containing consolidated research, requirements, and source code.
  * **[hybrid_python_julia_design.md](parallel_contraction_research/hybrid_python_julia_design.md)**: Design blueprint for hybrid Python-Julia execution.
  * **[gemini_3.5_flash/](parallel_contraction_research/gemini_3.5_flash/)**: Google Gemini 3.5 Flash results.
  * **[gemini_pro_3.1/](parallel_contraction_research/gemini_pro_3.1/)**: Gemini Pro 3.1 results.
  * **[gpt_oss_120b/](parallel_contraction_research/gpt_oss_120b/)**: GPT OSS 120b workspace.
* **[.github/workflows/](.github/workflows/)**: Contains continuous integration configurations.
* **[LICENSE](LICENSE)**: Project license (MIT License).
* **[README.md](README.md)**: Main entry point for the repository (this file).
