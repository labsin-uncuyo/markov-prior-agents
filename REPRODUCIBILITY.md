# Reproducibility guide

This document maps the manuscript's main empirical results to the artifacts and scripts in this repository.

## Repository sections

- `data/raw/`: raw or preprocessed experiment outputs.
- `data/processed/`: CSV summaries derived from `data/raw/`.
- `figures/`: generated figures copied from the analysis workflow.
- `scripts/analysis/`: scripts for regenerating processed summaries from raw artifacts.
- `data/MANIFEST.md`: description of each data folder and file naming convention.

## Running the summarizers

From the repository root:

```bash
python3 scripts/analysis/markov_chain/summarize_markov_chain_results.py
python3 scripts/analysis/llm/summarize_llm_results.py --dataset 150_episodes
Rscript scripts/analysis/llm/plot_llm_action_outcomes.R 150_episodes
```

Use `python3 scripts/analysis/llm/summarize_llm_results.py --all` to regenerate both the revised 150-episode LLM summaries and the earlier 30-episode exploratory summaries.

The Markov-chain summarizer streams the compressed `*.json.gz` files line by line, so it does not load the large raw artifacts fully into memory.

## Manuscript mapping

| Manuscript item | Scenario / section | Raw input | Processing script | Processed output |
| --- | --- | --- | --- | --- |
| Full-scenario standalone MC-agent table | Full NetSecGame scenario | `data/raw/markov_chain/full/` | `scripts/analysis/markov_chain/summarize_markov_chain_results.py` | `data/processed/markov_chain/full/table_metrics.csv` |
| Three-network transfer table | Three-network NetSecGame scenario | `data/raw/markov_chain/three_network/` | `scripts/analysis/markov_chain/summarize_markov_chain_results.py` | `data/processed/markov_chain/three_network/table_metrics.csv` |
| Defender table | Full scenario with stochastic defender | `data/raw/markov_chain/defender/` | `scripts/analysis/markov_chain/summarize_markov_chain_results.py` | `data/processed/markov_chain/defender/table_metrics.csv` |
| Markov-chain action outcome distributions | ALL and BEPR action outcomes | `data/raw/markov_chain/full/` | `scripts/analysis/markov_chain/summarize_markov_chain_results.py` | `data/processed/markov_chain/full/action_outcome_summary.csv` |
| LLM/ReAct action outcome distribution | 150-episode baseline vs Markov-guided LLM/ReAct | `data/raw/llm/150_episodes/baseline/`, `data/raw/llm/150_episodes/mc_guided/` | `scripts/analysis/llm/summarize_llm_results.py --dataset 150_episodes` | `data/processed/llm/150_episodes/action_outcome_summary.csv` |
| LLM/ReAct success and step statistics | 150-episode baseline vs Markov-guided LLM/ReAct | `data/raw/llm/150_episodes/` | `scripts/analysis/llm/summarize_llm_results.py --dataset 150_episodes` | `data/processed/llm/150_episodes/agent_summary.csv` |
| LLM/ReAct runtime and LLM-call overhead | 150-episode baseline vs Markov-guided LLM/ReAct | `data/raw/llm/150_episodes/*/summary.txt`, `data/raw/llm/150_episodes/*/llm_react.log.gz` | `scripts/analysis/llm/summarize_llm_results.py --dataset 150_episodes` | `data/processed/llm/150_episodes/runtime_summary.csv` |

## Metric definitions

### ALL

ALL refers to all episodes or actions across all runs for a given agent and scenario.

### BEPR

BEPR refers to the best episode per run. For terminal success percentages, each run contributes one BEPR outcome: success if the run contains at least one successful episode, otherwise non-success. For BEPR action counts, only the shortest successful episode in each successful run contributes action counts.

For manuscript table step counts, the summarizer also reports `bepr_steps_mean` and `bepr_steps_sd`, where a successful run contributes the length of its shortest successful episode and an unsuccessful run contributes the maximum observed episode length for that run, typically the 100-step timeout. This convention preserves comparability with agents that do not solve every run.

### Action outcome categories

For Markov-chain experiments:

- `Effective`: reward/evaluation score 1 or 9.
- `Redundant`: reward/evaluation score 0.
- `Bad`: any other reward/evaluation score, typically negative rewards such as detection.

For LLM/ReAct experiments:

- `Effective`: score 8 or 10.
- `Invalid`: score 0.
- `Redundant`: all other scores, typically 3.

The revised manuscript reports the 150-episode LLM/ReAct experiment. For this experiment, action-outcome percentages count all evaluation entries, including the terminal successful action. Runtime and step summaries use the manuscript convention in which the terminal successful action is not counted as an additional step; therefore `agent_summary.csv` reports both `actions_including_terminal_*` and `manuscript_steps_*` fields.

## Notes and caveats

- The raw Markov-chain artifacts are compressed because the uncompressed files are large.
- The repository contains preprocessed experiment outputs rather than a complete containerized environment for rerunning NetSecGame from scratch.
- Exact NetSecGame task YAMLs, dependency lockfiles, and container recipes are not yet included.
- LLM prompts are available in `mc_llm_qa/prompts.yaml` and `mc_llm_qa_baseline/prompts.yaml`. For the 150-episode LLM experiment, the run summaries record the model (`gemma3:4b`), provider/API endpoint (Ollama OpenAI-compatible endpoint at `http://127.0.0.1:11434/v1/`), episode count, agent script, and wall-clock timing.
