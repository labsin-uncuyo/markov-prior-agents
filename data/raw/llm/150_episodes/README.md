# 150-episode LLM/ReAct evaluation

This folder contains the revised 150-episode LLM/ReAct experiment used in the manuscript response to reviewer concerns about the original small LLM evaluation.

## Baseline

Folder: `baseline/`

- Agent: `NetSecGameAgents/agents/attackers/mc_llm_qa_baseline/llm_agent_qa.py`
- Planner: plain LLM/ReAct action planner without Markov-chain guidance
- Model: `gemma3:4b`
- Provider/API: Ollama OpenAI-compatible endpoint, `http://127.0.0.1:11434/v1/`
- Episodes: 150
- Result summary: 0 wins, 0 detections, 150 timeouts

## Markov-guided

Folder: `mc_guided/`

- Agent: `NetSecGameAgents/agents/attackers/mc_llm_qa/mc_llm_agent_qa.py`
- Planner: MC-guided action-type selection plus LLM/ReAct grounding
- Model: `gemma3:4b`
- Provider/API: Ollama OpenAI-compatible endpoint, `http://127.0.0.1:11434/v1/`
- Episodes: 150
- Result summary: 17 wins, 0 detections, 133 timeouts

## Files

- `episode_data.json.gz`: compressed per-episode data, including states, prompts/responses when recorded, evaluation scores, and end reasons.
- `llm_react.log.gz`: compressed runtime log used to compute wall-clock and LLM-call overhead.
- `summary.txt`: human-readable summary of the run.
- `NSG_coordinator.log.gz`: coordinator log, available for the baseline bundle.

Processed summaries are generated with:

```bash
python3 scripts/analysis/llm/summarize_llm_results.py --dataset 150_episodes
```

Outputs are written to:

```text
data/processed/llm/150_episodes/
```
