# Markov Prior Agents

Experiment code for studying **Markov Chain behavioral priors** for autonomous cyber agents in NetSecGame.

The central idea is simple: learn a compact transition model over attack action types, then use that model both as a lightweight attacker policy and as guidance for more complex agents. The repository includes code for generating attack trajectories with a Genetic Algorithm, using the resulting Markov transition matrix directly, and reusing it as a prior for Q-learning and LLM/ReAct-style agents.

## What is included

- **Genetic Algorithm trajectory search**: searches for efficient attack sequences in NetSecGame.
- **Markov-guided LLM/ReAct agent**: uses a Markov Chain transition matrix to suggest the next action type, while the LLM fills in action parameters.
- **Baseline LLM/ReAct agent**: similar LLM-based agent without Markov Chain guidance.
- **Initialized Q-learning agent**: initializes Q-values using transition probabilities from the learned Markov model.

This repository contains experiment code and artifacts. The LaTeX manuscript is maintained separately.

## Relationship with NetSecGame and NetSecGameAgents

These agents run against the [NetSecGame](https://github.com/stratosphereips/NetSecGame) environment and reuse the agent interface from Diego Forni's NetSecGameAgents repository:

```text
https://github.com/diegoforni/NetSecGameAgents/tree/main
```

NetSecGame is a configurable network-security simulation framework for training and evaluating offensive and defensive AI agents. It uses `AIDojoCoordinator` to run the game server and `NetSecGameAgents` to provide reusable agent infrastructure.

This repository is an experiment-focused adaptation of several attacker-agent components from `NetSecGameAgents`, especially:

```text
NetSecGameAgents/agents/attackers/markov_chain_agent/
NetSecGameAgents/agents/attackers/initialized_q_learning/
NetSecGameAgents/agents/attackers/llm_qa/
```

Local folders map to those components as follows:

```text
markov-prior-agents/GA/genetic/              -> markov_chain_agent/genetic/
markov-prior-agents/initialized_q_learning/  -> initialized_q_learning/
markov-prior-agents/mc_llm_qa_baseline/      -> llm_qa/ baseline-style LLM/ReAct agent
markov-prior-agents/mc_llm_qa/               -> modified Markov-guided LLM/ReAct agent
```

The scripts import NetSecGame components such as:

- `AIDojoCoordinator.game_components.Action`
- `AIDojoCoordinator.game_components.ActionType`
- `AIDojoCoordinator.game_components.Observation`
- `AIDojoCoordinator.game_components.AgentStatus`
- `NetSecGameAgents.agents.base_agent.BaseAgent`

The standard NetSecGame agent lifecycle is:

1. create a `BaseAgent(host, port, "Attacker")`;
2. call `register()` once to join the game;
3. repeatedly submit actions with `make_step(action)`;
4. call `request_game_reset()` between episodes;
5. call `terminate_connection()` when finished.

The agents in this repository focus on attacker behavior and use the NetSecGame action types represented in the Markov transition matrices:

- `ScanNetwork`
- `FindServices`
- `ExploitService`
- `FindData`
- `ExfiltrateData`

## Repository structure

```text
markov-prior-agents/
├── GA/genetic/
│   ├── genetic_agent.py               # GA-based attack trajectory search
│   ├── config.json                    # GA hyperparameters
│   └── results/parsed_population.json # Existing GA output artifact
├── initialized_q_learning/
│   ├── initialized_q_agent.py         # Q-learning agent initialized from MC probabilities
│   └── transition_probabilities.json  # Transition matrix used for initialization
├── mc_llm_qa/
│   ├── mc_llm_agent_qa.py             # Markov-guided LLM/ReAct evaluation runner
│   ├── mc_llm_action_planner.py       # Markov-guided LLM action planner
│   ├── prompts.yaml                   # Prompts and action-format instructions
│   ├── transition_probabilities.json  # Markov transition probabilities
│   ├── validate_responses.py          # LLM response validation helpers
│   └── jsonGraph.py                   # Plotting script for episode outputs
└── mc_llm_qa_baseline/
    ├── llm_agent_qa.py                # Baseline LLM/ReAct evaluation runner
    ├── llm_action_planner.py          # Baseline planner without Markov guidance
    ├── prompts.yaml / prompts2.yaml   # Baseline prompts
    ├── validate_responses.py          # LLM response validation helpers
    └── jsonGraph.py                   # Plotting script for episode outputs
```

## Main components

### 1. Genetic Algorithm trajectory search

```text
GA/genetic/genetic_agent.py
GA/genetic/config.json
```

The GA agent searches for efficient attack trajectories in NetSecGame. These trajectories can then be analyzed to derive a Markov Chain transition matrix over action types.

`config.json` controls population size, number of generations, mutation/crossover settings, tournament selection, replacement strategy, and reward threshold.

### 2. Markov-guided LLM/ReAct agent

```text
mc_llm_qa/mc_llm_agent_qa.py
mc_llm_qa/mc_llm_action_planner.py
mc_llm_qa/prompts.yaml
mc_llm_qa/transition_probabilities.json
```

This is the main Markov-prior LLM agent. The Markov Chain proposes a high-level action type according to the current transition state. The LLM then grounds that action type into concrete NetSecGame parameters and produces a valid action JSON.

The transition matrix includes these states:

- `Initial Action`
- `ScanNetwork`
- `FindServices`
- `ExploitService`
- `FindData`
- `ExfiltrateData`

### 3. Baseline LLM/ReAct agent

```text
mc_llm_qa_baseline/llm_agent_qa.py
mc_llm_qa_baseline/llm_action_planner.py
```

The baseline agent uses the LLM/ReAct-style prompting flow without Markov Chain action-type guidance. It is useful for comparing how much the Markov prior reduces redundant or invalid behavior.

### 4. Initialized Q-learning agent

```text
initialized_q_learning/initialized_q_agent.py
initialized_q_learning/transition_probabilities.json
```

This agent initializes Q-values using the Markov transition probabilities. It evaluates whether the same learned transition structure can reduce Q-learning training cost and state/action exploration overhead.

## Environment setup

There is currently no dependency lockfile in this repository. The expected setup follows the NetSecGame and NetSecGameAgents projects.

A typical workspace layout is:

```text
workspace/
├── NetSecGame/             # NetSecGame / AIDojoCoordinator environment
├── NetSecGameAgents/       # clone of https://github.com/diegoforni/NetSecGameAgents
└── markov-prior-agents/
```

Recommended setup:

```bash
python -m venv aidojo-agents
source aidojo-agents/bin/activate

# Install the NetSecGame coordinator package.
cd /path/to/NetSecGame
python -m pip install -e .

# Install Diego Forni's NetSecGameAgents package and the extras used here.
git clone https://github.com/diegoforni/NetSecGameAgents.git /path/to/NetSecGameAgents
cd /path/to/NetSecGameAgents
python -m pip install -e ".[llm,q_learning]"

# Return to this repository before running experiments.
cd /path/to/markov-prior-agents
```

The referenced `NetSecGameAgents` and current `NetSecGame` packaging require Python 3.12 in their `pyproject.toml` files. Dependency resolution may fail with other Python versions.

Additional inferred dependencies used directly by this repository include:

- `numpy`
- `pandas`
- `mlflow`
- `PyYAML`
- `jinja2`
- `python-dotenv` / `dotenv`
- `openai`
- `tenacity`
- `matplotlib`
- `transformers` may be imported by some scripts

## Starting a NetSecGame server

Start a compatible NetSecGame coordinator before running any agent. The NetSecGame documentation uses this pattern:

```bash
python3 -m AIDojoCoordinator.worlds.NSEGameCoordinator \
  --task_config ./examples/example_task_configuration.yaml \
  --game_port 9001
```

The agents in this repository must use the same host and port:

```text
--host 127.0.0.1 --port 9001
```

Important NetSecGame task configuration fields include:

- `env.scenario`: network topology, e.g. `one_network`, `two_networks_tiny`, `two_networks_small`, `two_networks`, or `three_net_scenario`;
- `env.use_dynamic_addresses`: whether IP addresses change between episodes;
- `env.use_firewall`: whether firewall rules are active;
- `env.required_players`: number of agents needed before the game starts;
- `coordinator.agents.Attacker.max_steps`: maximum steps per episode;
- `coordinator.agents.Attacker.goal`: target state for success;
- `coordinator.agents.Attacker.start_position`: initial attacker knowledge and controlled hosts.

For reproducible experiments, record the exact task YAML, scenario, firewall setting, dynamic-address setting, maximum steps, reward configuration, and random seed.

## Running experiments

### Markov-guided LLM/ReAct agent

```bash
python mc_llm_qa/mc_llm_agent_qa.py \
  --host 127.0.0.1 \
  --port 9001 \
  --llm gpt-3.5-turbo \
  --api_url http://127.0.0.1:11434/v1/ \
  --markov_chain mc_llm_qa/transition_probabilities.json \
  --disable_mlflow
```

For GPT models, provide an OpenAI API key through a `.env` file or environment variable:

```text
OPENAI_API_KEY=...
```

For local/OpenAI-compatible models, configure `--api_url` accordingly.

### Baseline LLM/ReAct agent

```bash
python mc_llm_qa_baseline/llm_agent_qa.py \
  --host 127.0.0.1 \
  --port 9001 \
  --llm gpt-3.5-turbo \
  --api_url http://127.0.0.1:11434/v1/ \
  --disable_mlflow
```

### Initialized Q-learning agent

```bash
python initialized_q_learning/initialized_q_agent.py \
  --host 127.0.0.1 \
  --port 9001 \
  --transition_path initialized_q_learning/transition_probabilities.json
```

Some defaults are relative to the current working directory, so explicit paths are recommended.

### Genetic Algorithm agent

```bash
python GA/genetic/genetic_agent.py --host 127.0.0.1 --port 9000
```

The GA script expects the external NetSecGame/AIDojo workspace layout to be available. Note that this example uses port `9000`, while the LLM and Q-learning examples above use `9001`; match the port to the coordinator you started.

## Outputs

Depending on the script and working directory, runtime outputs may include:

- `episode_data.json`
- `llm_react.log`
- Q-learning logs under `initialized_q_learning/logs/`
- MLflow runs, unless disabled with `--disable_mlflow`
- generated plots from `jsonGraph.py`

Some scripts write outputs to the current working directory rather than a dedicated output folder.

## NetSecGame action assumptions

Several NetSecGame constraints are important when interpreting redundant or failed actions:

- `ExploitService` expects that the target service has already been discovered with `FindServices`.
- `FindData` requires ownership/control of the target host.
- `ExfiltrateData` requires controlling both the source and target hosts.
- `FindServices` can discover hosts if they expose active services.
- `ScanNetwork` and `FindServices` parameters can be chosen even if the network or host is not already listed in the current known state.

These assumptions explain why a useful high-level action type from the Markov prior still needs to be grounded into valid parameters by the LLM or by the corresponding agent logic.

## Reproducibility notes

For each experiment, record:

- NetSecGame commit/version;
- Diego Forni's NetSecGameAgents commit/version;
- task YAML and scenario;
- server host/port;
- model name and API endpoint;
- random seeds, if any;
- transition matrix file;
- whether MLflow was enabled;
- output files produced by the run.

## Caveats

- This repository does not currently include a dependency lockfile or Docker environment.
- The code assumes a local workspace containing NetSecGame and NetSecGameAgents.
- Some code is adapted from Diego Forni's `NetSecGameAgents` repository rather than packaged as installable modules here.
- Default ports differ across components.
- The baseline folder includes some files that may not be used by the baseline runner, such as a transition-probability JSON.
