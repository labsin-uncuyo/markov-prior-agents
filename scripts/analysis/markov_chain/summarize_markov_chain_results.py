#!/usr/bin/env python3
"""Summarize Markov-chain experiment artifacts by scenario.

The raw files are stored as compressed JSON-like arrays of runs, episodes, and
stringified NetSecGame actions with their reward/evaluation score.  This script
uses a streaming parser so the large ``*.json.gz`` files do not need to be fully
loaded in memory.

Outputs are written under ``data/processed/markov_chain/<scenario>/``:

- ``table_metrics.csv``: run/episode-level metrics useful for manuscript tables.
- ``terminal_outcome_summary.csv``: terminal episode outcomes for ALL and BEPR.
- ``action_outcome_summary.csv``: action outcome distribution for ALL and BEPR.

Definitions:

- ALL: all episodes/actions across all runs.
- BEPR: shortest successful episode (terminal reward 9) per run; runs without a
  successful episode are excluded from BEPR action counts, while the BEPR win
  percentage in ``table_metrics.csv`` is computed over all runs.
- Effective/Good action: reward 1 or 9.
- Redundant/Boring action: reward 0.
- Invalid/Detected/Bad action: any other reward, typically negative rewards.
"""

from __future__ import annotations

import csv
import gzip
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "data" / "raw" / "markov_chain"
PROCESSED_ROOT = ROOT / "data" / "processed" / "markov_chain"

SCENARIOS = {
    "full": {
        "Genetic Agent": "parsed_populationGA.json.gz",
        "Random Agent": "parsed_populationRA10.json.gz",
        "GPT-4o Agent": "parsed_populationGPT.json.gz",
        "Expert Defined (GPT o3-mini)": "parsed_populationED.json.gz",
    },
    "three_network": {
        "Genetic Agent": "parsed_populationGA3.json.gz",
        "Random Agent": "parsed_populationRA103.json.gz",
        "GPT-4o Agent": "parsed_populationGPT3.json.gz",
        "Expert Defined (GPT o3-mini)": "parsed_populationED3.json.gz",
    },
    "defender": {
        "Genetic Agent": "parsed_populationGAD.json.gz",
        "Random Agent": "parsed_populationRA10D.json.gz",
        "GPT-4o Agent": "parsed_populationGPTD.json.gz",
        "Expert Defined (GPT o3-mini)": "parsed_populationEDD.json.gz",
    },
}

SCORE_PATTERN = re.compile(r", (-?\d+)\]", re.ASCII)
ACTION_CATEGORIES = ("Effective", "Redundant", "Bad")
TERMINAL_CATEGORIES = ("Success", "Timeout/No goal", "Detected")


@dataclass
class Episode:
    length: int = 0
    last_score: int | None = None
    action_counts: dict[str, int] = field(
        default_factory=lambda: {category: 0 for category in ACTION_CATEGORIES}
    )

    @property
    def successful(self) -> bool:
        return self.last_score == 9


@dataclass
class Run:
    episodes: list[Episode] = field(default_factory=list)

    def best_success(self) -> Episode | None:
        successful = [episode for episode in self.episodes if episode.successful]
        if not successful:
            return None
        return min(successful, key=lambda episode: episode.length)


def action_category(score: int) -> str:
    if score in (1, 9):
        return "Effective"
    if score == 0:
        return "Redundant"
    return "Bad"


def terminal_category(score: int | None) -> str:
    if score == 9:
        return "Success"
    if score == -9:
        return "Detected"
    return "Timeout/No goal"


def indent_level(line: str) -> int:
    return (len(line) - len(line.lstrip(" "))) // 4


def parse_runs(filepath: Path) -> list[Run]:
    runs: list[Run] = []
    current_run: Run | None = None
    current_episode: Episode | None = None

    with gzip.open(filepath, "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            depth = indent_level(line)

            if depth == 1 and stripped == "[":
                current_run = Run()
                runs.append(current_run)
                continue

            if depth == 2 and stripped == "[":
                current_episode = Episode()
                continue

            if depth == 2 and stripped in ("]", "],") and current_episode is not None:
                if current_run is None:
                    raise ValueError(f"Episode outside run in {filepath}")
                current_run.episodes.append(current_episode)
                current_episode = None
                continue

            match = SCORE_PATTERN.search(line)
            if not match or current_episode is None:
                continue

            score = int(match.group(1))
            current_episode.length += 1
            current_episode.last_score = score
            current_episode.action_counts[action_category(score)] += 1

    return runs


def pct(count: int | float, total: int | float) -> float:
    return (count / total * 100.0) if total else 0.0


def sd(values: list[float]) -> float:
    """Sample standard deviation, matching the manuscript tables."""
    return stdev(values) if len(values) > 1 else 0.0


def summarize_agent(scenario: str, agent: str, filepath: Path):
    runs = parse_runs(filepath)
    episodes = [episode for run in runs for episode in run.episodes]
    best_episodes = [best for run in runs if (best := run.best_success()) is not None]

    all_action_counts = {category: 0 for category in ACTION_CATEGORIES}
    bepr_action_counts = {category: 0 for category in ACTION_CATEGORIES}
    all_terminal_counts = {category: 0 for category in TERMINAL_CATEGORIES}
    bepr_terminal_counts = {category: 0 for category in TERMINAL_CATEGORIES}

    for episode in episodes:
        for category, count in episode.action_counts.items():
            all_action_counts[category] += count
        all_terminal_counts[terminal_category(episode.last_score)] += 1

    for run in runs:
        best = run.best_success()
        if best is None:
            bepr_terminal_counts["Timeout/No goal"] += 1
            continue
        bepr_terminal_counts["Success"] += 1
        for category, count in best.action_counts.items():
            bepr_action_counts[category] += count

    all_lengths = [episode.length for episode in episodes]
    best_lengths = [episode.length for episode in best_episodes]
    # Manuscript-table convention for BEPR steps: use the shortest successful
    # episode when the run has one; otherwise keep the maximum observed episode
    # length for that run (usually the 100-step timeout). This preserves the
    # distinction between BEPR success rate and BEPR step count for agents that
    # do not solve every run.
    bepr_lengths_all_runs = []
    for run in runs:
        best = run.best_success()
        if best is not None:
            bepr_lengths_all_runs.append(best.length)
        elif run.episodes:
            bepr_lengths_all_runs.append(max(episode.length for episode in run.episodes))
    run_win_pcts = [pct(sum(ep.successful for ep in run.episodes), len(run.episodes)) for run in runs]

    metrics = {
        "scenario": scenario,
        "agent": agent,
        "runs": len(runs),
        "episodes": len(episodes),
        "all_win_pct": pct(sum(ep.successful for ep in episodes), len(episodes)),
        "all_win_pct_run_mean": mean(run_win_pcts) if run_win_pcts else 0.0,
        "all_win_pct_run_sd": sd(run_win_pcts),
        "all_steps_mean": mean(all_lengths) if all_lengths else math.nan,
        "all_steps_sd": sd(all_lengths),
        "bepr_successful_runs": len(best_episodes),
        "bepr_win_pct_over_runs": pct(len(best_episodes), len(runs)),
        "bepr_steps_mean": mean(bepr_lengths_all_runs) if bepr_lengths_all_runs else math.nan,
        "bepr_steps_sd": sd(bepr_lengths_all_runs),
        "bepr_steps_mean_successful_runs": mean(best_lengths) if best_lengths else math.nan,
        "bepr_steps_sd_successful_runs": sd(best_lengths),
        "min_success_steps": min(best_lengths) if best_lengths else math.nan,
    }

    return metrics, all_action_counts, bepr_action_counts, all_terminal_counts, bepr_terminal_counts


def write_rows(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    for scenario, agents in SCENARIOS.items():
        metrics_rows = []
        action_rows = []
        terminal_rows = []

        for agent, filename in agents.items():
            filepath = RAW_ROOT / scenario / filename
            if not filepath.exists():
                raise FileNotFoundError(filepath)

            metrics, all_actions, bepr_actions, all_terminal, bepr_terminal = summarize_agent(
                scenario, agent, filepath
            )
            metrics_rows.append(metrics)

            for panel, counts in (("ALL", all_actions), ("BEPR", bepr_actions)):
                total = sum(counts.values())
                for category in ACTION_CATEGORIES:
                    action_rows.append(
                        {
                            "scenario": scenario,
                            "panel": panel,
                            "agent": agent,
                            "category": category,
                            "count": counts[category],
                            "pct": pct(counts[category], total),
                        }
                    )

            for panel, counts in (("ALL", all_terminal), ("BEPR", bepr_terminal)):
                total = sum(counts.values())
                for category in TERMINAL_CATEGORIES:
                    terminal_rows.append(
                        {
                            "scenario": scenario,
                            "panel": panel,
                            "agent": agent,
                            "category": category,
                            "count": counts[category],
                            "pct": pct(counts[category], total),
                        }
                    )

        out_dir = PROCESSED_ROOT / scenario
        write_rows(out_dir / "table_metrics.csv", list(metrics_rows[0].keys()), metrics_rows)
        write_rows(
            out_dir / "action_outcome_summary.csv",
            ["scenario", "panel", "agent", "category", "count", "pct"],
            action_rows,
        )
        write_rows(
            out_dir / "terminal_outcome_summary.csv",
            ["scenario", "panel", "agent", "category", "count", "pct"],
            terminal_rows,
        )
        print(f"Wrote summaries for {scenario}: {out_dir}")


if __name__ == "__main__":
    main()
