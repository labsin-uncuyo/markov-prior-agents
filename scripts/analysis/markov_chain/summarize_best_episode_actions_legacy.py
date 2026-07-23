#!/usr/bin/env python3
"""Summarize action-level percentages for ALL and BEPR markov-chain results.

Interpretation follows analyze_best_episodes.py:
- Good actions: rewards 1 or 9
- Boring actions: every other reward
- ALL: all actions across all episodes
- BEPR: actions inside the shortest episode per run whose last reward is 9;
  if there is a tie on length, keep the first such episode.
"""

import csv
import re
from pathlib import Path


DATA_DIR = Path("data/results_markov_chain")
OUTPUT_CSV = DATA_DIR / "best_episode_action_summary_markov_chain.csv"
FILE_CONFIG = [
    ("parsed_populationGA.json", "Genetic Agent"),
    ("parsed_populationRA10.json", "Random Agent"),
    ("parsed_populationGPT.json", "GPT4o Agent"),
    ("parsed_populationED.json", "Expert Defined (GPT o3-mini)"),
]
CATEGORIES = ("Good", "Boring")
SCORE_PATTERN = re.compile(r', (-?\d+)\]",?$')


def empty_counts():
    return {category: 0 for category in CATEGORIES}


def categorize_action_score(score):
    if score in (1, 9):
        return "Good"
    return "Boring"


def parse_markov_file(filepath):
    all_counts = empty_counts()
    bepr_counts = empty_counts()

    current_episode_counts = empty_counts()
    current_episode_length = 0
    current_episode_last_score = None
    in_episode = False

    best_run_counts = empty_counts()
    best_run_length = None
    run_has_best = False

    total_runs = 0
    eligible_runs = 0
    total_episodes = 0

    with filepath.open() as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")

            if line == "    [":
                total_runs += 1
                best_run_counts = empty_counts()
                best_run_length = None
                run_has_best = False
                continue

            if line == "        [":
                current_episode_counts = empty_counts()
                current_episode_length = 0
                current_episode_last_score = None
                in_episode = True
                continue

            if in_episode and line in ("        ],", "        ]"):
                total_episodes += 1
                if current_episode_last_score == 9:
                    if (
                        best_run_length is None
                        or current_episode_length < best_run_length
                    ):
                        best_run_length = current_episode_length
                        best_run_counts = current_episode_counts.copy()
                    run_has_best = True
                in_episode = False
                continue

            if line in ("    ],", "    ]"):
                if run_has_best:
                    eligible_runs += 1
                    for category in CATEGORIES:
                        bepr_counts[category] += best_run_counts[category]
                continue

            match = SCORE_PATTERN.search(line)
            if not match:
                continue

            score = int(match.group(1))
            category = categorize_action_score(score)

            all_counts[category] += 1
            current_episode_counts[category] += 1
            current_episode_length += 1
            current_episode_last_score = score

    return {
        "all_counts": all_counts,
        "bepr_counts": bepr_counts,
        "total_runs": total_runs,
        "eligible_runs": eligible_runs,
        "total_episodes": total_episodes,
    }


def build_rows():
    rows = []
    summaries = []

    for filename, display_name in FILE_CONFIG:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Missing input file: {filepath}")

        parsed = parse_markov_file(filepath)
        summaries.append((display_name, parsed))

        for panel, counts in (
            ("ALL", parsed["all_counts"]),
            ("BEPR", parsed["bepr_counts"]),
        ):
            total = sum(counts.values())
            if panel == "ALL" and total == 0:
                raise ValueError(f"No actions found for {display_name}")

            for category in CATEGORIES:
                count = counts[category]
                rows.append(
                    {
                        "panel": panel,
                        "display_file": display_name,
                        "category": category,
                        "count": count,
                        "pct": (count / total * 100) if total else 0,
                    }
                )

    return rows, summaries


def main():
    rows, summaries = build_rows()

    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["panel", "display_file", "category", "count", "pct"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summary written to: {OUTPUT_CSV}")
    print("\nCoverage:")
    for display_name, parsed in summaries:
        excluded_runs = parsed["total_runs"] - parsed["eligible_runs"]
        print(
            f"- {display_name}: "
            f"{parsed['total_episodes']} episodes, "
            f"{parsed['total_runs']} runs, "
            f"{parsed['eligible_runs']} BEPR-eligible runs, "
            f"{excluded_runs} excluded runs"
        )


if __name__ == "__main__":
    main()
