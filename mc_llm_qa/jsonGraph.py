#!/usr/bin/env python3
import os
import json
import argparse
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# --- CHANGE 1: INCREASE GLOBAL FONT SIZES ---
# Update default parameters for matplotlib to make all text elements larger.
plt.rcParams.update({
    'font.size': 16,          # Default text size
    'axes.titlesize': 20,     # Title of the plot
    'axes.labelsize': 16,     # X and Y axis labels
    'xtick.labelsize': 14,    # X-axis tick labels
    'ytick.labelsize': 14,    # Y-axis tick labels
    'legend.fontsize': 12     # Legend text
})

def analyze_file(path):
    """
    Returns:
      wins: int
      eval_counts: dict with keys 'bad','boring','good'
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    wins = 0
    all_evals = []

    for item in data:
        if item.get('end_reason') == "AgentStatus.Success":
            wins += 1
        all_evals.extend(item.get('evaluation', []))

    cnt = Counter()
    for e in all_evals:
        if e == 0:
            cnt['bad'] += 1
        elif e == 3:
            cnt['boring'] += 1
        elif e in (8, 10):
            cnt['good'] += 1
    return wins, cnt

def plot_wins(fnames, wins_list, output_path):
    plt.figure(figsize=(10, 6))
    x = np.arange(len(fnames))
    bars = plt.bar(x, wins_list)
    plt.xticks(x, fnames, rotation=45, ha='right')
    plt.ylabel('Number of Wins')
    plt.title('Wins per JSON')
    max_val = max(wins_list) if wins_list else 0
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + max_val*0.01,
                 f'{int(height)}', ha='center', va='bottom')
    plt.tight_layout(pad=3.0)
    plt.savefig(output_path)
    plt.close()

def plot_eval_percentages_grouped(fnames, eval_counters, output_path):
    perc = {'bad': [], 'boring': [], 'good': []}
    for c in eval_counters:
        total = sum(c.values())
        perc['bad'].append((c.get('bad', 0) / total * 100) if total else 0)
        perc['boring'].append((c.get('boring', 0) / total * 100) if total else 0)
        perc['good'].append((c.get('good', 0) / total * 100) if total else 0)

    x = np.arange(len(fnames))
    width = 0.2

    plt.figure(figsize=(10, 6))
    bars_bad = plt.bar(x - width, perc['bad'], width, label='Bad')
    bars_boring = plt.bar(x, perc['boring'], width, label='Boring')
    bars_good = plt.bar(x + width, perc['good'], width, label='Good')

    plt.xticks(x, fnames, rotation=45, ha='right')
    plt.ylabel('Percentage of Evaluations')
    plt.ylim(0, max(max(perc['boring']), max(perc['good']), max(perc['bad'])) * 1.2)
    plt.title('Action outcome distribution Scenario 1 Full')
    plt.legend(loc='upper right')
    for bars in (bars_bad, bars_boring, bars_good):
        for bar in bars:
            height = bar.get_height()
            # --- CHANGE 2: INCREASE BAR LABEL FONT SIZE ---
            # Increased font size for the percentage label on each bar.
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=14)
    plt.tight_layout(pad=3.0)
    plt.savefig(output_path)
    plt.close()

def plot_good_focus(fnames, eval_counters, output_path):
    perc_good = []
    for c in eval_counters:
        total = sum(c.values())
        perc_good.append((c.get('good', 0) / total * 100) if total else 0)

    plt.figure(figsize=(10, 4))
    x = np.arange(len(fnames))
    bars = plt.bar(x, perc_good)
    plt.xticks(x, fnames, rotation=45, ha='right')
    plt.ylabel('Good (%)')
    plt.title('Good Evaluation Percentage (Zoomed)')
    max_good = max(perc_good) if perc_good else 1
    plt.ylim(0, max_good * 1.4)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + max_good*0.02,
                 f'{height:.1f}%', ha='center', va='bottom')
    plt.tight_layout(pad=3.0)
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze JSON evaluations")
    parser.add_argument('json_dir', help="Directory or single JSON file")
    parser.add_argument('--out-dir', default='.', help="Output directory for PNGs")
    parser.add_argument('--order', action='store_true', help="Prompt to order files left to right")
    args = parser.parse_args()

    if os.path.isdir(args.json_dir):
        files = sorted([os.path.join(args.json_dir, f)
                        for f in os.listdir(args.json_dir) if f.lower().endswith('.json')])
    else:
        files = [args.json_dir]

    fnames, wins_list, eval_counters = [], [], []
    for p in files:
        w, ec = analyze_file(p)
        fnames.append(os.path.splitext(os.path.basename(p))[0])
        wins_list.append(w)
        eval_counters.append(ec)

    # If ordering flag is passed, prompt user for positions
    if args.order and len(fnames) > 1:
        print(f"Specify display position for each file (1 = leftmost, {len(fnames)} = rightmost):")
        positions = {}
        for name in fnames:
            while True:
                try:
                    pos = int(input(f"Position for {name}: "))
                    if 1 <= pos <= len(fnames) and pos not in positions.values():
                        positions[name] = pos
                        break
                    else:
                        print(f"Invalid or duplicate position. Enter a unique number between 1 and {len(fnames)}.")
                except ValueError:
                    print("Please enter an integer.")
        # Reorder lists based on provided positions
        sorted_names = sorted(fnames, key=lambda n: positions[n])
        idx_map = {name: i for i, name in enumerate(fnames)}
        fnames = sorted_names
        wins_list = [wins_list[idx_map[name]] for name in sorted_names]
        eval_counters = [eval_counters[idx_map[name]] for name in sorted_names]

    os.makedirs(args.out_dir, exist_ok=True)
    plot_wins(fnames, wins_list, os.path.join(args.out_dir, 'wins.png'))
    plot_eval_percentages_grouped(fnames, eval_counters, os.path.join(args.out_dir, 'eval_grouped.png'))
    plot_good_focus(fnames, eval_counters, os.path.join(args.out_dir, 'good_focus.png'))

    print("Charts saved to", args.out_dir)

if __name__ == '__main__':
    main()