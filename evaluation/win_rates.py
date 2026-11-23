# Copyright (c) 2026 Jakir Hossain, Ahmet Erdem Sariyuce.
# Licensed under the MIT License

"""Win / loss / tie rates and Wilcoxon p-values from judge.py outputs.

Usage
    python evaluation/win_rates.py <results_dir_or_json> [...] [--metrics ...]

Files named <pair>_<k>.json (repeated evaluations written by judge.py) are
grouped by <pair>. For each pair and metric:
    Win / Loss / Tie   % over all evaluations of all runs (win = answers1).
    p-value            Wilcoxon signed-rank test (default two-sided; --alternative
                       greater = answers1 better) on per-question scores
                       (win=100, tie=50, loss=0), averaged over runs, against the
                       complementary score.
    p_holm             p-value after Holm-Bonferroni correction. Default
                       (--holm folder-metric): within each input folder (dataset)
                       and metric. --holm folder: within each
                       folder over both metrics. --holm all: over all comparisons.
"""

import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import wilcoxon

METRICS = ["comprehensiveness", "diversity"]
SCORE = {1: 100.0, 0: 50.0, 2: 0.0}


def load_groups(paths):
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "*.json"))) if os.path.isdir(p) else [p]
    groups = defaultdict(list)
    for f in files:
        m = re.match(r"(.+)_(\d+)\.json$", os.path.basename(f))
        pair = m.group(1) if m else os.path.basename(f)[:-5]
        groups[f"{os.path.basename(os.path.dirname(os.path.abspath(f)))}/{pair}"].append(f)
    return groups


def pair_stats(files, metric, alternative="greater"):
    counts, per_question = Counter(), defaultdict(list)
    for f in files:
        with open(f) as fh:
            for row in json.load(fh):
                result = row.get("metrics", {}).get(metric)
                if result is None:
                    continue
                winner = result.get("final_winner", 0)
                counts[winner] += 1
                per_question[row["id"]].append(SCORE[winner])

    n = sum(counts.values())
    if n == 0:
        return None
    scores = np.array([np.mean(v) for v in per_question.values()])
    diff = np.round(scores - (100.0 - scores), 9)  # round so equal scores tie exactly
    p = wilcoxon(diff, alternative=alternative).pvalue if np.any(diff != 0) else 1.0
    return {
        "win": 100 * counts[1] / n, "loss": 100 * counts[2] / n, "tie": 100 * counts[0] / n,
        "n_eval": n, "n_questions": len(per_question), "p_value": p,
    }


def holm(p_values):
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1)."""
    m = len(p_values)
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda i: p_values[i])):
        running = max(running, (m - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="judge.py result files or directories")
    ap.add_argument("--metrics", nargs="+", default=METRICS)
    ap.add_argument("--alternative", choices=["two-sided", "greater"], default="two-sided")
    ap.add_argument("--holm", choices=["folder-metric", "folder", "all"], default="folder-metric",
                    help="Holm-Bonferroni groups: per folder and metric (default), per folder, or all comparisons")
    ap.add_argument("--verbose", action="store_true", help="also print one line per pair and metric")
    args = ap.parse_args()

    rows = []
    for pair, files in load_groups(args.paths).items():
        for metric in args.metrics:
            s = pair_stats(files, metric, args.alternative)
            if s is not None:
                rows.append((pair, metric, len(files), s))

    group = {
        "folder-metric": lambda key, metric: (key.split("/", 1)[0], metric),
        "folder": lambda key, metric: key.split("/", 1)[0],
        "all": lambda key, metric: None,
    }[args.holm]
    adjusted = [0.0] * len(rows)
    for g in {group(key, metric) for key, metric, _, _ in rows}:
        idx = [i for i, (key, metric, _, _) in enumerate(rows) if group(key, metric) == g]
        for i, a in zip(idx, holm([rows[i][3]["p_value"] for i in idx])):
            adjusted[i] = a
    if args.verbose:
        for (pair, metric, runs, s), p_holm in zip(rows, adjusted):
            print(f"{pair:45} {metric:18} Win {s['win']:5.1f}  Loss {s['loss']:5.1f}  Tie {s['tie']:5.1f}"
                  f"  p={fmt_p(s['p_value']):>6}  p_holm={fmt_p(p_holm):>6}"
                  f"  (runs={runs}, questions={s['n_questions']}, evaluations={s['n_eval']})")
    print_tables(rows, adjusted, args.metrics)
    side = "two-sided" if args.alternative == "two-sided" else "one-sided (answers1 better)"
    scope = {"folder-metric": "within each folder and metric", "folder": "within each folder",
             "all": f"over all {len(rows)} comparisons"}[args.holm]
    print(f"\nWilcoxon {side}; Holm-Bonferroni {scope}.")


def print_tables(rows, adjusted, metrics):
    """Win/Loss/Tie and Holm-adjusted p-value tables: rows = <condition>, columns = <folder> <baseline>,
    from files <folder>/<condition>_<baseline>_<run>.json (e.g. podcast/M2hC_LF_C2_0.json)."""
    cell, conds, bases = {}, [], []
    for (key, metric, _, s), p_holm in zip(rows, adjusted):
        folder, pair = key.split("/", 1)
        cond, base = pair.rsplit("_", 1) if "_" in pair else (pair, "")
        base = f"{folder} {base}"
        cell[cond, base, metric] = (s, p_holm)
        conds += [cond] if cond not in conds else []
        bases += [base] if base not in bases else []
    rank = {"RkH": 0, "M2hC": 1, "MRC": 2}
    conds.sort(key=lambda c: (rank.get(c.split("_")[0], 9), c))
    for metric in metrics:
        print(f"\n{metric.capitalize()}: Win / Loss / Tie (%)")
        print(f"{'':12}" + "".join(f"{b:>22}" for b in bases))
        for c in conds:
            vals = [cell.get((c, b, metric)) for b in bases]
            print(f"{c.replace('_', ' '):12}" + "".join(
                f"{v[0]['win']:10.0f}{v[0]['loss']:6.0f}{v[0]['tie']:6.0f}" if v else f"{'-':>22}" for v in vals))
        print(f"\n{metric.capitalize()}: Wilcoxon p-value (Holm-Bonferroni)")
        print(f"{'':12}" + "".join(f"{b:>22}" for b in bases))
        for c in conds:
            vals = [cell.get((c, b, metric)) for b in bases]
            print(f"{c.replace('_', ' '):12}" + "".join(f"{fmt_p(v[1]) if v else '-':>22}" for v in vals))


if __name__ == "__main__":
    main()
