#!/usr/bin/env python3
"""Print a markdown table comparing eval JSON results from multiple runs.

Usage:
    python scripts/summarize_results.py \
        Finetuned=/root/autodl-tmp/work/eval_baseline2/results.json \
        CSA-v2=/root/autodl-tmp/work/eval_csa_v2_2/results.json
"""

import argparse
import json
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # Each arg: NAME=path
    runs = []
    for arg in sys.argv[1:]:
        if "=" not in arg:
            print(f"Bad arg: {arg}")
            sys.exit(1)
        name, path = arg.split("=", 1)
        runs.append((name, load(path)))

    group_sizes = sorted({
        int(k.split("_")[0][1:])
        for _, run in runs
        for k in run["results"]
    })

    # Each "run" has BOTH Independent and CSA-v2 sub-conditions, but we report
    # the Independent column for the Finetuned baseline run (no CSA gain) and
    # the CSA-v2 column for the CSA-v2 run (CSA gain).
    print("\n## EM (%)\n")
    header = "| Strategy | " + " | ".join(f"G={g}" for g in group_sizes) + " |"
    sep = "|" + "---|" * (len(group_sizes) + 1)
    print(header)
    print(sep)

    rows = []
    for name, run in runs:
        for cond in ["Independent", "CSA-v2"]:
            label = name if cond == "CSA-v2" else f"{name} (no CSA)"
            cells = []
            for g in group_sizes:
                key = f"G{g}_{cond}"
                v = run["results"].get(key, {})
                if v:
                    cells.append(f"{v['em']:.2f}")
                else:
                    cells.append("-")
            rows.append((label, cells))

    for label, cells in rows:
        print(f"| {label} | " + " | ".join(cells) + " |")

    print("\n## F1 (%)\n")
    print(header)
    print(sep)
    for name, run in runs:
        for cond in ["Independent", "CSA-v2"]:
            label = name if cond == "CSA-v2" else f"{name} (no CSA)"
            cells = []
            for g in group_sizes:
                key = f"G{g}_{cond}"
                v = run["results"].get(key, {})
                cells.append(f"{v['f1']:.2f}" if v else "-")
            print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
