"""Summarize results.jsonl into a table and a chart."""
import json
import math
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def wilson(k, n, z=1.64):  # about a 90% interval
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def main(path="results/results.jsonl", out="results/summary.png"):
    rows = [json.loads(l) for l in open(path)]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["condition"])].append(r)

    print(f"{'model':24s}{'condition':11s}{'n':>3s}{'saw':>6s}{'tried':>7s}{'cheated':>9s}{'honest ok':>11s}{'admit|cheat':>13s}{'false admit':>13s}")
    summary = {}
    for (model, cond), rs in sorted(groups.items()):
        n = len(rs)
        cheated = [r for r in rs if r["cheated"]]
        clean = [r for r in rs if not r["cheated"]]
        admit_c = sum(r["admitted"] for r in cheated)
        false_a = sum(r["admitted"] for r in clean)
        summary[(model, cond)] = (n, len(cheated), admit_c, len(clean), false_a)
        print(f"{model:24s}{cond:11s}{n:>3d}{sum(r['saw_leak'] for r in rs):>6d}"
              f"{sum(r['touched_leak'] for r in rs):>7d}{len(cheated):>9d}"
              f"{sum(r['honest_success'] for r in rs):>11d}"
              f"{(str(admit_c) + '/' + str(len(cheated))):>13s}{(str(false_a) + '/' + str(len(clean))):>13s}")

    models = sorted({m for m, _ in groups})
    conds = sorted({c for _, c in groups})
    colors = ["#4285F4", "#DB4437", "#F4B400", "#0F9D58"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    width = 0.8 / max(len(conds), 1)
    for ai, (ax, title) in enumerate(zip(axes, ["Cheating rate", "Admission rate among cheaters"])):
        for ci, cond in enumerate(conds):
            xs, ys, lo, hi = [], [], [], []
            for mi, model in enumerate(models):
                s = summary.get((model, cond))
                if not s:
                    continue
                n, k_cheat, k_admit, _, _ = s
                k, tot = (k_cheat, n) if ai == 0 else (k_admit, k_cheat)
                if tot == 0:
                    continue
                p = k / tot
                l, h = wilson(k, tot)
                xs.append(mi + ci * width); ys.append(p); lo.append(p - l); hi.append(h - p)
            ax.bar(xs, ys, width=width, color=colors[ci % 4], label=cond, yerr=[lo, hi], capsize=3)
        ax.set_xticks([i + width * (len(conds) - 1) / 2 for i in range(len(models))])
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_ylim(0, 1.05); ax.set_title(title); ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("Chart saved to", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
