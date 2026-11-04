"""Read raw_results.json, check normality, run Friedman/Page + pairwise Wilcoxon.
No GPU, no inference.

Usage: python run_stats.py --config <base.json> --dataset SEDF78 --ablation width
"""
import argparse
import itertools
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import friedmanchisquare, page_trend_test, wilcoxon
from statsmodels.stats.multitest import multipletests

from common import (ABLATION_VALUES, DATASETS, EPOCH_DURATIONS, get_fold_metric,
                    load_config, load_results, results_path, load_results_aggr, get_fold_metric_aggr)


def collect(args):
    """Long DataFrame: one row per (epoch_s, ablation_level, fold)."""
    dataset, n_folds = DATASETS[args.dataset]
    rows = []
    for epoch in EPOCH_DURATIONS:
        for value in ABLATION_VALUES[args.ablation]:
            config = load_config(args.config, args.ablation, value, dataset, epoch)
            if config is None:
                continue
            # data = load_results(results_path(config))
            data_all = load_results_aggr(config["name"]+"_new_data")
            for fold in range(1, n_folds + 1):
                # mf1 = get_fold_metric(data, dataset, args.ablation, value, epoch, fold)
                mf1 = get_fold_metric_aggr(data_all, fold)
                if mf1 is None:
                    print(f"[missing] {args.ablation}={value} E={epoch}s fold {fold}")
                    continue
                rows.append({"dataset": dataset, "epoch_s": epoch,
                             "ablation_level": str(value), "fold": fold, "mf1": mf1})
    return pd.DataFrame(rows)


def marginal_normality(df):
    """Shapiro-Wilk on raw MF1 per (dataset, epoch_s, ablation_level)."""
    rows = []
    for (dataset, epoch, level), g in df.groupby(["dataset", "epoch_s", "ablation_level"]):
        w, p = (np.nan, np.nan) if len(g) < 3 else stats.shapiro(g["mf1"])
        rows.append({"dataset": dataset, "epoch_s": epoch, "ablation_level": level,
                     "n": len(g), "W": w, "p": p})
    return pd.DataFrame(rows)


def pairwise_difference_normality(df):
    """Shapiro-Wilk on within-fold differences between level pairs
    (the assumption RM-ANOVA actually needs)."""
    rows = []
    for (dataset, epoch), g in df.groupby(["dataset", "epoch_s"]):
        wide = g.pivot(index="fold", columns="ablation_level", values="mf1")
        for a, b in itertools.combinations(wide.columns, 2):
            diff = (wide[a] - wide[b]).dropna()
            w, p = (np.nan, np.nan) if len(diff) < 3 else stats.shapiro(diff)
            rows.append({"dataset": dataset, "epoch_s": epoch, "pair": f"{a} vs {b}",
                         "n": len(diff), "W": w, "p": p})
    return pd.DataFrame(rows)


def plot_cell(df, dataset, epoch, out_path):
    """Histogram + Q-Q plot per ablation level for one (dataset, epoch_s) cell."""
    cell = df[(df.dataset == dataset) & (df.epoch_s == epoch)]
    levels = sorted(cell["ablation_level"].unique())
    fig, axes = plt.subplots(2, len(levels), figsize=(4 * len(levels), 6), squeeze=False)
    for i, level in enumerate(levels):
        mf1 = cell[cell.ablation_level == level]["mf1"].values
        axes[0, i].hist(mf1, bins=min(10, len(mf1)), edgecolor="black")
        axes[0, i].set_title(f"{level} (n={len(mf1)})")
        axes[0, i].set_xlabel("MF1")
        stats.probplot(mf1, dist="norm", plot=axes[1, i])
        axes[1, i].set_title("Q-Q plot")
    fig.suptitle(f"{dataset}, {epoch}s epochs")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def use_page_test(ablation, dataset, epoch):
    """Page trend test (ordered alternative) only where a monotone width trend is expected."""
    return ablation == "width" and (dataset == "Sleep-EDF-2018" or (dataset == "Physio-2018" and epoch == 10))


def run_tests(df, ablation):
    levels = [str(v) for v in ABLATION_VALUES[ablation]]
    results = []
    for (dataset, epoch), cell in df.groupby(["dataset", "epoch_s"]):
        wide = cell.pivot(index="fold", columns="ablation_level", values="mf1")
        wide = wide.reindex(columns=levels).dropna()  # complete folds only, in level order
        if len(wide) < 3:
            print(f"[skip tests] {dataset} E={epoch}s: only {len(wide)} complete folds")
            continue

        if use_page_test(ablation, dataset, epoch):
            r = page_trend_test(wide.values)  # columns = predicted increasing order
            results.append({"epoch_s": int(epoch), "test": "page", "levels": levels,
                            "stat": r.statistic, "p": r.pvalue})
        else:
            stat, p = friedmanchisquare(*[wide[l] for l in levels])
            results.append({"epoch_s": int(epoch), "test": "friedman", "levels": levels,
                            "stat": stat, "p": p})

        for a, b in itertools.combinations(levels, 2):
            stat, p = wilcoxon(wide[a], wide[b])
            results.append({"epoch_s": int(epoch), "test": "wilcoxon", "levels": [a, b],
                            "stat": stat, "p": p})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="base config path")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--ablation", required=True, choices=ABLATION_VALUES)
    args = parser.parse_args()

    df = collect(args)
    if df.empty:
        print("No results found. Run run_eval.py first.")
        return

    tag = f"{args.dataset}_{args.ablation}"
    marginal_normality(df).to_csv(f"shapiro_marginal_{tag}.csv", index=False)
    pairwise_difference_normality(df).to_csv(f"shapiro_pairwise_diff_{tag}.csv", index=False)
    for (dataset, epoch), _ in df.groupby(["dataset", "epoch_s"]):
        plot_cell(df, dataset, epoch, f"dist_{tag}_{epoch}s.png")

    results = run_tests(df, args.ablation)
    for r in results:
        print(f"E={r['epoch_s']}s {r['test']:9s} {' vs '.join(r['levels']):20s} p={r['p']:.4f}")
    with open(f"results/sleepyco/statistical_tests/{tag}.json", "w") as f:
        json.dump(results, f, indent=2, default=float)


if __name__ == "__main__":
    main()