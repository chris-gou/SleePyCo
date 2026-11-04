"""Run inference per fold and write raw_results.json. Skips folds already saved.

Usage: python run_eval.py --config <base.json> --dataset SEDF78 --ablation width --gpu 0
"""
import argparse
import json
import os

import sklearn.metrics as skmet

from common import (ABLATION_VALUES, DATASETS, EPOCH_DURATIONS, get_fold_metric,
                    load_config, load_results, results_path)
from test import OneFoldEvaluator

LABELS = [0, 1, 2, 3, 4]  # W, N1, N2, N3, REM


def evaluate_fold(args, config, fold):
    y_true, y_prob, *_ = OneFoldEvaluator(args, fold, config).run()
    y_true = y_true.astype(int)
    y_pred = y_prob.argmax(axis=1)
    f1 = skmet.f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {
        "acc": skmet.accuracy_score(y_true, y_pred),
        "mf1": skmet.f1_score(y_true, y_pred, average="macro"),
        "kappa": skmet.cohen_kappa_score(y_true, y_pred),
        "wf1": f1[0], "n1f1": f1[1], "n2f1": f1[2], "n3f1": f1[3], "rf1": f1[4],
        "cm": skmet.confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def save_fold(path, dataset, ablation, value, epoch, fold, metrics):
    data = load_results(path)
    (data.setdefault(dataset, {}).setdefault(ablation, {})
         .setdefault(str(value), {}).setdefault(str(epoch), {}))[str(fold)] = metrics
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="base config path")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--ablation", required=True, choices=ABLATION_VALUES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="use CPU for streaming latency")
    args = parser.parse_args()

    dataset, n_folds = DATASETS[args.dataset]
    for epoch in EPOCH_DURATIONS:
        for value in ABLATION_VALUES[args.ablation]:
            config = load_config(args.config, args.ablation, value, dataset, epoch)
            if config is None:
                continue
            path = results_path(config)
            for fold in range(1, n_folds + 1):
                done = get_fold_metric(load_results(path), dataset, args.ablation, value, epoch, fold)
                if done is not None:
                    print(f"[skip] {args.ablation}={value} E={epoch}s fold {fold}")
                    continue
                print(f"[run]  {args.ablation}={value} E={epoch}s fold {fold}")
                metrics = evaluate_fold(args, config, fold)
                save_fold(path, dataset, args.ablation, value, epoch, fold, metrics)


if __name__ == "__main__":
    main()