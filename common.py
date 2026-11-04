"""Shared helpers for SLP"""
import json
import os

# CLI name -> (dataset name used in configs/results, number of folds)
DATASETS = {
    "SEDF78": ("Sleep-EDF-2018", 10),
    "PHYSIO": ("Physio2018", 5),
    "SHHS": ("SHHS", 1),
}
ABLATION_VALUES = {"scales": [1, 2, 3], "width": ["05", "075", "1"]}
BASELINE = {"scales": 3, "width": "1"}  # both point to the baseline config
EPOCH_DURATIONS = [5, 10, 30]


def update_config(base, override):
    for key, value in override.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            update_config(base[key], value)
        else:
            base[key] = value
    return base

def override_path(ablation, value, dataset, epoch):
    if value == BASELINE[ablation]:
        return f"configs/baseline/{epoch}/SleePyCo-Transformer_SL-10_numScales-3_{dataset}_{epoch}_freezefinetune.json"
    if ablation == "scales":
        return f"configs/ablations_scales/{value}/{epoch}/SleePyCo-Transformer_SL-10_numScales-{value}_{dataset}_{epoch}_freezefinetune.json"
    return f"configs/ablations_width/SleePyCo-Transformer_SL-10_numScales-3_{value}_{dataset}_{epoch}_freezefinetune.json"


def load_config(base_path, ablation, value, dataset, epoch):
    """Base config merged with the ablation override. None if the override is missing."""
    path = override_path(ablation, value, dataset, epoch)
    if not os.path.exists(path):
        print(f"Missing override {path}, skipping.")
        return None
    with open(base_path) as f:
        config = json.load(f)
    with open(path) as f:
        return update_config(config, json.load(f))

def results_path(config):
    return os.path.join("checkpoints", config["name"], "raw_results.json")

def load_results(path):
    if not os.path.exists(path):
        print(f"Missing results {path}, skipping.")
        return {}
    with open(path) as f:
        return json.load(f)

def load_results_aggr(name):
    with open("all_results.json") as f:
        results = json.load(f)
    results = results.get(name)
    if results is None:
        print(f"Missing results for {name}, skipping.")
        return {}
    return results
    

def get_fold_metric(data, dataset, ablation, value, epoch, fold, metric="mf1"):
    """Look up one fold's metric. The baseline may have been stored under either
    ablation (scales=3 or width=1), so both keys are tried. None if absent."""
    keys = [(ablation, str(value))]
    if value == BASELINE[ablation]:
        keys += [(a, str(v)) for a, v in BASELINE.items() if a != ablation]
    for abl, val in keys:
        entry = data.get(dataset, {}).get(abl, {}).get(val, {}).get(str(epoch), {}).get(str(fold))
        if entry:
            return entry[metric]
    return None

def get_fold_metric_aggr(data, fold ):
    per_fold_mf1_scores = data.get('per_fold_mf1', {})
    if not per_fold_mf1_scores:
        print(f"Missing per_fold_mf1 in aggregated results, skipping.")
        return None
    entry = per_fold_mf1_scores.get(str(fold))
    if entry:
        return entry
    return None 