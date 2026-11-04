import numpy as np
import sys

import json
import argparse
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.colors import PowerNorm
import os
from common import *

LABELS = [0, 1,2,3,4]
CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]
THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent.parent
sys.path.insert(0, str(ROOT))
CHECKPOINTS_DIR = ROOT / 'checkpoints'
# CONFIGS_DIR = ROOT / 'configs'

from loader import EEGDataLoader

def subject_lengths(config, fold):
    ds = EEGDataLoader(config, fold, set='test')
    return np.bincount([e[0] for e in ds.epochs])

def get_cm(config, fold, cm_path=None):
    if cm_path is not None:
        cm = np.load(cm_path)['confmat']
    else:
        cm = np.load(os.path.join(CHECKPOINTS_DIR, config['name'], f'fold_{fold}_reconstructed.npz'))['confmat']
    return cm

def plot_cm(cm, folds=None, filename = None):
    cm_dir = 'confusion_matrices'
    os.makedirs(cm_dir, exist_ok=True)
    cm = np.asarray(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_pct = cm / row_sums
    
    labels = np.array([
        f"{count}\n({pct:.1%})"
        for count, pct in zip(cm.flatten(), row_pct.flatten())
    ]).reshape(cm.shape)

    if folds:
        title = f"CM pooled over {folds} folds"
    else:
        title = "CM"

    if filename is None:
        filename = f'confusion_matrix_{name}.png'
    if not filename.endswith('.png'):
        filename += '.png'

    plt.figure(figsize=(6, 5))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    sns.heatmap(row_pct, annot=labels, fmt='', cmap='rocket_r', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                norm=PowerNorm(gamma=0.5, vmin=0, vmax=1), cbar_kws={'label': 'Proportion'})

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.savefig(f'{cm_dir}/{filename}', dpi=200)
    plt.close()

def aggregate(y_true, y_pred, epoch_duration, seq_len, lengths):
    """Average softmax over the K sub-epochs of each 30 s epoch."""
    K = 30 // int(epoch_duration)
    off = (K - (seq_len - 1) % K) % K  # 0 for 10", 3 for 5"
    ts, ps, n_mixed, start = [], [], 0, 0
    for n in lengths:
        t = y_true[start + off:start + n]
        p = y_pred[start + off:start + n]
        start += n
        m = len(t) // K * K
        t = t[:m].reshape(-1, K)
        p = p[:m].reshape(-1, K, p.shape[1]).mean(axis=1)
        n_mixed += (t != t[:, :1]).any(axis=1).sum()
        ts.append(t[:, 0])
        ps.append(p)
    return np.concatenate(ts), np.concatenate(ps), n_mixed

def metrics(y_true, y_pred):
    f1 = skmet.f1_score(y_true, y_pred, average=None, labels=LABELS) * 100
    return {
        'acc': skmet.accuracy_score(y_true, y_pred) * 100,
        'mf1': skmet.f1_score(y_true, y_pred, average='macro') * 100,
        'kappa': skmet.cohen_kappa_score(y_true, y_pred),
        'f1': dict(zip(CLASS_NAMES, f1.tolist())),
        'support': dict(zip(CLASS_NAMES, np.bincount(y_true, minlength=5).tolist())),
    }

def save_results(path, name, entry):
    path = Path(path)
    results = json.loads(path.read_text()) if path.exists() else {}
    results[name] = entry
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(results, indent=2, sort_keys=True))
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written file

def update_config(base_config, override_config):
    for key, value in override_config.items():
        if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
            update_config(base_config[key], value)
        else:
            base_config[key] = value
    return base_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # parser.add_argument('--gpu', type=str, default="0", help='gpu id(s)')
    parser.add_argument('--config', type=str, default="IITNetV2", help='.json')
    parser.add_argument('--test-only', action='store_true', help='if true, only evaluation is conducted')
    parser.add_argument('--override', type=str, help='config override path')
    # parser.add_argument('--resume', action='store_true', help='if true, resume training from checkpoint')
    # parser.add_argument('--cpu', action='store_true', help='if true, use cpu for training and evaluation')
    parser.add_argument('--results', type=str,
                    default=str(Path.home() / 'Documents' / 'thesis_results.json'))
    args = parser.parse_args()

    args.test_only = True
    config_path = args.config

    with open(config_path) as config_file:
        config = json.load(config_file)
    if args.override:
        with open(args.override) as f:
            override_config = json.load(f)
        update_config(config, override_config)
        
    name = config.get('name', Path(config_path).stem)
    name = name + "_new_data"
    epoch_duration = config["training_params"].get('epoch_duration', 30)

    run_dir = CHECKPOINTS_DIR / name
    paths = sorted(run_dir.glob('fold_*_reconstructed.npz'))
    if len(paths) != config["dataset"]['num_splits']:
        print(f"Warning: number of folds found ({len(paths)}) does not match the expected number of folds ({config['dataset']['num_splits']})")
        raise ValueError(f"Number of folds found ({len(paths)}) does not match the expected number of folds ({config['dataset']['num_splits']})")
    
    all_true, all_pred, native_true, native_pred = [], [], [], []
    per_fold_mf1 = {}

    for path in paths:
        fold = int(path.name.split('_')[1])
        d = np.load(path)
        y_true, y_pred = d['y_true'].astype(int), d['y_pred']
        native_true.append(y_true)
        native_pred.append(y_pred.argmax(axis=1))

        if epoch_duration != 30:
            lengths = subject_lengths(config, fold)
            assert lengths.sum() == len(y_true), f"{path.name}: length mismatch"
            y_true, y_pred, n_mixed = aggregate(y_true, y_pred, epoch_duration, 10, lengths)
            print(f"{path.name}: {n_mixed}/{len(y_true)} mixed-label groups")
        all_true.append(y_true)
        all_pred.append(y_pred.argmax(axis=1))
        per_fold_mf1[fold] = skmet.f1_score(y_true, y_pred.argmax(axis=1), average='macro') * 100

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    entry = {
        'epoch_duration': epoch_duration,
        'num_folds': len(paths),
        'per_fold_mf1': per_fold_mf1,
        **metrics(y_true, y_pred)
    }
    if epoch_duration != 30:
        entry['native'] = metrics(np.concatenate(native_true), np.concatenate(native_pred))

    save_results(args.results, name, entry)

    print(f"ACC {skmet.accuracy_score(y_true, y_pred) * 100:.4f}  "
          f"MF1 {skmet.f1_score(y_true, y_pred, average='macro') * 100:.4f}  "
          f"kappa {skmet.cohen_kappa_score(y_true, y_pred):.4f}")
    print("per-class F1:", [float(f"{f1*100:.4f}") for f1 in skmet.f1_score(y_true, y_pred, average=None)])
    # print("per-class F1:", {i: f"{f1*100:.4f}" for i, f1 in enumerate(skmet.f1_score(y_true, y_pred, average=None))})
    print("support W/N1/N2/N3/R:", np.bincount(y_true, minlength=5))

    cm = skmet.confusion_matrix(y_true, y_pred, labels=LABELS)
    plot_cm(cm, folds=len(paths), filename=name)