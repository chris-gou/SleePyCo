import numpy as np
import sys

import json
import argparse
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

LABELS = [0, 1,2,3,4]
CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]
THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent.parent

sys.path.insert(0,str(ROOT))
from loader import *
from test import OneFoldEvaluator

CHECKPOINTS_DIR = ROOT / 'checkpoints'
CONFIGS_DIR = ROOT / 'configs'

def get_cm(path, config, fold):
    cm = np.load(os.path.join(path, f'fold_{fold}_reconstructed.npz'))['confmat']
    return cm

def evaluate_fold(args, fold, config):
    save_path = os.path.join(CHECKPOINTS_DIR, config['name'])
    evaluator = OneFoldEvaluator(args, fold, config)
    y_true, y_pred, _, _, _ = evaluator.run()
    y_pred_labels = y_pred.argmax(axis=1)  
    cm = skmet.confusion_matrix(y_true, y_pred_labels)
    np.savez(os.path.join(save_path, f"fold_{fold}_reconstructed.npz",), y_true=y_true, y_pred=y_pred, confmat=cm)

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
    parser.add_argument('--gpu', type=str, default="0", help='gpu id(s)')
    parser.add_argument('--config', type=str, default="SleePyCo", help='.json')
    parser.add_argument('--test-only', action='store_true', help='if true, only evaluation is conducted')
    parser.add_argument('--override', type=str, help='config override path')
    parser.add_argument('--resume', action='store_true', help='if true, resume training from checkpoint')
    parser.add_argument('--cpu', action='store_true', help='if true, use cpu for training and evaluation')
    parser.add_argument('--streaming', action='store_true', help='if true, use streaming evaluation')
    parser.add_argument('--overwrite', action='store_true', help='if true, overwrite existing confusion matrices')
    args = parser.parse_args()

    override_path = args.override
    config_path = args.config
    with open(config_path) as config_file:
        config = json.load(config_file)
    with open(override_path) as f:
        override_config = json.load(f)
    update_config(config, override_config)

    print(f"Loaded config from {config_path} and override from {override_path}")
    available_cm = 0
    confmats = []
    confmats_available = {} # keep track of which folds have available confusion matrices
    
    # get nr of folds based on how many files there are
    checkpoint_path = f'{CHECKPOINTS_DIR}/{config["name"]}_new_data'
    for file in os.listdir(checkpoint_path):
        if file.startswith('fold') and file.endswith('_reconstructed.npz'):
            fold = int(file.split('_')[1])  # extract fold number from filename
            available_cm += 1
            confmats_available[fold] = True
            confmats.append(get_cm(checkpoint_path, config, fold))

    total_folds = config['dataset']['num_splits']
    # total_folds = 1 # for test
    for fold in range(1, total_folds+1):
        if fold in confmats_available and not args.overwrite:
            print(f"Confusion matrix for fold {fold} already exists. Skipping evaluation.")
            continue
        else:
            # break
            print(f"Evaluating fold {fold}...")
            evaluate_fold(args, fold, config)
