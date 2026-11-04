import os
import json
import argparse
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import *
from loader import EEGDataLoader
from train_mtcl import OneFoldTrainer
from models.main_model import MainModel
from cpu_usage import *

from pyutils.helpers import notify

class OneFoldEvaluator(OneFoldTrainer):
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold
        
        self.cfg = config
        self.ds_cfg = config['dataset']
        self.tp_cfg = config['training_params']
        
        if args.cpu:
            self.device = torch.device('cpu') # force to use cpu
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('[INFO] Config name: {}'.format(config['name']))

        if torch.cuda.is_available():
            torch.cuda.synchronize() # profile kernel execution time

        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()
        
        self.criterion = nn.CrossEntropyLoss()
        self.ckpt_path = os.path.join('checkpoints', config['name'])
        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)
        self.events = EventMarker()
        
    def build_model(self):
        model = MainModel(self.cfg)

        if self.device.type == 'cpu':
            model = model.to(self.device)
            print('[INFO] Model prepared, Device used: {}'.format(self.device))
        else:
            model = torch.nn.DataParallel(model, device_ids=list(range(len(self.args.gpu.split(","))))) # parallelize since we're using gpu
            model.to(self.device)
            print('[INFO] Model prepared, Device used: {} GPU:{}'.format(self.device, self.args.gpu))

        print('[INFO] Number of params of model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

        return model
    
    def build_dataloader(self):
        test_dataset = EEGDataLoader(self.cfg, self.fold, set='test')
        if self.args.streaming:
            test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)
        else:
            test_loader = DataLoader(dataset=test_dataset, batch_size= self.tp_cfg['batch_size'], shuffle=False,
                          num_workers=4*len(self.args.gpu.split(",")), pin_memory=True)
        print('[INFO] Dataloader prepared')
        return {'test': test_loader} 
   
    def run(self):
        print('\n[INFO] Fold: {}'.format(self.fold))
        test_cpu_sampler = CPUSampler(interval=1)
        test_gpu_sampler = GPUSampler(interval=1)
        self.events.start_time = time.time()
        test_cpu_sampler.start()
        test_gpu_sampler.start()

        plots_dir = os.path.join('plots', self.cfg['name'])
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, f'fold{self.fold:02d}_testing_resources_{self.device}.png')
        periodic_plotter = PeriodicPlotter(test_cpu_sampler, test_gpu_sampler, plot_path, interval_sec=1*3600) # plot every 2 hours
        periodic_plotter.start()

        state_dict = torch.load(os.path.join(self.ckpt_path, self.ckpt_name), map_location=self.device) # get the dictionary with the model parameters
        if self.device.type == 'cpu':
            print('[INFO] Loading model on CPU')
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()} # remove module prefix if DataParallel was not used
        self.model.load_state_dict(state_dict)
        print('[INFO] Model loaded from {}'.format(os.path.join(self.ckpt_path, self.ckpt_name)))

        y_true, y_pred, latency_stats, inference_times, cpu_stats = self.evaluate(mode='test')
        save_cm(y_true, y_pred, self.ckpt_path+'_new_data', self.fold)
        return y_true, y_pred, latency_stats, inference_times, cpu_stats

def update_config(base_config, override_config):
    for key, value in override_config.items():
        if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
            update_config(base_config[key], value)
        else:
            base_config[key] = value
    return base_config

def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    warnings.filterwarnings("ignore", category=UserWarning) 

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument('--config', type=str, help='config file path')
    parser.add_argument('--override', type=str, help='config override path')
    parser.add_argument('--threads', type=int, default=None, help='CPU threads for torch')
    parser.add_argument('--cpu', action='store_true', help='cpu evaluation')
    parser.add_argument('--streaming', action='store_true', help='streaming evaluation')
    parser.add_argument('--max_preds', type=int, default=None, help='cap on streaming predictions per fold (latency only)')
    args = parser.parse_args()

    if args.threads is not None:
        torch.set_num_threads(args.threads)
    print(f"[INFO] Torch threads set to: {torch.get_num_threads()}")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    all_latency_stats = []
    with open(args.config) as config_file:
        config = json.load(config_file)
    if args.override:
        with open(args.override) as f:
            override_config = json.load(f)
        update_config(config, override_config)
    
    if config['name'] is None:
        config['name'] = os.path.basename(args.config).replace('.json', '') 
    # config['name'] = config['name']+"_new_data"
    Y_true = np.zeros(0)
    Y_pred = np.zeros((0, config['classifier']['num_classes']))

    try:

        for fold in range(1, config['dataset']['num_splits'] + 1):
            evaluator = OneFoldEvaluator(args, fold, config)
            y_true, y_pred, latency_stats, inference_times, cpu_stats = evaluator.run()
            Y_true = np.concatenate([Y_true, y_true])
            Y_pred = np.concatenate([Y_pred, y_pred])
            all_latency_stats.append(latency_stats)

            valid_stats = [ls for ls in all_latency_stats if ls['mean'] is not None]
            if valid_stats:
                avg_latency = {
                    'mean': np.mean([ls['mean'] for ls in valid_stats]),
                    'std': np.mean([ls['std'] for ls in valid_stats]),
                    'min': np.min([ls['min'] for ls in valid_stats]),
                    'max': np.max([ls['max'] for ls in valid_stats]),
                    'p50': np.mean([ls['p50'] for ls in valid_stats]),
                    'p95': np.mean([ls['p95'] for ls in valid_stats]),
                    'p99': np.mean([ls['p99'] for ls in valid_stats]),
                }
            else:
                avg_latency = None
            summarize_result(config, fold, Y_true, Y_pred, streaming=args.streaming, latency_stats=avg_latency, inference_times=inference_times, cpu_sampler=cpu_stats, device=evaluator.device)
            class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
            plot_confusion_and_f1(Y_true, Y_pred, class_names, f"plots/{config['name']}_{fold}")
        plot_confusion_and_f1(Y_true, Y_pred, class_names, f"plots/{config['name']}_all_folds")
        notify("SLP evaluation completed for config: {}".format(config['name']))
    except Exception as e:
        notify("SLP evaluation failed for config: {}. Error: {}".format(config['name'], str(e)))
        raise e
if __name__ == "__main__":
    main()
