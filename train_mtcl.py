import os
import json
import argparse
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import *
from loader import EEGDataLoader
from cpu_usage import CPUSampler
from models.main_model import MainModel


class OneFoldTrainer:
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold
        
        self.cfg = config
        self.ds_cfg = config['dataset']
        self.fp_cfg = config['feature_pyramid']
        self.tp_cfg = config['training_params']
        self.es_cfg = self.tp_cfg['early_stopping']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('[INFO] Config name: {}'.format(config['name']))

        self.train_iter = 0
        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()
        
        self.criterion = nn.CrossEntropyLoss()
        self.activate_train_mode()
        self.optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=self.tp_cfg['lr'], weight_decay=self.tp_cfg['weight_decay'])
        
        self.ckpt_path = os.path.join('checkpoints', config['name'])
        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)
        self.early_stopping = EarlyStopping(patience=self.es_cfg['patience'], verbose=True, ckpt_path=self.ckpt_path, ckpt_name=self.ckpt_name, mode=self.es_cfg['mode'])
        

    def build_model(self):
        model = MainModel(self.cfg)
        print('[INFO] Number of params of model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        model = torch.nn.DataParallel(model, device_ids=list(range(len(self.args.gpu.split(",")))))
        if self.tp_cfg['mode'] != 'scratch':
            print('[INFO] Model loaded for finetune')
            load_name = self.cfg['name'].replace('SL-{:02d}'.format(self.ds_cfg['seq_len']), 'SL-01')
            load_name = load_name.replace('numScales-{}'.format(self.fp_cfg['num_scales']), 'numScales-1')
            load_name = load_name.replace(self.tp_cfg['mode'], 'pretrain')
            load_path = os.path.join('checkpoints', load_name, 'ckpt_fold-{0:02d}.pth'.format(self.fold))
            model.load_state_dict(torch.load(load_path), strict=False)
        model.to(self.device)
        print('[INFO] Model prepared, Device used: {} GPU:{}'.format(self.device, self.args.gpu))

        return model
    
    def build_dataloader(self):
        train_dataset = EEGDataLoader(self.cfg, self.fold, set='train')
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=True, num_workers=4*len(self.args.gpu.split(",")), pin_memory=True)
        val_dataset = EEGDataLoader(self.cfg, self.fold, set='val')
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=False, num_workers=4*len(self.args.gpu.split(",")), pin_memory=True)
        test_dataset = EEGDataLoader(self.cfg, self.fold, set='test')
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=False, num_workers=4*len(self.args.gpu.split(",")), pin_memory=True)
        print('[INFO] Dataloader prepared')

        return {'train': train_loader, 'val': val_loader, 'test': test_loader}
    
    def activate_train_mode(self):
        self.model.train()
        if self.tp_cfg['mode'] == 'freezefinetune':
            print('[INFO] Freeze backone')
            self.model.module.feature.train(False)
            for p in self.model.module.feature.parameters():
                p.requires_grad = False

            print('[INFO] Unfreeze conv_c5')
            self.model.module.feature.conv_c5.train(True)
            for p in self.model.module.feature.conv_c5.parameters(): p.requires_grad = True
            
            if self.fp_cfg['num_scales'] > 1:
                print('[INFO] Unfreeze conv_c4')
                self.model.module.feature.conv_c4.train(True)
                for p in self.model.module.feature.conv_c4.parameters(): p.requires_grad = True
                
            if self.fp_cfg['num_scales'] > 2:
                print('[INFO] Unfreeze conv_c3')
                self.model.module.feature.conv_c3.train(True)
                for p in self.model.module.feature.conv_c3.parameters(): p.requires_grad = True
                
    def train_one_epoch(self, epoch):
        correct, total, train_loss = 0, 0, 0

        for i, (inputs, labels) in enumerate(self.loader_dict['train']):
            loss = 0
            total += labels.size(0)
            inputs = inputs.to(self.device)
            labels = labels.view(-1).to(self.device)

            outputs = self.model(inputs)
            outputs_sum = torch.zeros_like(outputs[0])

            for j in range(len(outputs)):
                loss += self.criterion(outputs[j], labels)
                outputs_sum += outputs[j]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            predicted = torch.argmax(outputs_sum, 1)
            correct += predicted.eq(labels).sum().item()
            self.train_iter += 1

            progress_bar(i, len(self.loader_dict['train']), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss / (i + 1), 100. * correct / total, correct, total))
            
            if self.train_iter % self.tp_cfg['val_period'] == 0:
                print('')
                val_acc, val_loss = self.evaluate(mode='val')
                self.early_stopping(val_acc, val_loss, self.model)
                self.activate_train_mode()
                if self.early_stopping.early_stop:
                    break
            
    @torch.no_grad()
    def evaluate(self, mode, cpu_sampling=False):
        streaming = self.args.streaming

        self.model.eval()
        correct, total, eval_loss = 0, 0, 0
        y_true = np.zeros(0)
        y_pred = np.zeros((0, self.cfg['classifier']['num_classes']))

        inference_times = []
        cpu_sampler = None

        if streaming and mode == 'test':
            print(f"[DEBUG] Streaming evaluation enabled for mode: {mode}")
            if cpu_sampling:
                cpu_sampler = CPUSampler(interval=0.01)
                print(f"[DEBUG] Starting CPU sampler for streaming evaluation")
                cpu_sampler.start()


        for i, (inputs, labels) in enumerate(self.loader_dict[mode]):
            loss = 0
            total += labels.size(0)
            inputs = inputs.to(self.device)
            labels = labels.view(-1).to(self.device)

            # evaluation timing
            if streaming and mode == 'test':
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

            outputs = self.model(inputs)

            if streaming and mode == 'test':
                end.record()
                torch.cuda.synchronize()
                inference_times.append(start.elapsed_time(end))

                # Periodic cache flush to prevent CUDA allocator fragmentation
                if i > 0 and i % 500 == 0:
                    torch.cuda.empty_cache()
            outputs_sum = torch.zeros_like(outputs[0])

            for j in range(len(outputs)):
                loss += self.criterion(outputs[j], labels)
                outputs_sum += outputs[j]
                
            eval_loss += loss.item()
            predicted = torch.argmax(outputs_sum, 1)
            correct += predicted.eq(labels).sum().item()
            
            y_true = np.concatenate([y_true, labels.cpu().numpy()])
            y_pred = np.concatenate([y_pred, outputs_sum.cpu().numpy()])

            progress_bar(i, len(self.loader_dict[mode]), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (eval_loss / (i + 1), 100. * correct / total, correct, total))
            
        if streaming and mode == 'test' and cpu_sampling:
            cpu_sampler.stop()
            # cpu_stats = cpu_sampler.stats()

        # Remove warm-up times
        if inference_times:
            warmup = max(10, int(0.05 * len(inference_times)))
            inference_times = inference_times[warmup:]

        def compute_stats(times):
            if len(times) == 0:
                return {k: None for k in ['mean', 'std', 'min', 'max', 'p50', 'p95', 'p99']}
            return {
                'mean': float(np.mean(times)),
                'std':  float(np.std(times)),
                'min':  float(np.min(times)),
                'max':  float(np.max(times)),
                'p50':  float(np.percentile(times, 50)),
                'p95':  float(np.percentile(times, 95)),
                'p99':  float(np.percentile(times, 99)),
            }

        latency_stats = compute_stats(inference_times)

        if mode == 'test' and streaming:
            print(f"\n{'='*50}")
            print(f"Latency Statistics:")
            print("Latency Statistics (streaming, batch_size=1):")
            for key in ['mean', 'std', 'min', 'max', 'p50', 'p95', 'p99']:
                val = latency_stats[key]
                print(f"  {key.upper():>4}: {val:.3f} ms" if val is not None else f"  {key.upper():>4}: N/A")
            if cpu_sampler:
                s = cpu_sampler.stats()
                print(f"  CPU mean: {s['mean']:.1f}%  p95: {s['p95']:.1f}%  max: {s['max']:.1f}%")
            print(f"{'='*50}")

        if mode == 'val':
            return 100. * correct / total, eval_loss
        elif mode == 'test':
            return y_true, y_pred, latency_stats, inference_times, cpu_sampler.stats() if cpu_sampler else None
        else:
            raise NotImplementedError
    
    def run(self):
        trained_epochs = 0

        resume_path = os.path.join(self.ckpt_path, self.ckpt_name)
        if os.path.exists(resume_path):
            print(f'[INFO] Resuming from checkpoint: {resume_path}')
            self.model.load_state_dict(torch.load(resume_path))
            # Restore early stopping best score so it doesn't overwrite immediately
            self.early_stopping.best_score = None  # will re-evaluate on first val step

        for epoch in range(self.tp_cfg['max_epochs']):
            print('\n[INFO] Fold: {}, Epoch: {}'.format(self.fold, epoch))
            self.train_one_epoch(epoch)
            trained_epochs += 1
            if self.early_stopping.early_stop:
                break
        
        self.model.load_state_dict(torch.load(os.path.join(self.ckpt_path, self.ckpt_name)))
        y_true, y_pred, latency, inference_times, cpu_sampler = self.evaluate(mode='test')
        print('')

        return y_true, y_pred, latency, trained_epochs, inference_times, cpu_sampler
    
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

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # For reproducibility
    set_random_seed(args.seed, use_cuda=True)

    with open(args.config) as config_file:
        config = json.load(config_file)
    if args.override:
        with open(args.override) as f:
            override_config = json.load(f)
        update_config(config, override_config)
    
    if config['name'] is None:
        config['name'] = os.path.basename(args.config).replace('.json', '') 
    
    Y_true = np.zeros(0)
    Y_pred = np.zeros((0, config['classifier']['num_classes']))

    # latency stats
    all_latency_stats = []

    epochs_per_fold = []

    try:
        for fold in range(1, config['dataset']['num_splits'] + 1):
            trainer = OneFoldTrainer(args, fold, config)
            y_true, y_pred, latency_stats, trained_epochs, inference_times, cpu_sampler  = trainer.run()
            epochs_per_fold.append(trained_epochs)
            Y_true = np.concatenate([Y_true, y_true]) # accumulate across folds
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
            
            summarize_result(config, fold, Y_true, Y_pred, latency_stats=avg_latency,
                              epochs_per_fold=epochs_per_fold, inference_times=inference_times, cpu_sampler=cpu_sampler)

    except Exception as e:
        print(f"[ERROR] An error occurred during training of config {config['name']} and fold {fold}: {e}")
        

if __name__ == "__main__":
    main()
