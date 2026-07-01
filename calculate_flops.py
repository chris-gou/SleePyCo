from sympy import false
import torchprofile
import torch
from torchinfo import summary
import json
import calflops
from models.sleepyco import SleePyCoBackbone
from models.classifiers import Transformer
from models.main_model import MainModel
import os

results_file = "flops_results.json"
results = {}

def calculate_flops(config, model, model_name=None):
    epoch_duration = config['training_params']['epoch_duration']
    input_tensor = torch.randn(1, 1, 100*epoch_duration) # batch size 1, 1 channels, 3000 time points (30s at 100Hz)
    input_shape = (1, 1, 100*epoch_duration) # batch size 1, 1 channels, 3000 time points (30s at 100Hz)

    model.eval() # set to eval mode
    with torch.no_grad(): # disable gradient calculation
        macs = torchprofile.profile_macs(model, args=(input_tensor,))
    with torch.no_grad():
        flops, macs, params = calflops.calculate_flops(model=model, 
                                        input_shape=input_shape,
                                        output_as_string=False,
                                        output_precision=4)
    print(f"{model_name} FLOPs:{flops}   MACs:{macs}   Params:{params} \n")
    model_stats = summary(model, input_size=input_shape, col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], depth=5, verbose=0)
    stats_dict = {
        "total_params": model_stats.total_params,
        "trainable_params": model_stats.trainable_params,
        "total_mult_adds": model_stats.total_mult_adds,
        # "input_bytes": model_stats.total_input_bytes,
        # "output_bytes": model_stats.total_output_bytes,
        "param_bytes": model_stats.total_param_bytes,
        "layers": [
            {
                "name": str(layer.class_name),
                "input_size": str(layer.input_size),
                "output_size": str(layer.output_size),
                "num_params": layer.num_params,
                "mult_adds": layer.macs,
            }
            for layer in model_stats.summary_list
        ]
    }
    print(f"==================================== END {model_name} FLOP calculation for {epoch_duration}s, width: {config['training_params'].get('width_multiplier', 1.0)}, scales: {config['training_params'].get('num_scales', 3)}  ====================================\n")
    return flops, macs, params, stats_dict if stats_dict else None


def calculate_flops_total(config, key):
    model = MainModel(config)
    backbone = SleePyCoBackbone(config)
    backbone_flops, backbone_macs, backbone_params, backbone_stats = calculate_flops(config, backbone, model_name="Backbone")
    total_flops, total_macs, total_params, total_stats = calculate_flops(config, model, model_name="Total")
    results = {}
    results[key] = {
        "backbone": {
            "FLOPs": backbone_flops,
            "MACs": backbone_macs,
            "Params": backbone_params,
            "Summary": str(backbone_stats),
        },
        "classifier": {
            "FLOPs": total_flops - backbone_flops,
            "MACs": total_macs - backbone_macs,
            "Params": total_params - backbone_params,
            "Summary": str(total_stats),
        },
        "total": {
            "FLOPs": total_flops,
            "MACs": total_macs,
            "Params": total_params,
            "Summary": str(total_stats),
        }
    }
    return results

if __name__ == "__main__":
    # create "config" per ablation
    epoch_durations = [5, 10, 30]
    widths = [0.5, 0.75, 1.0]
    scales = [1, 2, 3]

    # base config
    config = {
         "dataset": {
        "name": "SHHS",
        "eeg_channel": "EEG",
        "num_splits": 1,
        "seq_len": 10,
        "target_idx": -1,
        "root_dir": "./" ,
        "max_subjects": 600},

        "backbone": {
            "name": "SleePyCo",
            "init_weights": false,
            "dropout": false
        },
        "feature_pyramid": {
            "dim": 128,
            "num_scales": 3
        },
        "classifier": {
            "name": "Transformer",
            "model_dim": 128,
            "feedforward_dim": 128,
            "pool": "attn",
            "dropout": false,
            "num_classes": 5,
            "pos_enc": {
                "dropout": false
            }
        },
        "training_params": {
            "mode": "freezefinetune",
            "max_epochs": 500,
            "batch_size": 64,
            "lr": 0.0005,
            "weight_decay": 0.0001,
            "val_period": 500,
            "early_stopping": {
                "mode": "min",
                "patience": 10
            },
            "epoch_duration": 30,
            "base_epoch_duration": 30,
            "fs": 100,
            "hop": 0,
        }
    }

    # variable stuff in configs: widt_multiplier, num_scales, epoch_duration

    # with open(config_path) as config_file:
    #     config = json.load(config_file)
    #baseline

    for epoch_duration in epoch_durations:
        for width in widths:
            key = f"{epoch_duration}s_width_{width}_scales_3"
            config['training_params']['epoch_duration'] = epoch_duration
            config['training_params']['width_multiplier'] = width
            result = calculate_flops_total(config, key)
            results[key] = result[key]
        for num_scales in scales:
            key = f"{epoch_duration}s_width_1.0_scales_{num_scales}"
            config['feature_pyramid']['num_scales'] = num_scales   
            config['training_params']['epoch_duration'] = epoch_duration
            result = calculate_flops_total(config, key)
            results[key] = result[key]

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)