#!/bin/bash
CONFIGS_CRL=("SleePyCo-Transformer_SL-10_numScales-3_075_Sleep-EDF-2018_10_freezefinetune" "SleePyCo-Transformer_SL-10_numScales-3_075_Sleep-EDF-2018_5_freezefinetune"
            "SleePyCo-Transformer_SL-10_numScales-3_05_Sleep-EDF-2018_10_freezefinetune" "SleePyCo-Transformer_SL-10_numScales-3_05_Sleep-EDF-2018_5_freezefinetune")

CONFIGS_SCALES=("SleePyCo-Transformer_SL-10_numScales-2_Sleep-EDF-2018_10_freezefinetune" "SleePyCo-Transformer_SL-10_numScales-2_Sleep-EDF-2018_5_freezefinetune"
            "SleePyCo-Transformer_SL-10_numScales-1_Sleep-EDF-2018_10_freezefinetune" "SleePyCo-Transformer_SL-10_numScales-1_Sleep-EDF-2018_5_freezefinetune"
            "SleePyCo-Transformer_SL-10_numScales-1_Sleep-EDF-2018_freezefinetune" "SleePyCo-Transformer_SL-10_numScales-2_Sleep-EDF-2018_freezefinetune")
CONFIGS_PATH="configs/ablations_width"

python train_mtcl.py --config "configs/SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2018_freezefinetune.json" --override "configs/ablations_scales/2/10/SleePyCo-Transformer_SL-10_numScales-2_Sleep-EDF-2018_10_freezefinetune.json"  --resume
python train_mtcl.py --config "configs/SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2018_freezefinetune.json" --override "configs/ablations_scales/2/5/SleePyCo-Transformer_SL-10_numScales-2_Sleep-EDF-2018_5_freezefinetune.json"  --resume
python train_mtcl.py --config "configs/SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2018_freezefinetune.json" --override "configs/ablations_scales/1/10/SleePyCo-Transformer_SL-10_numScales-1_Sleep-EDF-2018_10_freezefinetune.json"  --resume
python train_mtcl.py --config "configs/SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2018_freezefinetune.json" --override "configs/ablations_scales/1/5/SleePyCo-Transformer_SL-10_numScales-1_Sleep-EDF-2018_5_freezefinetune.json"  --resume
python train_mtcl.py --config "configs/SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2018_freezefinetune.json" --override "configs/ablations_scales/1/30/SleePyCo-Transformer_SL-10_numScales-2_Sleep-EDF-2018_freezefinetune.json"  --resume
python train_mtcl.py --config "configs/SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2018_freezefinetune.json" --override "configs/ablations_scales/2/30/SleePyCo-Transformer_SL-10_numScales-2_Sleep-EDF-2018_freezefinetune.json"  --resume