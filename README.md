# SINKT
## Introduction
This is the pytorch implementation of SINKT.
## Requirements
+ numpy
+ pickle
+ transformers
+ torch
+ torch_geometric

## Quick Start
Run following code in command line to start:
```
python main.py --data_dir /path --run_dir run/ --model GCNKT --dataset doubletext --LMmodel_name bert --n_layer 2 --dim 256 --n_epochs 30 --data_num 2000 --lr 0.0005 --batch_size 64
```
