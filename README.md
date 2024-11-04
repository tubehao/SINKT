# SINKT: A Structure-Aware Inductive Knowledge Tracing Model with Large Language Model

## Introduction
This repository contains the PyTorch implementation of the paper SINKT: A Structure-Aware Inductive Knowledge Tracing Model with Large Language Model. You can access the paper [here](https://dl.acm.org/doi/10.1145/3627673.3679760).

## Requirements
+ numpy
+ pickle
+ transformers
+ torch
+ torch_geometric

## Quick Start
To get started, run the following command in your terminal:
```
python main.py --data_dir /path --run_dir run/ --model SINKT --dataset doubletext --LMmodel_name bert --n_layer 2 --dim 256 --n_epochs 30 --data_num 2000 --lr 0.0005 --batch_size 64
```
## Dataset
The processed datasets we used are available for download [here](https://drive.google.com/drive/folders/1RlqySlxhhZIHXCo3NVrE3agmisf-2esw?usp=sharing), which include ASSIST09, ASSIST12, and Junyi. For usage instructions, please refer to [dataset_doubletext.py](https://github.com/tubehao/SINKT/blob/main/dataset_doubletext.py).

Please note that the programming dataset mentioned in our paper is proprietary and cannot be open-sourced.

## Citation
If you use this work in your research, please cite the following paper:
```
@inproceedings{fusinkt,
author = {Fu, Lingyue and Guan, Hao and Du, Kounianhua and Lin, Jianghao and Xia, Wei and Zhang, Weinan and Tang, Ruiming and Wang, Yasheng and Yu, Yong},
title = {SINKT: A Structure-Aware Inductive Knowledge Tracing Model with Large Language Model},
year = {2024},
isbn = {9798400704369},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3627673.3679760},
doi = {10.1145/3627673.3679760},
booktitle = {Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
pages = {632–642},
numpages = {11},
keywords = {inductive learning, knowledge tracing, online education},
location = {Boise, ID, USA},
series = {CIKM '24}
}
```
