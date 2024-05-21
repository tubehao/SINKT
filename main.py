#!/usr/bin/env python
# encoding: utf-8

import torch
from torch.utils.data import DataLoader

import os
import pickle
import argparse
import logging as log
import importlib
import models

import numpy as np
import random
from dataset_doubletext import KTDataset
import math
from torch.utils.data import Dataset, DataLoader, Subset

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
log.basicConfig(level=log.INFO)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(20)

#############################################
parser = argparse.ArgumentParser(description='multi behavior KT')
parser.add_argument('--debug',          action='store_true',        help='log debug messages or not')
parser.add_argument('--run_exist',      action='store_true',        help='run dir exists ok or not')
parser.add_argument('--run_dir',        type=str,   default='run/', help='dir to save log and models')
parser.add_argument('--data_dir',       type=str)
parser.add_argument('--device',       type=int,   default=1)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--n_epochs', type=int, default=30)
parser.add_argument('--model', type=str, default='dhkt')
parser.add_argument('--LMmodel_name', type=str, default='bert')
parser.add_argument('--data_num', type=int, default=5000)
parser.add_argument('--plan', type=str, default='1')
parser.add_argument('--dataset', type=str, default="doubletext")
parser.add_argument('--n_layer', type=int, default=2)
parser.add_argument('--lamda',type=float, default=0.001)


args = parser.parse_args()
dataset = ['ass09', 'ass12-2000', 'ass12', 'junyi']
for data in dataset:
    if data in args.data_dir:
        dataset_name = data
        break

if not args.debug:
    if not os.path.exists(args.run_dir):
        os.makedirs(args.run_dir)
    args.run_dir = os.path.join(args.run_dir , dataset_name)
    if not os.path.exists(args.run_dir):
        os.makedirs(args.run_dir)
    args.run_dir = os.path.join(args.run_dir , args.model+'bs{}-lr{}-size{}-layer{}-dataNum'.format(args.batch_size, args.lr, args.dim,args.n_layer,args.data_num))
    if not os.path.exists(args.run_dir):
        os.makedirs(args.run_dir)
else:
    args.run_dir = 'debug/'


log_name = 'log_{}-bs{}-lr{}-size{}-layer{}-dataNum{}-plan{}.txt'.format(args.model, args.batch_size, args.lr, args.dim, args.n_layer, args.data_num, args.plan)
log.getLogger().addHandler(log.FileHandler(os.path.join(args.run_dir, log_name), mode='w'))
log.info('args: %s' % str(args))
args.device = 'cpu' if args.device < 0 else 'cuda:%i' % args.device
args.device = torch.device(args.device)

def preprocess():
    datasets = {}
    with open(args.data_dir + 'problem_skill_maxSkillOfProblem_number.pkl', 'rb') as fp:
            problem_number, lesson_number, concept_number, max_concept_of_problem = pickle.load(fp)

    setattr(args, 'max_concepts', max_concept_of_problem)
    setattr(args, 'concept_num', concept_number+1)
    setattr(args, 'dropout', 0.5)
    setattr(args, 'seq_len', 200)
    setattr(args, 'problem_number', problem_number)
    setattr(args, 'lesson_number', lesson_number)
    setattr(args, 'dropout', 0.5)
    setattr(args, 'n_layer', 1)

    if not os.path.exists(os.path.join(args.data_dir, args.LMmodel_name)):
        os.mkdir(os.path.join(args.data_dir, args.LMmodel_name))
    dataset_name = 'dataset_' + args.dataset
    dataset_module = importlib.import_module(dataset_name)

    for split in [ 'valid', 'test','train']:
        file_name =  dataset_name + '_{}.pkl'.format(split)
        file_path = os.path.join(args.data_dir, 'bert', file_name)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                datasets[split] = pickle.load(f)
                log.info('Dataset split %s loaded' % split)
                print('Dataset split %s loaded' % split)
        else:
            datasets[split] = dataset_module.KTDataset(args, split=split)
            with open(file_path, 'wb') as f:
                pickle.dump(datasets[split], f)
            log.info('Dataset split %s created and dumpped' % split)
            print('Dataset split %s created and dumpped' % split)

    loaders = {}
    for split in ['train', 'valid', 'test']:
        if split == 'train' and datasets[split].data_list['q_seq'].shape[0] > args.data_num:
            datasets[split] = Subset(datasets[split], list(range(args.data_num)))
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=args.batch_size,
            shuffle=True if split == 'train' else False
        )
        
    return datasets, loaders


if __name__=='__main__':
    torch.set_num_threads(2)
    datasets, loaders = preprocess()
    Model = getattr(models, args.model)
    model = Model(args).to(args.device)
    train_module = importlib.import_module('train')
    print('Model initialized!')
    train_module.train(model, loaders, args)


    