import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.modules import module
import numpy as np
import pickle
from torch.distributions import Categorical
import random
import logging as log
from torch.autograd import Variable
from tqdm import tqdm
from torch.autograd import Variable
import os
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from scipy import sparse
from torch_geometric.data import HeteroData

from torch_geometric.nn import GATConv, Linear, to_hetero

class SIEnc(nn.Module):
    def __init__(self, args, hidden_channels, out_channels,add_self_loops=False):
        super().__init__()
        self.args = args
        self.n_layer = args.n_layer
        self.conv_list = nn.ModuleList([GATConv((-1, -1), hidden_channels, add_self_loops=add_self_loops) for _ in range(self.n_layer)])
        self.lin_list = nn.ModuleList([Linear(-1, hidden_channels) for _ in range(self.n_layer)])
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = x.float()
        for i in range(self.n_layer):
            x = self.conv_list[i](x, edge_index) + self.lin_list[i](x)
            x = self.relu(x)
        return x

class SINKT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.node_dim = args.dim
        self.device = args.device
        self.fc1 = nn.Linear(args.dim*2, args.dim*2)
        self.fc2 = nn.Linear(args.dim*2, args.dim)
        self.hidden_dim = self.node_dim
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_dim*4, self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(self.hidden_dim, 1)
        )
        self.sigmoid = torch.nn.Sigmoid()
        self.gru = nn.GRU(self.node_dim * 2, self.node_dim * 2, batch_first=True)
        self.ones = torch.tensor(1.0).to(args.device)
        self.zeros = torch.tensor(0.0).to(args.device)
        
        with open(args.data_dir + 'skillname_id_dict_{}.npy'.format(self.args.LMmodel_name), 'rb') as fp:
            self.c_emb = torch.tensor(np.load(fp)).to(args.device)
            indices = torch.arange(1, self.c_emb.size(0)).to(args.device)
            self.c_emb = nn.Parameter(torch.index_select(self.c_emb.data, 0, indices), requires_grad=False)
        
        self.bert_emb_size = self.c_emb.shape[1]
        if os.path.exists(args.data_dir +'question_emb_{}.npy'.format(self.args.LMmodel_name)):
            with open(args.data_dir + 'question_emb_{}.npy'.format(self.args.LMmodel_name),'rb') as fp:
                self.q_emb = torch.tensor(np.load(fp)).to(args.device)
                indices = torch.arange(1, self.q_emb.size(0)).to(args.device)
                self.q_emb = nn.Parameter(torch.index_select(self.q_emb.data, 0, indices), requires_grad=False)
                self.q_emb_size = self.q_emb.shape[1]
        else:
            self.q_emb_size = self.bert_emb_size
            self.q_emb = nn.Parameter(torch.randn(args.problem_number, self.bert_emb_size).to(args.device), requires_grad=True)
        self.merge_zeros = torch.zeros(1, self.hidden_dim).to(args.device)
        self.gen_graph()
        self.gcn = SIEnc(args,hidden_channels=self.hidden_dim, out_channels=self.hidden_dim)
        self.gcn = to_hetero(self.gcn , self.graph_data.metadata(), aggr='sum')



    def gen_graph(self):
        concept_number = self.args.concept_num
        problem_number = self.args.problem_number
        edge_matrix = torch.from_numpy(sparse.load_npz(self.args.data_dir+ 'adj_matrix.npz').toarray()).float().to(self.args.device)
        self.graph_data = HeteroData()
        if self.q_emb_size != self.bert_emb_size:
            self.graph_data['question'].x = torch.matmul(self.q_emb,self.q_aligner)
        else:
            self.graph_data['question'].x = self.q_emb
        self.graph_data['concept'].x = self.c_emb
        c_c_e = torch.nonzero(edge_matrix == 1).transpose(0,1)
        q_c_e = torch.nonzero(edge_matrix[concept_number:, :concept_number] == 2).transpose(0,1)
        c_q_e = torch.nonzero(edge_matrix[:concept_number, concept_number:] == 2).transpose(0,1)

        self.graph_data['concept', 'prereq', 'concept'].edge_index = c_c_e
        self.graph_data['question','contain','concept'].edge_index = q_c_e
        self.graph_data['concept', 'compose', 'question'].edge_index = c_q_e
        

    def forward(self, q_seq, l_seq, concepts, operate,  btype, text, text1, text2):
        bs = q_seq.shape[0]
        node_features = self.gcn(self.graph_data.x_dict, self.graph_data.edge_index_dict)
        
        q_emb = node_features["question"]
        c_emb = node_features["concept"]
        q_emb_concat = torch.concat([self.merge_zeros, q_emb], dim=0)
        c_emb_concat = torch.concat([self.merge_zeros, c_emb], dim=0)
        e_q = q_emb_concat[q_seq]
        filt = torch.where(concepts == 0, self.zeros, self.ones).to(self.device)
        idx_tmp = torch.arange(0, bs - 1)
        
        e_c_m = nn.functional.embedding(concepts, c_emb_concat, padding_idx=0)
        filt_sum = torch.sum(filt, dim=-1)
        div = torch.where(filt_sum == 0, torch.tensor(1.0).to(self.device), filt_sum).unsqueeze(-1).repeat(1, 1,
                                                                                                           self.node_dim)
        e_c = torch.sum(e_c_m, dim=-2) / div
        v_q = e_c
        v_q_o = torch.cat([v_q.mul(operate.unsqueeze(-1).repeat(1, 1, self.node_dim)),
                           v_q.mul((1 - operate).unsqueeze(-1).repeat(1, 1, self.node_dim))], dim=-1)

        h, _ = self.gru(v_q_o)
        merge_h = torch.zeros(bs, 1, self.node_dim * 2).to(self.device)
        h = torch.concat([merge_h, h], dim=1)[:, :-1, :] 
        predict_x = torch.cat([h, e_q, e_c], dim=-1)
        prob = self.sigmoid(self.predictor(predict_x))
        return prob.squeeze(), 0
