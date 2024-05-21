import tqdm
import os
import pickle
import logging as log
import torch
from torch.utils import data
import math
import random
import numpy as np
# import bert
from transformers import AutoTokenizer, AutoModel
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, BertForMaskedLM
import openai
import click
import time
import csv
import jsonlines
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel


class KTDataset(data.Dataset):
    def __init__(self, args, split):
        with open(args.data_dir + 'problem_skill_maxSkillOfProblem_number.pkl', 'rb') as fp:
            problem_number, lesson_num, concept_number, max_concept_of_problem = pickle.load(fp)
        self.problem_number = problem_number
        self.max_concept_of_problem = max_concept_of_problem
        self.concept_num = concept_number
        self.split = split
        self.path = args.data_dir
        self.max_len = 200
        self.device = args.device
        self.args = args
        self.LMmodel_name = args.LMmodel_name

        if self.LMmodel_name == 'bert':
            self.BERT_EMB_SIZE = 512
        elif self.LMmodel_name == 'sentence_bert':
            self.BERT_EMB_SIZE = 768
        elif self.LMmodel_name == 'codebert':
            self.BERT_EMB_SIZE = 768
        elif self.LMmodel_name == 'llama7b':
            self.BERT_EMB_SIZE = 4096
        elif self.LMmodel_name == 'moss':
            self.BERT_EMB_SIZE = 2048
        elif self.LMmodel_name == 'chatglm2':
            self.BERT_EMB_SIZE = 4096

        if self.LMmodel_name == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained("/path")
            self.bert_model = BertModel.from_pretrained("/path").to(self.device)
        elif self.LMmodel_name == 'sentence_bert':
            self.bert_model = SentenceTransformer('KBLab/sentence-bert-swedish-cased')
        elif self.LMmodel_name == 'codebert':
            tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            self.bert_model = AutoModel.from_pretrained("microsoft/codebert-base")
        elif self.LMmodel_name == 'llama7b':
            weight_path = "/path"
            self.tokenizer = LlamaTokenizer.from_pretrained(weight_path)
            self.bert_model = LlamaForCausalLM.from_pretrained(weight_path).to(self.device)
        elif self.LMmodel_name == "moss":
            path = "/path"
            self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            self.tokenizer.eos_token_id = 106068  # The eos_token_id of base model is 106028. We need map the eos token to <eom> (its token id is 106068)
            self.bert_model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, use_cache=True).to(self.device)
        elif self.LMmodel_name == 'chatglm2':
            path = '/path'
            self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            self.bert_model = AutoModel.from_pretrained(path, trust_remote_code=True).to(self.device)
        else:
            print('no model')

        log.info('Processing data...')
        self.process()
        self.data_size = len(self.data_list['len_list'])
        self.tokenizer = None
        self.bert_model = None
        log.info('Processing data done!')
        

    def __len__(self):
        return self.data_size

    def __getitem__(self, i):
        return self.data_list['len_list'][i], self.data_list['q_seq'][i], 0, \
            self.data_list['concepts'][i], self.data_list['operate'][i], 0, \
            self.data_list['mask'][i], 0, 0,0
   


    def reduce_dataset(self, keep_indices):
        """
        Reduce the dataset to only keep data items with indices in keep_indices.
        """
        for key in self.data_list:
            if isinstance(self.data_list[key], torch.Tensor):
                self.data_list[key] = self.data_list[key][keep_indices]
            elif isinstance(self.data_list[key], list):
                self.data_list[key] = [self.data_list[key][i] for i in keep_indices]
            else:
                pass

        self.data_size = len(self.data_list['len_list'])
        
    def collate_fn(self, batch):
        bs = len(batch)
        len_list = []
        q_seq = []
        lesson_seq = []
        concepts = []
        operates = []
        masks = []
        btypes = []

        concept_text = []
        stu_text1, stu_text2= [], []
        for i, line in enumerate(batch):
            length, qid, lid, cid, op, bt, mask, c_text, s_text_t, s_text_f = line
            len_list.append(length)
            q_seq.append(qid)
            lesson_seq.append(lid)
            concepts.append(cid)
            operates.append(op)
            masks.append(mask)
            btypes.append(bt)
            concept_text.append(c_text)
            stu_text1.append(s_text_t)
            stu_text2.append(s_text_f)


        return (torch.tensor(len_list), torch.stack(q_seq), torch.stack(lesson_seq),
                torch.stack(concepts), torch.stack(operates), torch.stack(btypes),
                torch.stack(masks) ,  torch.stack(concept_text), torch.stack(stu_text1), torch.stack(stu_text2))


    def get_word_vec(self, text):
        if self.LMmodel_name == 'codebert':
            text = self.tokenizer.tokenize(text)
            tokens = [self.tokenizer.cls_token] + text + [self.tokenizer.eos_token]
            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            context_embeddings = []
            for i in range(0, len(tokens_ids), 512):
                context_embedding = self.bert_model(torch.tensor(tokens_ids[i:i + 512])[None, :])[0]
                context_embeddings.append(context_embedding.squeeze(0))
            res = torch.concatenate(context_embeddings, dim=0).mean(dim=0)
            return res  # 768,
        elif self.LMmodel_name == 'bert':
            inputs = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
            with torch.no_grad():
                res = self.bert_model(inputs).last_hidden_state[:, 0]
            return res
        elif self.LMmodel_name == 'sentence_bert':
            res = self.bert_model.encode(text)
            return res
        elif self.LMmodel_name == 'llama7b':
            with torch.no_grad():
                tokens = self.tokenizer(text, return_tensors="pt", return_attention_mask=True).to(self.device)
                outputs = self.bert_model(**tokens, output_hidden_states=True, return_dict=True)
                hidden_states = outputs.hidden_states 
                text_encoding = hidden_states[32][0, 0, :]
                return text_encoding

        elif self.LMmodel_name == 'chatglm2':
            with torch.no_grad():
                x = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt",
                              return_attention_mask=True).to(self.device)
                mask = x['attention_mask']
                x.pop('attention_mask')
                outputs = self.bert_model.transformer(**x, output_hidden_states=True, return_dict=True)
                outputs.last_hidden_state = outputs.last_hidden_state.transpose(1, 0)
                res = outputs.last_hidden_state.squeeze(0).mean(dim=0)
                return res
        else:
            assert False

    def read_data(self, records):
        qid, conceptid, operate, st_list_true, st_list_false, ct_list, mask =[], [], [], [], [], [],[]
        length = 0
        prev_lid = 0
        problem_prompt = ['A question contains the concept ', '.']
        stu_prefix = ['A student correctly answers concepts:', 'A student mistakenly answers concepts:']
        prefix_problems_true,  prefix_problems_false = [], []

        for idx, record in enumerate(records):
            if idx >= 200:
                break

            question_id, concepts, response, concept_text = record

            qid.append(question_id)
            conceptid.append(concepts + [0] * (self.max_concept_of_problem - len(concepts)))
            operate.append(response)
            mask.append(1)
            length += 1
            text1 = problem_prompt[0] + concept_text + problem_prompt[1] + self.concept_text_sum[str(concepts[0])]
            text2 = stu_prefix[0] + '|'.join(list(set(prefix_problems_true))) + '.'
            text3 = stu_prefix[1] + '|'.join(list(set(prefix_problems_false))) + '.'
            if prefix_problems_true == []:
                text2 = "The student has no correct record in the system."
            if prefix_problems_false == []:
                text3 = 'The student has no wrong record in the system.'
            if response == 1:
                prefix_problems_true.append(self.concept_name_text[str(concepts[0])][:-1])
            else:
                prefix_problems_false.append(self.concept_name_text[str(concepts[0])][:-1])

            ct_list.append(self.get_word_vec(text1).squeeze().cpu().numpy())
            st_list_true.append(self.get_word_vec(text2).squeeze().cpu().numpy())
            st_list_false.append(self.get_word_vec(text3).squeeze().cpu().numpy())

        if length == 0:
            return 0, 0, 0, 0, 0, 0, 0, 0
        align_num = self.max_len - len(qid)
        qid += [0] * align_num
        conceptid += [[0] * self.max_concept_of_problem] * align_num
        operate += [0] * align_num
        mask += [0.] * align_num
        ct_list +=[ np.zeros(ct_list[-1].shape)] * align_num
        st_list_true += [np.zeros(st_list_true[-1].shape)] * align_num
        st_list_false += [np.zeros(st_list_false[-1].shape)] * align_num
        return length, qid, qid, conceptid, operate, qid, mask, ct_list, [st_list_true, st_list_false]

    def process(self):
        with open(self.path + 'history_' + self.split + '.pkl', 'rb') as fp:
            histories = pickle.load(fp)

        with open(os.path.join(self.path, 'id_skill_desc_dict.json'), 'rb') as f:
            self.concept_text_sum = json.load(f)

        with open(os.path.join(self.path, 'id_skillname_dict.json'), 'rb') as f:
            self.concept_name_text = json.load(f)

        len_list = []
        q_tensor = []
        lesson_tensor = []
        concept_tensor = []
        operate_tensor = []
        mask_tensor = []
        btype_tensor = []
        concept_text, stu_text_true, stu_text_false  = [], [], []
        histories = histories[:6000]
        for record in tqdm.tqdm(histories):
            if record[0] < 10:
                continue
            length, qid, lid, concepts, operate, btype, mask, c_text, s_text = self.read_data(record[1])

            if length == 0:
                continue
            len_list.append(length)
            q_tensor.append(qid)
            lesson_tensor.append(lid)
            concept_tensor.append(concepts)
            operate_tensor.append(operate)
            mask_tensor.append(mask)
            btype_tensor.append(btype)
            stu_text_true.append(s_text[0])
            stu_text_false.append(s_text[1])
            concept_text.append(c_text)

        self.data_list = {
            'len_list': len_list,
            'q_seq': torch.tensor(q_tensor).long().cpu(),
            'lesson_seq': torch.tensor(lesson_tensor).long().cpu(),
            'concepts': torch.tensor(concept_tensor).long().cpu(),
            'operate': torch.tensor(operate_tensor).long().cpu(),
            'btype': torch.tensor(btype_tensor).float().cpu(),
            'mask': torch.tensor(mask_tensor).float().cpu(),
            'stu_text_t': torch.tensor(stu_text_true).float().cpu(),
            'stu_text_f':torch.tensor(stu_text_false).float().cpu(),
            'concept_text': torch.tensor(concept_text).float().cpu()
        }
        self.data_num = len(q_tensor)


