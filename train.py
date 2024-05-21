import os
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import accuracy_score
import logging as log
import numpy as np
from tqdm import tqdm
import pickle
import os

from tensorboardX import SummaryWriter

'''
train with concept_text, stu_text_true, stu_text_false
dataset_doubletext
'''


def get_list(len_list, logits, operate, mask):
    logit_list = []
    label_list = []
    for i, l in enumerate(len_list):
        mask_i = mask[i]
        idx = torch.nonzero(mask_i == 1).squeeze()
        if logits[i][idx].shape:
            logit_list.append(logits[i][idx])
            label_list.append(operate[i][idx])
    logit_list = torch.concat(logit_list, axis=0)
    label_list = torch.concat(label_list, axis=0)
    return logit_list, label_list


def get_acc_of_ques(len_list, logits, operate, q_seq, args, mask):
    problem_acc = np.zeros([args.problem_number, 2])
    for i, l in enumerate(len_list):
        for pos in range(l):
            qid = q_seq[i][pos]
            if mask[i][pos] == 1:
                problem_acc[qid][1] += 1
                if logits[i][pos] == operate[i][pos]:
                    problem_acc[qid][0] += 1
    return problem_acc


def train(model, loaders, args):
    log.info("training...")
    BCELoss = torch.nn.BCELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20)
    train_sigmoid = torch.nn.Sigmoid()
    best_auc = 0

    if not args.debug:
        summary_name = args.model + '_' + args.LMmodel_name
        summary_name += '_lr' + str(args.lr)
        summary_name += '_bs' + str(args.batch_size)
        if not os.path.exists(os.path.join(args.run_dir, args.model)):
            os.mkdir(os.path.join(args.run_dir, args.model))

        writer = SummaryWriter(os.path.join(args.run_dir, args.model, summary_name))

    for epoch in range(args.n_epochs):
        loss_all, acc_all, auc_all, count_all, student_all = 0, 0, 0, 0, 0
        for data in loaders['train']:
            with (torch.no_grad()):
                len_list, q_seq, lesson_seq, concepts, operate, btype, mask, c_text, s_text_t, s_text_f = data
                q_seq = q_seq.to(args.device)
                lesson_seq = lesson_seq.to(args.device)
                concepts = concepts.to(args.device)
                operate = operate.to(args.device)
                mask = mask.to(args.device)
                c_text = c_text.to(args.device)
                s_text_t = s_text_t.to(args.device)
                s_text_f = s_text_f.to(args.device)


            model.train()
            if args.model == "gkt" or args.model == "akt" or args.model == "skt" or args.model == "lbkt" or args.model == "simple_kt" or args.model == "iekt":
                predict, pre_hiddenstates, logits, extra_loss = model(len_list, q_seq, lesson_seq, concepts, operate, btype, c_text, s_text_t, s_text_f)
            else:
                logits, extra_loss = model(q_seq, lesson_seq, concepts, operate, btype, c_text, s_text_t, s_text_f)

            loss_tensor = BCELoss(logits, operate.float()) * mask
            
            loss = torch.sum(loss_tensor) / torch.sum(mask) + extra_loss * args.lamda

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logit_list, label_list = get_list(len_list, logits, operate, mask)

            with torch.no_grad():
                loss_all += loss.item()
                count_all += torch.sum(mask)
                hat_y = (logit_list > 0.5).int()
                acc = accuracy_score(label_list.cpu().numpy(), hat_y.detach().cpu().numpy())
                fpr, tpr, thresholds = metrics.roc_curve(label_list.int().cpu().numpy(), hat_y.detach().cpu().numpy(),
                                                         pos_label=1)
                auc = metrics.auc(fpr, tpr)
                auc_all += auc
                acc_all += acc
                student_all += 1

        loss = loss_all / count_all
        acc = acc_all / student_all
        auc = auc_all / student_all

        vacc, vauc, vll, vlogits, problem_acc, v_pn_acc = evaluate(model, loaders['valid'], args)
        tacc, tauc, tll, tlogits, tproblem_acc, t_pn_accs = evaluate(model, loaders['test'], args)
        problem_acc += tproblem_acc
        if not args.debug:
            writer.add_scalar("valid/ll", vll, epoch)
            writer.add_scalar("valid/auc", vauc, epoch)
            writer.add_scalar("valid/acc", vacc, epoch)
            writer.add_scalar("test/auc", tauc, epoch)
            writer.add_scalar("test/acc", tacc, epoch)
            writer.add_scalar("test/ll", tll, epoch)
            writer.add_scalar("train/loss", loss, epoch)

        if not args.debug and vauc > best_auc:
            best_auc = vauc
            print('save problem auc:', best_auc)
            torch.save(model.state_dict(), os.path.join(args.run_dir, args.model, summary_name, 'best_model_para.pth'))

        log.info('Epoch: {:03d}, Loss: {:.7f}, \n v_acc: {:.7f}, v_auc: {:.7f}, vll:{:.4f}, \n  t_acc: {:.7f}, t_auc:{:.7f}, t_ll:{:4f}, t_acc_prev:{}, t_acc_next:{}'.format(
                epoch, loss, vacc, vauc, vll, tacc, tauc, tll, t_pn_accs[0], t_pn_accs[1]
            ))
        schedular.step()


def evaluate(model, loader, args):
    model.eval()
    logit_list, label_list = [], []
    logit_list_p, label_list_p = [], []
    logit_list_n, label_list_n = [], []
    BCELoss = torch.nn.BCELoss(reduction='none')
    problem_acc = np.zeros([args.problem_number, 2])
    with torch.no_grad():
        for data in loader:
            len_list, q_seq, lesson_seq, concepts, operate, btype, mask,  c_text, s_text_t, s_text_f = data
            q_seq = q_seq.to(args.device)
            lesson_seq = lesson_seq.to(args.device)
            concepts = concepts.to(args.device)
            operate = operate.to(args.device)
            mask = mask.to(args.device)
            btype = btype.to(args.device)
            c_text = c_text.to(args.device)
            s_text_t = s_text_t.to(args.device)
            s_text_f = s_text_f.to(args.device)
            if args.model == "gkt" or args.model == "akt" or args.model == "skt" or args.model == "lbkt" or args.model == "simple_kt" or args.model == "iekt":
                predict, pre_hiddenstates, logits, extra_loss = model(len_list, q_seq, lesson_seq, concepts, operate, btype, c_text, s_text_t, s_text_f)
            else:
                logits, _ = model(q_seq, lesson_seq, concepts, operate, btype, c_text, s_text_t, s_text_f)
            logit_l, label_l = get_list(len_list, logits, operate, mask)
            logit_l_prev, label_l_prev = get_list(len_list, logits[:, :50], operate[:, :50], mask[:, :50])
            logit_l_next, label_l_next = get_list(len_list, logits[:, 50:], operate[:, 50:], mask[:, 50:])

            logit_list.append(logit_l)
            label_list.append(label_l)
            logit_list_p.append(logit_l_prev)
            logit_list_n.append(logit_l_next)
            label_list_p.append(label_l_prev)
            label_list_n.append(label_l_next)
            problem_acc += get_acc_of_ques(len_list, (logits > 0.5).int(), operate, q_seq, args, mask)

    label_list = torch.cat(label_list, dim=0)
    logit_list = torch.cat(logit_list, dim=0)
    fpr, tpr, thresholds = metrics.roc_curve(label_list.cpu().numpy(), logit_list.cpu().numpy(), pos_label=1)
    auc = metrics.auc(fpr, tpr)
    label_list_p = torch.cat(label_list_p, dim=0)
    logit_list_p = torch.cat(logit_list_p, dim=0)
    label_list_n = torch.cat(label_list_n, dim=0)
    logit_list_n = torch.cat(logit_list_n, dim=0)

    p_acc = accuracy_score(label_list_p.cpu().numpy(), (logit_list_p > 0.5).int().cpu().numpy())
    n_acc = accuracy_score(label_list_n.cpu().numpy(), (logit_list_n > 0.5).int().cpu().numpy())

    y_list = (logit_list > 0.5).int()
    ll = 0
    ll = torch.sum(BCELoss(logit_list.float(), label_list.float())) / label_list.shape[0]
    acc = accuracy_score(label_list.cpu().numpy(), y_list.int().cpu().numpy())
    fpr, tpr, thresholds = metrics.roc_curve(label_list.cpu().numpy(), logit_list.cpu().numpy(), pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return acc, auc, ll, logit_list, problem_acc, [p_acc, n_acc]

