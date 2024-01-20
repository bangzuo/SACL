import setproctitle
import random
import torch
import numpy as np
import os
from time import time
from prettytable import PrettyTable
import datetime
from utils.parser import parse_args
from utils.data_loader import load_data
from modules.SACL import SACL, Discriminator_ui, Discriminator_kg, train_disc, train_gen
from utils.helper import early_stopping, init_logger
from logging import getLogger
from utils.sampler import UniformSampler
from collections import defaultdict
from sklearn.metrics import roc_auc_score, f1_score
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True


seed = 2020
n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0

try:
    from cppimport import imp_from_filepath
    from os.path import join, dirname
    path = join(dirname(__file__), "utils/ext/sampling.cpp")
    sampling = imp_from_filepath(path)
    sampling.seed(seed)
except:
    sampling = UniformSampler(seed)

setproctitle.setproctitle('EXP@SACL')


def _get_topk_feed_data(user, items):
    res = list()
    for item in items:
        res.append([user, item, 0])
    return np.array(res)

def _get_user_record(data, is_train):
    user_history_dict = dict()
    for rating in data:
        user = rating[0]
        item = rating[1]
        label = rating[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict

def get_feed_dict_topk(train_entity_pairs, start, end):
    train_entity_pairs = torch.LongTensor(np.array([[cf[0], cf[1], cf[2]] for cf in train_entity_pairs], np.int32))
    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['items'] = entity_pairs[:, 1]
    feed_dict['labels'] = entity_pairs[:, 2]
    return feed_dict


def topk_eval(model, train_data, data):
    k_list = [5, 10, 20, 50, 100]
    recall_list = {k: [] for k in k_list}
    ndcg_list = {k: [] for k in k_list}
    item_set = set(train_data[:, 1].tolist() + data[:, 1].tolist())
    train_record = _get_user_record(train_data, True)
    test_record = _get_user_record(data, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    user_num = 100
    if len(user_list) > user_num:
        np.random.seed()
        user_list = np.random.choice(user_list, size=user_num, replace=False)

    model.eval()
    entity_gcn_emb, user_gcn_emb = model.generate()
    for user in user_list:
        test_item_list = list(item_set - set(train_record[user]))
        item_score_map = dict()
        start = 0
        while start + args.batch_size <= len(test_item_list):
            items = test_item_list[start:start + args.batch_size]
            input_data = _get_topk_feed_data(user, items)
            batch = get_feed_dict_topk(input_data, start, start + args.batch_size)

            u1 = batch['users']
            i1 = batch['items']
            u_e = user_gcn_emb[u1]
            i_e = entity_gcn_emb[i1]
            scores = (u_e * i_e).sum(dim=1)
            scores = torch.sigmoid(scores)

            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += args.batch_size
        # padding the last incomplete mini-batch if exists
        if start < len(test_item_list):
            res_items = test_item_list[start:] + [test_item_list[-1]] * (args.batch_size - len(test_item_list) + start)
            input_data = _get_topk_feed_data(user, res_items)
            batch = get_feed_dict_topk(input_data, start, start + args.batch_size)

            u2 = batch['users']
            i2 = batch['items']
            u_e = user_gcn_emb[u2]
            i_e = entity_gcn_emb[i2]
            scores = (u_e * i_e).sum(dim=1)
            scores = torch.sigmoid(scores)
            for item, score in zip(res_items, scores):
                item_score_map[item] = score
        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        hits = np.zeros(len(item_sorted))
        index = [i for i, x in enumerate(item_sorted) if x in test_record[user]]
        hits[index] = 1

        for k in k_list:
            hit_k = hits[:k]
            hit_num = np.sum(hit_k)
            recall_list[k].append(hit_num / len(set(test_record[user])))
            dcg = np.sum((2 ** hit_k - 1) / np.log2(np.arange(2, k + 2)))
            sorted_hits_k = np.flip(np.sort(hits))[:k]
            idcg = np.sum((2 ** sorted_hits_k - 1) / np.log2(np.arange(2, k + 2)))
            idcg[idcg == 0] = np.inf
            ndcg_list[k].append(dcg / idcg)
    model.train()

    recall = [np.mean(recall_list[k]) for k in k_list]
    ndcg = [np.mean(ndcg_list[k]) for k in k_list]
    return recall, ndcg

def ctr_eval(model, data):
    auc_list = []
    f1_list = []
    model.eval()
    start = 0
    entity_gcn_emb, user_gcn_emb = model.generate()
    while start < data.shape[0]:

        batch = get_feed_dict(data, start, start + args.test_batch_size)
        user = batch['users']
        item = batch['items']
        labels = data[start:start + args.test_batch_size, 2]

        u_e = user_gcn_emb[user]
        i_e = entity_gcn_emb[item]
        scores = (u_e * i_e).sum(dim=1)
        scores = torch.sigmoid(scores)
        scores = scores.detach().cpu().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        f1 = f1_score(y_true=labels, y_pred=predictions)
        auc_list.append(auc)
        f1_list.append(f1)
        start += args.batch_size
    model.train()
    auc = float(np.mean(auc_list))
    f1 = float(np.mean(f1_list))
    return auc, f1

def get_feed_dict(train_entity_pairs, start, end):
    train_entity_pairs = torch.LongTensor(np.array([[cf[0], cf[1], cf[2]] for cf in train_entity_pairs], np.int32))
    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['items'] = entity_pairs[:, 1]
    feed_dict['labels'] = entity_pairs[:, 2]
    feed_dict['batch_start'] = start
    return feed_dict

if __name__ == '__main__':
    try:
        """fix the random seed"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        """read args"""
        global args, device
        args = parse_args()
        device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

        log_fn = init_logger(args)
        logger = getLogger()
        
        logger.info('PID: %d', os.getpid())
        logger.info(f"DESC: {args.desc}\n")

        """build dataset"""
        train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args)
        test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1], cf[2]] for cf in test_cf], np.int32))
        train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1], cf[2]] for cf in train_cf], np.int32))
        adj_mat_list, norm_mat_list, mean_mat_list = mat_list
        n_users = n_params['n_users']
        n_items = n_params['n_items']
        n_entities = n_params['n_entities']
        n_relations = n_params['n_relations']
        n_nodes = n_params['n_nodes']

        """define model"""
        model = SACL(n_params, args, graph, mean_mat_list[0]).to(device)

        """define optimizer"""
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)#定义优化器
        # ui图 生成对抗训练
        disc_model_ui = Discriminator_ui(args.dim).to(device)
        disc_pseudo_real_ui = Discriminator_ui(args.dim).to(device)
        optimizer_D_ui = torch.optim.RMSprop(disc_model_ui.parameters(), lr=args.lr * 0.1)
        optimizer_D_pseudo_real_ui = torch.optim.RMSprop(disc_pseudo_real_ui.parameters(), lr=args.lr * 0.1)
        # kg图 生成对抗训练
        disc_model_kg = Discriminator_kg(args.dim).to(device)
        disc_pseudo_real_kg = Discriminator_kg(args.dim).to(device)
        optimizer_D_kg = torch.optim.RMSprop(disc_model_kg.parameters(), lr=args.lr * 0.1)
        optimizer_D_pseudo_real_kg = torch.optim.RMSprop(disc_pseudo_real_kg.parameters(), lr=args.lr * 0.1)

        cur_best_auc = 0.
        cur_best_f1 = 0.
        cur_stopping_step = 0
        cur_best_pre_epoch = 0
        should_stop = False
        best_auc, best_f1 = 0., 0.

        logger.info("start training ...")
        trans_k_loss = 0.

        for epoch in range(args.epoch):
            """training CF"""
            """cf data"""
            index = np.arange(len(train_cf))
            np.random.shuffle(index)
            train_cf = train_cf[index]
            """training"""
            model.train()
            add_loss_dict, s = defaultdict(float), 0
            train_s_t = time()
            step = 0
            step_num = (len(train_cf)-1) // args.batch_size + 1
            if args.dataset == "music":
                discD_every_iter = 3
                discG_every_iter = discD_every_iter * 4
            elif args.dataset == "book":
                discD_every_iter = 7
                discG_every_iter = discD_every_iter * 4
            elif args.dataset == "movie":
                discD_every_iter = 16
                discG_every_iter = discD_every_iter * 4
            else:
                discD_every_iter = 4
                discG_every_iter = discD_every_iter * 4
            while s + args.batch_size <= len(train_cf):
                batch = get_feed_dict(train_cf,s, s + args.batch_size)
                if step == 0:
                    batch_loss, batch_loss_dict, _ = model(step, batch)
                else:
                    batch_loss, batch_loss_dict, _ = model(step, batch)
                step += 1
                optimizer.zero_grad(set_to_none=True)
                batch_loss.backward()
                optimizer.step()

                # 生成对抗模块训练
                if (args.no_ad is False) and (step % discD_every_iter == 0):
                    # 直接 full batch 4:1 训练这两部分 1、is_ui=True 训练ui图， 2、is_ui=Flase 训练kg
                    if args.no_uiAug is False:
                        train_disc(model, disc_model_ui, optimizer_D_ui, disc_pseudo_real_ui, optimizer_D_pseudo_real_ui, step, step_num, batch, annealing_type=args.annealing_type, is_ui=True)
                    if args.no_kgAug is False:
                        train_disc(model, disc_model_kg, optimizer_D_kg, disc_pseudo_real_kg, optimizer_D_pseudo_real_kg, step, step_num, batch, annealing_type=args.annealing_type, is_ui=False)
                    if step % discG_every_iter == 0 and step > 0:
                        if args.no_kgAug is False:
                            train_gen(model, optimizer, disc_model_kg, disc_pseudo_real_kg, step, batch, is_ui=True)
                        if args.no_uiAug is False:
                            train_gen(model, optimizer, disc_model_ui, disc_pseudo_real_ui, step, batch, is_ui=False)

                for k, v in batch_loss_dict.items():
                    add_loss_dict[k] += v
                s += args.batch_size
            train_e_t = time()
            if epoch>0:
                """testing"""
                test_s_t = time()
                model.eval()
                test_auc, test_f1 = ctr_eval(model, test_cf_pairs)
                # recall, ndcg = topk_eval(model, train_cf, test_cf)
                test_e_t = time()

                train_res = PrettyTable()

                """ ctr= """
                train_res.field_names = ["Epoch", "training time", "tesing time", "Loss",  "auc", "f1"]
                train_res.add_row(
                    [epoch, train_e_t - train_s_t, test_e_t - test_s_t, list(add_loss_dict.values()), test_auc, test_f1 ]
                )
                logger.info(train_res)
            else:
                logger.info('{}: using time {}, training loss at epoch {}: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), train_e_t - train_s_t, epoch, list(add_loss_dict.values())))

        logger.info('{}: '.format(
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

        logger.info('stopping at %d,best epoch:%d, AUC:%.4f, F1:%.4f' % (epoch, cur_best_pre_epoch, best_auc, best_f1))

    except Exception as e:
        logger.exception(e)




