import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.utils import softmax as scatter_softmax
from logging import getLogger

init = nn.init.xavier_uniform_


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return torch.spmm(adj, embeds)

class RelationF_KG(nn.Module):

    def __init__(self, nfeat):
        super(RelationF_KG, self).__init__()
        self.fc1 = nn.Linear(nfeat * 4, nfeat)
        self.fc2 = nn.Linear(nfeat, nfeat)

    def forward(self, x, neighbor, masked_neighbor):
        # 保证 fc12 只干自己的事情，不能把梯度传给输入，且 output 需过滤为仅 user 的部分
        x = x.detach()
        neighbor = neighbor.detach()
        masked_neighbor = masked_neighbor.detach()

        ngb_seq = torch.stack(
            [x, neighbor, neighbor * x, (neighbor + x) / 2.0], dim=1)
        missing_info = self.fc1(ngb_seq.reshape(len(ngb_seq), -1))
        missing_info = F.relu(missing_info)
        missing_info = self.fc2(missing_info)
        support_out = missing_info - masked_neighbor
        return missing_info, support_out

class RelationF_ui(nn.Module):

    def __init__(self, nfeat):
        super(RelationF_ui, self).__init__()
        self.fc1 = nn.Linear(nfeat * 4, nfeat)
        self.fc2 = nn.Linear(nfeat, nfeat)

    def forward(self, x, neighbor, masked_neighbor):
        # 保证 fc12 只干自己的事情，不能把梯度传给输入，且 output 需过滤为仅 user 的部分
        x = x.detach()
        neighbor = neighbor.detach()
        masked_neighbor = masked_neighbor.detach()

        ngb_seq = torch.stack(
            [x, neighbor, neighbor * x, (neighbor + x) / 2.0], dim=1)
        missing_info = self.fc1(ngb_seq.reshape(len(ngb_seq), -1))
        missing_info = F.relu(missing_info)
        missing_info = self.fc2(missing_info)
        support_out = missing_info - masked_neighbor
        return missing_info, support_out

class AttnHGCN(nn.Module):
    """
    Heterogeneous Graph Convolutional Network
    """

    def __init__(self, channel, n_hops, n_users, n_items, n_relations,
                 node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(AttnHGCN, self).__init__()

        self.logger = getLogger()
        self.no_attn_convs = nn.ModuleList()
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        initializer = nn.init.xavier_uniform_
        relation_emb = initializer(torch.empty(n_relations, channel))  # not include interact
        self.relation_emb = nn.Parameter(relation_emb)  # [n_relations - 1, in_channel]
        self.W_Q = nn.Parameter(torch.Tensor(channel, channel))
        self.W_UI = nn.Parameter(torch.Tensor(channel, channel))
        self.n_heads = 2
        self.d_k = channel // self.n_heads

        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_UI)

        self.n_hops = n_hops
        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout
        self.trans_relation_kg = RelationF_KG(64)
        self.trans_relation_ui = RelationF_ui(64)

    def non_attn_agg(self, user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w, relation_emb):

        n_entities = entity_emb.shape[0]

        """KG aggregate"""
        head, tail = edge_index
        edge_relation_emb = relation_emb[
            edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

        """user aggregate"""
        item_agg = inter_edge_w.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_agg, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
        return entity_agg, user_agg

    def shared_layer_agg(self, user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w, relation_emb):
        n_entities = entity_emb.shape[0]
        head, tail = edge_index

        query = entity_emb[head] @ self.W_Q
        key = entity_emb[tail] @ self.W_Q

        query = query * relation_emb[edge_type - 1]
        key = key * relation_emb[edge_type - 1]

        edge_attn_score = (query + key).sum(dim=-1)

        relation_emb = relation_emb[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * relation_emb  # [-1, channel]
        value = neigh_relation_emb

        edge_attn_score = scatter_softmax(edge_attn_score, head)
       
        entity_agg = value * edge_attn_score.view(-1, 1)
        # attn weight makes mean to sum
        entity_agg = scatter_sum(src=entity_agg, index=head, dim_size=n_entities, dim=0)

        quary_u = user_emb[inter_edge[0, :]] @ self.W_UI
        quary_u = quary_u * relation_emb[-1]
        key_i = entity_emb[inter_edge[1, :]] @ self.W_UI
        key_i = key_i * relation_emb[-1]
        edge_attn_score_ui = (quary_u + key_i).sum(dim=-1)
        edge_attn_score_ui = scatter_softmax(edge_attn_score_ui, inter_edge[0, :])

        item_agg = inter_edge_w.unsqueeze(-1) * entity_emb[inter_edge[1, :]] * edge_attn_score_ui.view(-1, 1)

        user_agg = scatter_sum(src=item_agg, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
        return entity_agg, user_agg

    # 生成用于推荐的用户与项目表示
    def forward(self, user_emb, entity_emb, edge_index, edge_type,
                inter_edge, inter_edge_w, mess_dropout=True, item_attn=None):

        if item_attn is not None:  # 第一遍运行，为推荐生成嵌入的时候这个是None
            item_attn = item_attn[inter_edge[1, :]]
            item_attn = scatter_softmax(item_attn, inter_edge[0, :])
            norm = scatter_sum(torch.ones_like(inter_edge[0, :]), inter_edge[0, :], dim=0, dim_size=user_emb.shape[0])
            norm = torch.index_select(norm, 0, inter_edge[0, :])
            item_attn = item_attn * norm
            inter_edge_w = inter_edge_w * item_attn

        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]
        for i in range(self.n_hops):
            entity_emb, user_emb = self.shared_layer_agg(user_emb, entity_emb, edge_index, edge_type, inter_edge,
                                                         inter_edge_w, self.relation_emb)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        return entity_res_emb, user_res_emb

    def forward_ui(self, embeddings, denoise_inter_edge, denoise_inter_edge_w, pseudo_inter_edge, pseudo_inter_edge_w,
            noselect_inter_edge, noselect_inter_edge_w, head_ui, tail_ui, ui_degrees, mess_dropout=True):

        denoise_ui_embs, pseudo_ui_embs = embeddings, embeddings
        denoise_ui_res_embs, pseudo_ui_res_embs = denoise_ui_embs, pseudo_ui_embs
        kt_origin_pseudo_ui = pseudo_ui_embs
        # 用于计算长尾增强图没有进行知识转移得到的嵌入表示
        pseudo_ui_no_aug_emb = pseudo_ui_embs
        pseudo_ui_no_aug_embs = pseudo_ui_no_aug_emb

        kl_calc_loss_ui_embs = {}
        mi_calc_loss_ui_embs = {"missing_loss_ui_embs": []}

        for i in range(self.n_hops):
            denoise_ui_embs = self.ui_agg(embeddings, denoise_inter_edge, denoise_inter_edge_w)
            pseudo_ui_embs = self.ui_agg(pseudo_ui_embs, pseudo_inter_edge, pseudo_inter_edge_w)
            pseudo_ui_no_aug_emb = pseudo_ui_embs.clone()
            noselect_ui_embs = self.ui_agg(pseudo_ui_embs, noselect_inter_edge, noselect_inter_edge_w)

            missing_ui_info, missing_ui_pre_loss_emb = self.trans_relation_ui(kt_origin_pseudo_ui, pseudo_ui_embs, noselect_ui_embs)
            pseudo_ui_embs[:self.n_users] = pseudo_ui_embs[:self.n_users] + missing_ui_info[:self.n_users] / (ui_degrees + 2).reshape(-1, 1)
            kt_origin_pseudo_ui = pseudo_ui_embs

            mi_calc_loss_ui_embs["missing_loss_ui_embs"].append(missing_ui_pre_loss_emb[:self.n_users])

            """message dropout"""
            if mess_dropout:
                denoise_ui_embs = self.dropout(denoise_ui_embs)
                pseudo_ui_embs = self.dropout(pseudo_ui_embs)
                pseudo_ui_no_aug_emb = self.dropout(pseudo_ui_no_aug_emb)
            denoise_ui_embs = F.normalize(denoise_ui_embs)
            pseudo_ui_embs = F.normalize(pseudo_ui_embs)
            pseudo_ui_no_aug_emb = F.normalize(pseudo_ui_no_aug_emb)

            """result emb"""
            denoise_ui_res_embs = torch.add(denoise_ui_res_embs, denoise_ui_embs)
            pseudo_ui_res_embs = torch.add(pseudo_ui_res_embs, pseudo_ui_embs)
            pseudo_ui_no_aug_embs = torch.add(pseudo_ui_no_aug_embs, pseudo_ui_no_aug_emb)
        aug_ui_res_embs = denoise_ui_res_embs.clone()
        aug_ui_res_embs[tail_ui] = pseudo_ui_res_embs[tail_ui]
        kl_calc_loss_ui_embs["denoise_ui_embs"] = denoise_ui_res_embs
        kl_calc_loss_ui_embs["pseudo_ui_embs"] = pseudo_ui_res_embs
        return aug_ui_res_embs, denoise_ui_res_embs, pseudo_ui_res_embs, pseudo_ui_no_aug_embs, kl_calc_loss_ui_embs, mi_calc_loss_ui_embs

    #用于对比学习，利用去噪之后的知识图谱生成项目表示（只生成项目，两张图的项目之间进行对比学习）
    def forward_kg(self, entity_emb, denoise_edge_index, denoise_edge_type, pseudo_edge_index, pseudo_edge_type,
                   noselect_kg_index, noselect_kg_edge_type, head_kg, tail_kg, kg_degrees, mess_dropout=True):

        denoise_entity_emb, pseudo_entity_emb = entity_emb, entity_emb
        denoise_entity_res_emb, pseudo_entity_res_emb = denoise_entity_emb, pseudo_entity_emb
        kt_origin_pseudo_kg = pseudo_entity_emb
        # 用于计算长尾增强图没有进行知识转移时的嵌入表示
        pseudo_kg_no_aug_emb = pseudo_entity_emb
        pseudo_kg_no_aug_embs = pseudo_kg_no_aug_emb

        kl_calc_loss_kg_embs = {}
        mi_calc_loss_kg_embs = {"missing_loss_kg_embs": []}

        for i in range(self.n_hops):
            denoise_entity_emb = self.kg_agg(denoise_entity_emb, denoise_edge_index, denoise_edge_type)
            pseudo_entity_emb = self.kg_agg(pseudo_entity_emb, pseudo_edge_index, pseudo_edge_type)
            pseudo_kg_no_aug_emb = pseudo_entity_emb.clone()
            noselect_entity_emb = self.kg_agg(pseudo_entity_emb, noselect_kg_index, noselect_kg_edge_type)

            missing_kg_info, missing_kg_pre_loss_emb = self.trans_relation_kg(kt_origin_pseudo_kg, pseudo_entity_emb, noselect_entity_emb)
            pseudo_entity_emb = pseudo_entity_emb + missing_kg_info / (kg_degrees + 4).reshape(-1, 1)
            kt_origin_pseudo_kg = pseudo_entity_emb

            mi_calc_loss_kg_embs["missing_loss_kg_embs"].append(missing_kg_pre_loss_emb)
            """message dropout"""
            if mess_dropout:
                denoise_entity_emb = self.dropout(denoise_entity_emb)
                pseudo_entity_emb = self.dropout(pseudo_entity_emb)
                pseudo_kg_no_aug_emb = self.dropout(pseudo_kg_no_aug_emb)
            denoise_entity_emb = F.normalize(denoise_entity_emb)
            pseudo_entity_res_emb = F.normalize(pseudo_entity_res_emb)
            pseudo_kg_no_aug_emb = F.normalize(pseudo_kg_no_aug_emb)

            """result emb"""
            denoise_entity_res_emb = torch.add(denoise_entity_res_emb, denoise_entity_emb)
            pseudo_entity_res_emb = torch.add(pseudo_entity_res_emb, pseudo_entity_emb)
            pseudo_kg_no_aug_embs = torch.add(pseudo_kg_no_aug_embs, pseudo_kg_no_aug_emb)
        aug_entity_res_emb = denoise_entity_res_emb.clone()
        aug_entity_res_emb[tail_kg] = pseudo_entity_res_emb[tail_kg]
        kl_calc_loss_kg_embs["denoise_kg_emb"] = denoise_entity_res_emb
        kl_calc_loss_kg_embs["pseudo_kg_emb"] = pseudo_entity_res_emb
        return aug_entity_res_emb, denoise_entity_res_emb, pseudo_entity_res_emb, pseudo_kg_no_aug_embs, kl_calc_loss_kg_embs, mi_calc_loss_kg_embs

    def ui_agg(self, embeddings, inter_edge, inter_edge_w):
        user_emb, item_emb = embeddings[:self.n_users, :], embeddings[self.n_users:, :]
        num_items = item_emb.shape[0]
        item_emb = inter_edge_w.unsqueeze(-1) * item_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_emb, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
        user_emb = inter_edge_w.unsqueeze(-1) * user_emb[inter_edge[0, :]]
        item_agg = scatter_sum(src=user_emb, index=inter_edge[1, :], dim_size=num_items, dim=0)
        agg_embeddings = torch.cat((user_agg, item_agg), dim=0)
        return agg_embeddings

    def kg_agg(self, entity_emb, edge_index, edge_type):
        n_entities = entity_emb.shape[0]
        head, tail = edge_index
        edge_relation_emb = self.relation_emb[edge_type - 1]
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]

        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

        return entity_agg

    def _edge_mask_adapt_mixed(self, edge_index, edge_type, topk_egde_id):

        n_edges = edge_index.shape[1]
        topk_egde_id = topk_egde_id.cpu().numpy()
        topk_mask = np.zeros(n_edges, dtype=bool)
        topk_mask[topk_egde_id] = True
        random_indices = np.random.choice(
            n_edges, size=topk_egde_id.shape[0], replace=False)
        random_mask = np.zeros(n_edges, dtype=bool)
        random_mask[random_indices] = True
        mask = topk_mask | random_mask
        remain_edge_index = edge_index[:, ~mask]
        remain_edge_type = edge_type[~mask]
        masked_edge_index = edge_index[:, mask]
        masked_edge_type = edge_type[mask]

        return remain_edge_index, remain_edge_type, masked_edge_index, masked_edge_type, mask

    @torch.no_grad()
    def norm_attn_computer(self, entity_emb, edge_index, n_entities, aug_kg_rate, mae_rate=0.1, edge_type=None, print_=False, return_logits=False, no_viewGen=False):
        keep_rate = 1 - mae_rate
        head_keep_rate = aug_kg_rate
        tail_keep_rate = aug_kg_rate / (1-aug_kg_rate)

        head, tail = edge_index
        query = (entity_emb[head] @ self.W_Q)
        key = (entity_emb[tail] @ self.W_Q)

        if edge_type is not None:
            query = query * self.relation_emb[edge_type - 1]
            key = key * self.relation_emb[edge_type - 1]

        edge_attn_logits = (query + key).sum(-1).detach()
        edge_attn_score = scatter_softmax(edge_attn_logits, head)
        #normalization by head_node degree
        norm = scatter_sum(torch.ones_like(head), head, dim=0, dim_size=entity_emb.shape[0])
        norm = torch.index_select(norm, 0, head)
        nedge_attn_score = edge_attn_score * norm
        noise = -torch.log(-torch.log(torch.rand_like(nedge_attn_score)))
        nedge_attn_score = nedge_attn_score + noise

        least_topk_v, least_topk_attn_edge_id = torch.topk(
            nedge_attn_score, int(keep_rate * len(nedge_attn_score)), sorted=False)
        denoise_edge_index, denoise_edge_type = edge_index[:,least_topk_attn_edge_id], edge_type[least_topk_attn_edge_id]
        head, tail = denoise_edge_index
        edge_attn_logits = edge_attn_logits[head]

        # 以下至 end_80_ind = torch.where(torch.sum(head.unsqueeze(1) == end_80, dim=1))[0] 一行为计算头部节点（度在top20的节点）在图中的下标
        # 合并两个张量列表
        if return_logits is False:
            entities = head
            # 使用torch.bincount()函数计算每个值的度
            entity_degrees = torch.bincount(entities, minlength=n_entities)

            top_20 = int(len(torch.unique(entities)) * head_keep_rate)
            end_80 = int(len(entity_degrees) - top_20)

            top_20_degree, top_20 = torch.topk(entity_degrees, top_20)
            end_80_degree, end_80 = torch.topk(-1 * entity_degrees, end_80)
            head_kg = top_20
            tail_kg = end_80
            
            top_20_ind = torch.where(torch.isin(head, top_20))[0]
            end_80_ind = torch.where(torch.isin(head, end_80))[0]
            scores_top20 = edge_attn_logits[top_20_ind]
            """ prob based drop """
            softmax_values_top20 = scatter_softmax(scores_top20, head[top_20_ind])
            select_num = int(len(end_80_ind) * tail_keep_rate)
            
            if no_viewGen:
                softmax_values_top20 = torch.ones_like(softmax_values_top20)
            
            pseudo_head_ind = torch.multinomial(softmax_values_top20, select_num, replacement=False)
            pseudo_head_ind = top_20_ind[pseudo_head_ind]
            # 使用布尔索引获取 A 中剩余的值
            noselect_head_ind = torch.where(~torch.isin(top_20_ind, pseudo_head_ind))[0]

            pseudo_ind = torch.cat([pseudo_head_ind, end_80_ind], dim=-1)
            pseudo_edge_type = edge_type[pseudo_ind]
            pseudo_head = head[pseudo_ind]
            pseudo_tail = tail[pseudo_ind]
            pseudo_index = torch.stack([pseudo_head, pseudo_tail], dim=0)
            pseudo_head, _ = pseudo_index
            pseudo_degrees = torch.bincount(pseudo_head, minlength=n_entities)

            noselect_kg_index = torch.stack([head[noselect_head_ind], tail[noselect_head_ind]], dim=0)
            noselect_kg_edge_type = edge_type[noselect_head_ind]

            return edge_attn_score, pseudo_index, pseudo_edge_type, head_kg, tail_kg, pseudo_degrees, noselect_kg_index, noselect_kg_edge_type, denoise_edge_index, denoise_edge_type
        else:
            if print_:
                self.logger.info("edge_attn_score std: {}".format(edge_attn_score.std()))
            return edge_attn_score, edge_attn_logits
            
    @torch.no_grad() # SACL的forward中调用传入参数为True
    def process_ui_graph(self, inter_edge, inter_edge_w, user_emb, item_emb, mae_rate, aug_ui_rate, is_return_neigh_emb=False, no_viewGen=False):
        n_node = self.n_users+self.n_items
        # 去噪图
        p_kg = (1-mae_rate)
        embs_u = user_emb[inter_edge[0]]
        quary_u = embs_u @ self.W_UI
        quary_u = quary_u * self.relation_emb[-1] 
        embs_i = item_emb[inter_edge[1]]
        key_i = embs_i @ self.W_UI
        key_i = key_i * self.relation_emb[-1] 
        scores_ui = (quary_u + key_i).sum(-1)
        softmax_values_scores_ui = scatter_softmax(scores_ui, inter_edge[0])
        # normalization by head_node degree
        norm = scatter_sum(torch.ones_like(softmax_values_scores_ui), inter_edge[0], dim=0, dim_size=user_emb.shape[0])
        norm = torch.index_select(norm, 0, inter_edge[0])
        softmax_values_scores_ui = softmax_values_scores_ui * norm
        noise = -torch.log(-torch.log(torch.rand_like(softmax_values_scores_ui)))
        softmax_values_scores_ui = softmax_values_scores_ui + noise

        _, select_indeices = torch.topk(softmax_values_scores_ui, int(len(softmax_values_scores_ui) * p_kg), sorted=False)
        denoise_inter_edge, denoise_inter_edge_w = inter_edge[:,select_indeices], inter_edge_w[select_indeices]/p_kg

        # 长尾增强图
        denoiseAdj = torch.sparse.FloatTensor(torch.stack([denoise_inter_edge[0], denoise_inter_edge[1]]), torch.ones_like(denoise_inter_edge[0]).cuda(),
                                             (n_node, n_node)).coalesce()
        degrees = torch.sparse.sum(denoiseAdj, dim=1).to_dense().view(-1)[:self.n_users]  # 只获取用户的度
        top_20 = int(len(degrees) * aug_ui_rate)
        end_80 = int(len(degrees) - top_20)
        top_20_degrees, top_20 = torch.topk(degrees, top_20)
        end_80_degrees, end_80 = torch.topk(-1 * degrees, end_80)
        head_ui, head_ui_degree = top_20, top_20_degrees
        tail_ui, tail_ui_degree = end_80, end_80_degrees

        # 使用高级索引提取满足条件的元素的索引和值
        top_20_ind = torch.where(torch.isin(denoise_inter_edge[0], top_20.unsqueeze(1)))[0]
        end_80_ind = torch.where(torch.isin(denoise_inter_edge[0], end_80.unsqueeze(1)))[0]

        top_20_ind, top_20_val = denoise_inter_edge[:, top_20_ind], denoise_inter_edge_w[top_20_ind]
        end_80_ind, end_80_val = denoise_inter_edge[:, end_80_ind], denoise_inter_edge_w[end_80_ind]

        embs_u_ = user_emb[top_20_ind[0]]
        quary_u_ = embs_u_ @ self.W_UI
        quary_u_ = quary_u_ * self.relation_emb[-1] 
        embs_i_ = item_emb[top_20_ind[1]]
        key_i_ = embs_i_ @ self.W_UI
        key_i_ = key_i_ * self.relation_emb[-1] 
        values_top20 = (quary_u_ + key_i_).sum(-1)
        
        softmax_values_top20 = scatter_softmax(values_top20, top_20_ind[0])
        end_keep_rate = aug_ui_rate / (1-aug_ui_rate)
        need_select_num = int(len(end_80_ind[0]) * end_keep_rate)
        pseudo_keep_rate = need_select_num / len(top_20_val)
        
        if no_viewGen:
            softmax_values_top20 = torch.ones_like(softmax_values_top20)
        
        pseudo_head_ind = torch.multinomial(softmax_values_top20, need_select_num, replacement=False)
        pseudo_head_indeices, pseudo_head_vals = top_20_ind[:,pseudo_head_ind], top_20_val[pseudo_head_ind] / pseudo_keep_rate

        pseudo_inter_edge = torch.cat((pseudo_head_indeices, end_80_ind), dim=1)
        pseudo_inter_edge_w = torch.cat([pseudo_head_vals, end_80_val])
        pseudoAdj = torch.sparse.FloatTensor(torch.stack([pseudo_inter_edge[0], pseudo_inter_edge[1]]), torch.ones_like(pseudo_inter_edge[0]).cuda(),
                                             (n_node, n_node)).coalesce()
        pseudo_degrees = torch.sparse.sum(pseudoAdj, dim=1).to_dense().view(-1)[:self.n_users]  # 只获取用户的度

        # 找到 A 中的值在 B 中不存在的值的坐标
        top_20_i = torch.arange(len(top_20_ind[0])).cuda()
        noselect_pseudo_ind = torch.where(~torch.isin(top_20_i, pseudo_head_ind))[0]
        noselect_inter_edge, noselect_inter_edge_w = top_20_ind[:,noselect_pseudo_ind], top_20_val[noselect_pseudo_ind] / (1 - pseudo_keep_rate)

        return denoise_inter_edge, denoise_inter_edge_w, pseudo_inter_edge, pseudo_inter_edge_w, noselect_inter_edge, \
               noselect_inter_edge_w, head_ui, tail_ui, pseudo_degrees
