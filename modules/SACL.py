import random
import numpy as np
import torch
import torch.nn as nn
from .AttnHGCN import AttnHGCN
from .contrast import Contrast
from logging import getLogger
import torch.nn.functional as F
from scipy import sparse as sp


def _relation_aware_edge_sampling(edge_index, edge_type, n_relations, samp_rate=0.5):

    for i in range(n_relations - 1):
        edge_index_i, edge_type_i = _edge_sampling(
            edge_index[:, edge_type == i], edge_type[edge_type == i], samp_rate)
        if i == 0:
            edge_index_sampled = edge_index_i
            edge_type_sampled = edge_type_i
        else:
            edge_index_sampled = torch.cat(
                [edge_index_sampled, edge_index_i], dim=1)
            edge_type_sampled = torch.cat(
                [edge_type_sampled, edge_type_i], dim=0)

    return edge_index_sampled, edge_type_sampled


def _edge_sampling(edge_index, edge_type, samp_rate=0.5):
    n_edges = edge_index.shape[1]

    random_indices = np.random.choice(
        n_edges, size=int(n_edges * samp_rate), replace=False)#replace为false代表不允许重复采样

    return edge_index[:, random_indices], edge_type[random_indices]


def _sparse_dropout(i, v, keep_rate=0.5):
    noise_shape = i.shape[1]
    random_tensor = keep_rate
    # the drop rate is 1 - keep_rate
    random_tensor += torch.rand(noise_shape).to(i.device)
    dropout_mask = torch.floor(random_tensor).type(torch.bool)

    i = i[:, dropout_mask]
    v = v[dropout_mask] / keep_rate

    return i, v

class Discriminator_ui(nn.Module):

    def __init__(self, in_features):
        super(Discriminator_ui, self).__init__()

        self.d_ui = nn.Linear(in_features, in_features, bias=True)
        self.wd_ui = nn.Linear(in_features, 1, bias=False)
        self.sigmoid_ui = nn.Sigmoid()

    def forward(self, ft_ui):
        ft_ui = F.elu(ft_ui)
        ft_ui = F.dropout(ft_ui, 0.5, training=self.training)
        fc_ui = F.elu(self.d_ui(ft_ui))
        prob_ui = self.wd_ui(fc_ui)
        return prob_ui


class Discriminator_kg(nn.Module):

    def __init__(self, in_features):
        super(Discriminator_kg, self).__init__()

        self.d_kg = nn.Linear(in_features, in_features, bias=True)
        self.wd_kg = nn.Linear(in_features, 1, bias=False)
        self.sigmoid_kg = nn.Sigmoid()

    def forward(self, ft_kg):

        ft_kg = F.elu(ft_kg)
        ft_kg = F.dropout(ft_kg, 0.5, training=self.training)
        ft_kg = F.elu(self.d_kg(ft_kg))
        prob_kg = self.wd_kg(ft_kg)
        return prob_kg


class SACL(nn.Module):

    def __init__(self, data_config, args_config, graph, adj_mat, hp_dict=None):
        super(SACL, self).__init__()
        self.args_config = args_config
        self.logger = getLogger()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.lr = args_config.lr
        self.l2 = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")
        
        self.ablation = args_config.ab

        self.mae_rate = args_config.mae_rate
        self.cross_cl_reg = args_config.cross_cl_reg
        self.cross_cl_tau = args_config.cross_cl_tau

        self.kg_dropout = args_config.kg_dropout
        self.ui_dropout = args_config.ui_dropout

        # 知识端局部对比学习参数
        self.kg_cl_reg = args_config.kg_cl_reg
        self.kg_cl_tau = args_config.kg_cl_tau

        self.dataset = args_config.dataset

        self.samp_func = "torch"

        """ 针对不同数据集的超参数设置 """
        if args_config.dataset == 'music':
            # 消融实验
            self.no_uiAug = args_config.no_uiAug
            self.no_kgAug = args_config.no_kgAug
            self.no_cross_cl = args_config.no_cross_cl
            self.no_viewGen = args_config.no_viewGen
            self.no_denoise = args_config.no_denoise
            
            # 跨视图（ui图和kg）对比学习超参（第四个要调的）
            self.cross_cl_reg = 0.01        # 跨视图对比学习权重
            self.cross_cl_tau = 0.1         # 跨视图对比学习温度系数

            # 学习率和l2正则化权重（第一个要调的）
            self.l2 = 1e-4                  # l2正则化系数

            # GNN层数（第二个要调的）
            self.context_hops = 2

            # 去噪图去噪比例（第三个要调的）
            self.mae_rate = 0.1

            # 知识转移模块权重（第五个要调的）
            self.kl_eps = 0.1
            self.mi_eps = 0.1
            
            # 长尾增强时 head部节点 比率
            self.aug_ui_rate = 0.8		# ui图
            self.aug_kg_rate = 0.2             # kg图
 
        elif args_config.dataset == 'book':
            # 跨视图（ui图和kg）对比学习超参
            self.cross_cl_reg = 0.01
            self.cross_cl_tau = 0.1

            # l2正则化权重
            self.l2 = 1e-4
            
            # 消融实验
            self.no_uiAug = args_config.no_uiAug
            self.no_kgAug = args_config.no_kgAug
            self.no_cross_cl = args_config.no_cross_cl
            self.no_viewGen = args_config.no_viewGen
            self.no_denoise = args_config.no_denoise

            # GNN层数
            self.context_hops = 1

            # 去噪图去噪比例
            self.mae_rate = 0.1

            # 知识转移模块权重
            self.kl_eps = 0.1    # 去噪图及长尾增强图学习到的嵌入表示的KL分布优化权重
            self.mi_eps = 0.1       # 知识转移模块为尾实体增强的表示与原表示的误差权重
            
            # 长尾增强时 head部节点 比率
            self.aug_ui_rate = 0.2		# ui图
            self.aug_kg_rate = 0.8              # kg图

        elif args_config.dataset == 'movie':
            # 消融实验
            self.no_uiAug = args_config.no_uiAug
            self.no_kgAug = args_config.no_kgAug
            self.no_cross_cl = args_config.no_cross_cl
            self.no_viewGen = args_config.no_viewGen
            self.no_denoise = args_config.no_denoise
        
            # 跨视图（ui图和kg）对比学习超参
            self.cross_cl_reg = 0.01
            self.cross_cl_tau = 0.9
            
            # GNN层数
            self.context_hops = 2
            
            # 去噪图去噪比例
            self.mae_rate = 0.1

            # l2正则化权重
            self.l2 = 1e-4

            # 知识转移模块权重
            self.kl_eps = 0.1
            self.mi_eps = 0.1
            
            # 长尾增强时 head部节点 比率
            self.aug_ui_rate = 0.8		# ui图
            self.aug_kg_rate = 0.2              # kg图

        if hp_dict is not None:
            for k,v in hp_dict.items():
                setattr(self, k, v)

        self.inter_edge, self.inter_edge_w = self._convert_sp_mat_to_tensor(
            adj_mat)
        self.edge_index, self.edge_type = self._get_edges(graph)
        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)

        self.gcn = AttnHGCN(channel=self.emb_size,
                       n_hops=self.context_hops,
                       n_users=self.n_users,
                        n_items = self.n_items,
                        n_relations=self.n_relations,
                        node_dropout_rate=self.node_dropout_rate,
                        mess_dropout_rate=self.mess_dropout_rate)
        self.contrast_fn = Contrast(self.emb_size, tau=self.kg_cl_tau)

    def forward(self, step, batch=None):

        user = batch['users']
        item = batch['items']
        labels = batch['labels']
        user_emb = self.all_embed[:self.n_users, :]
        entity_emb = self.all_embed[self.n_users:, :]
        item_emb = entity_emb[:self.n_items]
        
        # 去噪的消融
        if self.no_denoise:
            self.mae_rate = 0
        
        # 交互图 dropout 操作
        if self.ui_dropout:
            inter_edge, inter_edge_w = _sparse_dropout(
                self.inter_edge, self.inter_edge_w, self.node_dropout_rate)
        else:
            inter_edge, inter_edge_w = self.inter_edge, self.inter_edge_w

        denoise_inter_edge, denoise_inter_edge_w, pseudo_inter_edge, pseudo_inter_edge_w, noselect_inter_edge, noselect_inter_edge_w, head_ui, tail_ui, ui_degrees \
            = self.gcn.process_ui_graph(inter_edge, inter_edge_w, user_emb, item_emb, self.mae_rate, self.aug_ui_rate, is_return_neigh_emb=True, no_viewGen=self.no_viewGen)

        # 知识图谱 dropout 操作
        if self.kg_dropout:
            edge_index, edge_type = _relation_aware_edge_sampling(
                self.edge_index, self.edge_type, self.n_relations, self.node_dropout_rate)
        else:
            edge_index, edge_type = self.edge_index, self.edge_type

        edge_attn_score, pseudo_edge_index, pseudo_edge_type, head_kg, tail_kg, kg_degrees, \
        noselect_kg_index, noselect_kg_edge_type, denoise_edge_index, denoise_edge_type = self.gcn.norm_attn_computer(entity_emb,
        edge_index, self.n_entities, self.aug_kg_rate, self.mae_rate, edge_type, print_=False, return_logits=False, no_viewGen=self.no_viewGen)


        aug_entity_res_emb, denoise_entity_res_emb, pseudo_entity_res_emb, pseudo_kg_no_aug_embs, kl_calc_loss_kg_embs, mi_calc_loss_kg_embs \
            = self.gcn.forward_kg(entity_emb, denoise_edge_index, denoise_edge_type, pseudo_edge_index, pseudo_edge_type,
                                  noselect_kg_index, noselect_kg_edge_type, head_kg, tail_kg, kg_degrees)

        embeddings = torch.cat((user_emb, entity_emb), dim=0)
        aug_ui_res_embs, denoise_ui_res_embs, pseudo_ui_res_embs, pseudo_ui_no_aug_embs, kl_calc_loss_ui_embs, mi_calc_loss_ui_embs = self.gcn.forward_ui(
            embeddings, denoise_inter_edge, denoise_inter_edge_w, pseudo_inter_edge, pseudo_inter_edge_w,
            noselect_inter_edge, noselect_inter_edge_w, head_ui, tail_ui, ui_degrees)
            
        if self.no_uiAug:
            aug_ui_res_embs = denoise_ui_res_embs
        if self.no_kgAug:
            aug_entity_res_emb = denoise_entity_res_emb
        aug_user_emb, aug_item_emb = aug_ui_res_embs[:self.n_users], aug_ui_res_embs[self.n_users:]

        """ 生成对抗网络优化使用的嵌入集  1、aug_embs 就是原始数据的嵌入表示 2、pseudo_embs 就是增强尾节点（对头部节点进行drop）得到的视图进行增强的嵌入表示 """
        embs_dict = {"ui_res_embs":denoise_ui_res_embs, "pseudo_ui_res_embs":pseudo_ui_res_embs,
                     "entity_res_emb":denoise_entity_res_emb, "pseudoentity_entity_res_emb":pseudo_entity_res_emb,
                     "pseudo_kg_no_aug_embs":pseudo_kg_no_aug_embs, "pseudo_ui_no_aug_embs":pseudo_kg_no_aug_embs,
                     "head_ui":head_ui, "tail_ui":tail_ui, "head_kg":head_kg, "tail_kg":tail_kg}

        """ 去噪图学习到的嵌入和长尾增强图学习到的嵌入的 KL loss """
        kl_calc_loss_ui_emb_1 = kl_calc_loss_ui_embs["pseudo_ui_embs"][:self.n_users][head_ui]
        kl_calc_loss_ui_emb_2 = kl_calc_loss_ui_embs["denoise_ui_embs"][:self.n_users][head_ui]
        kl_calc_loss_entity_emb_1 = kl_calc_loss_kg_embs["pseudo_kg_emb"][head_kg]
        kl_calc_loss_entity_emb_2 = kl_calc_loss_kg_embs["denoise_kg_emb"][head_kg]
        
        loss_kl_corr_user, loss_kl_corr_entity = torch.tensor(0.0), torch.tensor(0.0)
        # ui KL
        if self.no_uiAug is False:
            loss_kl_corr_user = torch.nn.functional.kl_div(
                kl_calc_loss_ui_emb_1.log_softmax(dim=-1),
                kl_calc_loss_ui_emb_2.softmax(dim=-1),
                reduction='batchmean',
                log_target=False)
                
        # kg KL
        if self.no_kgAug is False:
            loss_kl_corr_entity = torch.nn.functional.kl_div(
                kl_calc_loss_entity_emb_1.log_softmax(dim=-1),
                kl_calc_loss_entity_emb_2.softmax(dim=-1),
                reduction='batchmean', log_target=False)
                    
        kl_loss = loss_kl_corr_user * self.kl_eps + loss_kl_corr_entity * self.kl_eps
 
        """ 计算 Knowledge Trans Module 中预测邻域缺失信息的 Loss """
        missing_loss = torch.tensor(0.0)
        if self.no_uiAug is False:
            #（1） ui
            missing_loss_user_embs = mi_calc_loss_ui_embs["missing_loss_ui_embs"]
            mask_head = head_ui       
            for embs in missing_loss_user_embs:
                m_regularizer_ui = torch.mean(torch.norm(embs[mask_head], dim=1))
                missing_loss = missing_loss + m_regularizer_ui
                
        if self.no_kgAug is False:
            #（2） KG
            missing_loss_entity_embs = mi_calc_loss_kg_embs["missing_loss_kg_embs"]
            mask_head = head_kg       
            for embs in missing_loss_entity_embs:
                m_regularizer_kg = torch.mean(torch.norm(embs[mask_head], dim=1))
                missing_loss = missing_loss + m_regularizer_kg
            missing_loss = missing_loss * self.mi_eps
        """ ckg 视图编码  依旧是使用 denoise图 进行编码 """
        entity_gcn_emb, user_gcn_emb = self.gcn(aug_user_emb,
                                                aug_entity_res_emb,
                                                denoise_edge_index,
                                                denoise_edge_type,
                                                denoise_inter_edge,
                                                denoise_inter_edge_w,
                                                mess_dropout=self.mess_dropout)
        
        cross_cl_loss = torch.tensor(0.0)
        if self.no_cross_cl is False:
            cross_cl_loss = self.cross_cl_reg * self.contrast_fn(aug_entity_res_emb, aug_item_emb)

        # 最终的实体嵌入和用户嵌入是全局级别的 user-item-entity 图编码得到的嵌入和 局部级别的交互图和知识图谱编码得到的嵌入的拼接
        entity_gcn_emb = torch.cat([entity_gcn_emb, aug_item_emb], dim=1)
        user_gcn_emb = torch.cat([user_gcn_emb, aug_user_emb], dim=1)

        u_e = user_gcn_emb[user]
        pos_e = entity_gcn_emb[item]

        loss = torch.tensor(0.0)
        bpr_loss, rec_loss, reg_loss,scores = self.create_bpr_loss(u_e, pos_e, labels)
        loss = loss + bpr_loss              # 主任务，交叉熵
        loss = loss + kl_loss               # 训练期间使用长尾增强图（denoise 图基础上的长尾增强）编码得到的分布，和原始图（denoise 图）的分布的KL损失
        loss = loss + missing_loss          # 长尾增强图的预测损失，计算方式是最小化长尾增强图中编码得到的 head 节点嵌入表示 与原denoise图内 head节点的嵌入表示
        loss = loss + cross_cl_loss         # 交互图和知识图谱跨视图对比损失，目的是对齐协作信号和kg信号

        loss_dict = {
            "total_loss": loss.item(),
            "bpr_loss": bpr_loss.item(),
            "kl_loss": kl_loss.item(),
            "missing_loss": missing_loss.item(),
            "cross_cl_loss": cross_cl_loss.item(),
        }

        return loss, loss_dict, embs_dict

    def _make_torch_adj_one(self,inter_edge):
        values = torch.ones(inter_edge.shape[1])
        mat = sp.coo_matrix((values.cpu().numpy(), (inter_edge[0].cpu().numpy(), inter_edge[1].cpu().numpy())), \
                            shape=(self.n_users,self.n_items))
        a = sp.csr_matrix((self.n_users, self.n_users)) 
        b = sp.csr_matrix((self.n_items, self.n_items))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])]) 
        mat = (mat != 0) * 1.0 
        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        mat = mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)

        return  torch.sparse.FloatTensor(idxs, vals, shape).cuda()
    
    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))

    def _convert_sp_mat_to_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return i.to(self.device), v.to(self.device)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))

        #提取出前两列，是边的位置信息
        index = graph_tensor[:, :-1]  # [-1, 2]
        #提取出最后一列，是边的类型信息
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def center_embs(self, denoiseAdj, embeds):
        order = torch.sparse.sum(denoiseAdj, dim=-1).to_dense().view([-1, 1])
        fstEmbeds = torch.spmm(denoiseAdj, embeds) - embeds
        fstNum = order  # fstNum是一阶邻居节点数量
        scdEmbeds = (torch.spmm(denoiseAdj, fstEmbeds) - fstEmbeds) - order * embeds
        scdNum = (torch.spmm(denoiseAdj, fstNum) - fstNum) - order
        subgraphEmbeds = (fstEmbeds + scdEmbeds) / (fstNum + scdNum + 1e-8)

        subgraphEmbeds = F.normalize(subgraphEmbeds, p=2)
        centerEmbeds = F.normalize(embeds, p=2)
        return centerEmbeds, subgraphEmbeds

    def generate(self,):
        user_emb = self.all_embed[:self.n_users, :]
        entity_emb = self.all_embed[self.n_users:, :]
        item_emb = entity_emb[:self.n_items]

        denoise_inter_edge, denoise_inter_edge_w, pseudo_inter_edge, pseudo_inter_edge_w, noselect_inter_edge, noselect_inter_edge_w, head_ui, tail_ui, ui_degrees \
            = self.gcn.process_ui_graph(self.inter_edge, self.inter_edge_w, user_emb, item_emb, self.mae_rate, self.aug_ui_rate, is_return_neigh_emb=True)
        
               
        edge_attn_score, pseudo_edge_index, pseudo_edge_type, head_kg, tail_kg, kg_degrees, \
        noselect_kg_index, noselect_kg_edge_type, denoise_edge_index, denoise_edge_type = self.gcn.norm_attn_computer(entity_emb,
        self.edge_index, self.n_entities, self.aug_kg_rate, self.mae_rate, self.edge_type)
        
        aug_entity_res_emb, denoise_entity_res_emb, _, _, _, _ \
            = self.gcn.forward_kg(entity_emb, denoise_edge_index, denoise_edge_type, pseudo_edge_index, pseudo_edge_type,
                                  noselect_kg_index, noselect_kg_edge_type, head_kg, tail_kg, kg_degrees)

        embeddings = torch.cat((user_emb, entity_emb), dim=0)
        aug_ui_res_embs, denoise_ui_res_embs, _, _, _, _ = self.gcn.forward_ui(
            embeddings, denoise_inter_edge, denoise_inter_edge_w, pseudo_inter_edge, pseudo_inter_edge_w,
            noselect_inter_edge, noselect_inter_edge_w, head_ui, tail_ui, ui_degrees)
        if self.no_uiAug:
            aug_ui_res_embs = denoise_ui_res_embs
        if self.no_kgAug:
            aug_entity_res_emb = denoise_entity_res_emb
                
        aug_user_emb, aug_item_emb = aug_ui_res_embs[:self.n_users], aug_ui_res_embs[self.n_users:]

        entity_gcn_emb, user_gcn_emb = self.gcn(aug_user_emb,
                                                aug_entity_res_emb,
                                                denoise_edge_index,
                                                denoise_edge_type,
                                                denoise_inter_edge,
                                                denoise_inter_edge_w,
                                                mess_dropout=self.mess_dropout)

        entity_gcn_emb = torch.cat([entity_gcn_emb, aug_item_emb], dim=1)
        user_gcn_emb = torch.cat([user_gcn_emb, aug_user_emb], dim=1)
        return entity_gcn_emb, user_gcn_emb

    def create_bpr_loss(self, users, items,labels):
        batch_size = users.shape[0]
        scores = (items * users).sum(dim=1)
        scores = torch.sigmoid(scores)
        criteria = nn.BCELoss()
        bce_loss = criteria(scores, labels.float())
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(items) ** 2 )/ 2
        emb_loss = self.l2 * regularizer / batch_size

        return bce_loss + emb_loss, bce_loss, emb_loss,scores

def train_disc(embed_model, disc_model, optimizer_D,
               disc_pseudo_real, optimizer_D_pseudo_real, step, step_num,  batch, annealing_type=1, is_ui=True):
    disc_model.train()
    disc_pseudo_real.train()
    embed_model.train()

    for p in disc_model.parameters():
        p.requires_grad = True  # to avoid computation
    for p in disc_pseudo_real.parameters():
        p.requires_grad = True
    with torch.no_grad():
        _, _, embs_dict = embed_model(step, batch)
        # u_emb: u-i图编码得到的， i_emb: KG编码得到的
        u_emb_h, u_emb_t, u_emb_nt = embs_dict['ui_res_embs'], embs_dict['pseudo_ui_res_embs'], embs_dict['pseudo_ui_no_aug_embs']
        e_emb_h, e_emb_t, e_emb_nt = embs_dict['entity_res_emb'], embs_dict['pseudoentity_entity_res_emb'], embs_dict['pseudo_kg_no_aug_embs']
        head_ui, tail_ui, head_kg, tail_kg = embs_dict['head_ui'], embs_dict['tail_ui'], embs_dict['head_kg'], embs_dict['tail_kg']

    # 头部点的范围是否缩小到仅 user 节点中
    # 一些可调节的超参，暂时放在这里
    noise_eps = 0.1
    lambda_gp = 1.0
    if is_ui:
        all_head_emb_h = u_emb_h[head_ui]
        all_emb_t = u_emb_t
        all_emb_nt = u_emb_nt
        # cell_mask = head_ui
        head, tail = head_ui, tail_ui
    else:
        all_head_emb_h = e_emb_h[head_ui]
        all_emb_t = e_emb_t
        all_emb_nt = e_emb_nt

        head, tail = head_kg, tail_kg

    if True:
        if annealing_type == 0:
            # 不做退火
            noise_eps = noise_eps
        elif annealing_type == 1:
            # 均匀直线下降
            annealing_stop_rate = 0.7  # 超参数，可调节
            rate = 1 - min(step /
                           (annealing_stop_rate * step_num), 1.0)
            noise_eps = noise_eps * rate
        elif annealing_type == 2:
            # 先快后慢，凹函数下降
            annealing_stop_rate = 0.6  # 超参数，可调节
            annealing_rate = (0.01 ** (1 / step_num / annealing_stop_rate))
            noise_eps = noise_eps * (annealing_rate ** step)
        else:
            # 不加噪声
            noise_eps = 0.0

        def exec_perturbed(x, noise_eps):
            random_noise = torch.rand_like(x)
            x = x + torch.sign(x) * F.normalize(random_noise,
                                                dim=-1) * noise_eps
            return x

        # 对 real、fake 数据加噪声
        all_head_emb_h = exec_perturbed(all_head_emb_h, noise_eps=noise_eps)
        all_emb_t = exec_perturbed(all_emb_t, noise_eps=noise_eps)

    prob_h = disc_model(all_head_emb_h)
    prob_t = disc_model(all_emb_t)

    errorD = -prob_h.mean()
    errorG = prob_t.mean()

    def get_select_idx(max_value, select_num, strategy='uniform'):
        """
        max_value: 在 [0, max_value) 之间采样
        select_num: 采样数目
        strategy: 采样策略，uniform 均极度均匀，random 为随机采样
        """
        select_idx = None
        if strategy == 'uniform':
            select_idx = torch.randperm(max_value).repeat(
                int(np.ceil(select_num / max_value)))[:select_num]
        elif strategy == 'random':
            select_idx = np.random.randint(0, max_value, select_num)
        return select_idx

    def calc_gradient_penalty(netD, real_data, fake_data, lambda_gp):
        alpha = torch.rand(real_data.shape[0], 1).to("cuda")
        alpha = alpha.expand(real_data.size())

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates.requires_grad_(True)

        disc_interpolates = netD(interpolates)

        import torch.autograd as autograd
        gradients = autograd.grad(outputs=disc_interpolates,
                                  inputs=interpolates,
                                  grad_outputs=torch.ones(
                                      disc_interpolates.size(),
                                      device="cuda"),
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp

        return gradient_penalty

    # disc 1
    gp_fake_data = all_emb_t
    gp_real_data = all_head_emb_h[get_select_idx(len(all_head_emb_h),
                                                 len(gp_fake_data),
                                                 strategy='random')]
    gradient_penalty = calc_gradient_penalty(netD=disc_model,
                                             real_data=gp_real_data,
                                             fake_data=gp_fake_data,
                                             lambda_gp=lambda_gp)
    L_d = errorD + errorG + gradient_penalty

    optimizer_D.zero_grad()
    L_d.backward()
    optimizer_D.step()

    # disc 2
    pseudo_embs = all_emb_nt[head]
    real_tail_embs = all_emb_nt[tail]
    if len(pseudo_embs) > len(real_tail_embs):
        gp_fake_data = pseudo_embs
        gp_real_data = real_tail_embs[get_select_idx(len(real_tail_embs),
                                                     len(gp_fake_data),
                                                     strategy='random')]
    else:
        gp_real_data = real_tail_embs
        gp_fake_data = pseudo_embs[get_select_idx(len(pseudo_embs),
                                                  len(gp_real_data),
                                                  strategy='random')]
    L_gp2 = calc_gradient_penalty(netD=disc_pseudo_real,
                                  real_data=gp_real_data,
                                  fake_data=gp_fake_data,
                                  lambda_gp=lambda_gp)

    prob_t_with_miss = disc_pseudo_real(all_emb_nt)
    errorR_pseudo = prob_t_with_miss[head].mean()
    errorR_real_tail = -prob_t_with_miss[tail].mean()
    L_d2 = errorR_pseudo + errorR_real_tail + L_gp2

    optimizer_D_pseudo_real.zero_grad()
    L_d2.backward()
    optimizer_D_pseudo_real.step()

    log = {
        'loss/disc1_errorD': errorD.item(),
        'loss/disc1_errorG': errorG.item(),
        'loss/disc1_errorG_real': prob_t[tail].mean().item(),
        'loss/disc1_errorG_pseudo': prob_t[head].mean().item(),
        'loss/disc1_gp': gradient_penalty.item(),
        'loss/disc1_full': L_d.item(),
        'loss/disc2_full': L_d2.item(),
        'loss/disc2_gp': L_gp2.item(),
        'loss/disc2_errorR_pseudo': errorR_pseudo.item(),
        'loss/disc2_errorR_real_tail': errorR_real_tail.item(),
        'noise_eps': noise_eps,
    }
    # if os.environ.get('use_wandb'):
    #     wandb.log(log)
    # print(log)
    return L_d

def train_gen(embed_model, optimizer, disc_model, disc_pseudo_real, step, batch, is_ui=True):
    embed_model.train()
    disc_model.train()
    disc_pseudo_real.train()
    for p in disc_model.parameters():
        p.requires_grad = False  # to avoid computation
    for p in disc_pseudo_real.parameters():
        p.requires_grad = False  # to avoid computation

    _, _, embs_dict = embed_model(step, batch)
    # u_emb: u-i图编码得到的， i_emb: KG编码得到的
    u_emb_h, u_emb_t, u_emb_nt = embs_dict['ui_res_embs'], embs_dict['pseudo_ui_res_embs'], embs_dict[
        'pseudo_ui_no_aug_embs']
    e_emb_h, e_emb_t, e_emb_nt = embs_dict['entity_res_emb'], embs_dict['pseudoentity_entity_res_emb'], embs_dict[
        'pseudo_kg_no_aug_embs']
    head_ui, tail_ui, head_kg, tail_kg = embs_dict['head_ui'], embs_dict['tail_ui'], embs_dict['head_kg'], embs_dict[
        'tail_kg']

    if is_ui:
        emb_t = u_emb_t
        head = head_ui
    else:
        emb_t = e_emb_t
        head = head_kg
    # disc 1
    all_emb_t = emb_t

    prob_t = disc_model(all_emb_t)
    L_disc1 = -prob_t.mean() * 0.1  # 超参数，可调节

    # disc 2
    all_emb_nt = u_emb_nt[head]

    prob_t_with_miss = disc_pseudo_real(all_emb_nt)

    L_disc2 = -prob_t_with_miss.mean() * 0.1

    L_d = L_disc1 + L_disc2

    optimizer.zero_grad()
    L_d.backward()
    optimizer.step()

    log = {
        'loss/discG_full': L_d.item(),
        'loss/discG_1': L_disc1.item(),
        'loss/discG_2': L_disc2.item(),
    }
    # if os.environ.get('use_wandb'):
    #     wandb.log(log)
    return L_d



