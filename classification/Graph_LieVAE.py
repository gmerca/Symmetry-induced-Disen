import math
from abc import ABC

import numpy as np
from torch.nn import Linear

from graph_ae.Layer import SGAT, MultiGCNConv
from torch_sparse import spspmm, coalesce
from graph_ae.SAGEConv import SAGEConv
from torch_geometric.nn import TopKPooling, GCNConv, Sequential, avg_pool_neighbor_x
from torch_geometric.utils import sort_edge_index, add_remaining_self_loops
import torch.nn.functional as F
import torch
from torch import nn
from torch.autograd import Variable


class Net(torch.nn.Module, ABC):
    """
    subgroup_sizes_ls : [100]
    args.subspace_sizes_ls: [10] (assert latent_dim == sum(self.subspace_sizes_ls))
    """
    def __init__(self, input_size, kernels, depth, rate, shapes, device, batch_size, group, subspace, hes, cond, beta, cap, loss, hy_rec, hy_commute):
        super(Net, self).__init__()
        self.group_feat_G = None
        self.group_feats_E = None
        size = kernels
        self.device = device
        self.depth = depth
        self.direction = 1
        self.batch = batch_size

        self.loss = loss

        self.down_list = torch.nn.ModuleList()
        self.up_list = torch.nn.ModuleList()
        self.pool_list = torch.nn.ModuleList()

        # VAE
        self.beta = beta
        self.capacity = cap
        self.capacity_leadin = 1e5
        self.global_step = 0

        # Group loss
        self.hy_hes = hes
        self.hy_rec = hy_rec
        self.hy_commute = hy_commute

        # conditional disentanglement
        self.cond_disen = cond

        # graph enc 64 ->64->64->100
        # to_z  100 -> 10 - replace with vae
        # train_exp 10 -> 10 x 10
        # graph dec 100->64->64 -> 64

        # group encoder
        self.group_size = group
        self.sub_space_size = subspace

        self.depth = depth
        self.depth += 2
        shapes = [shapes[0]] * self.depth
        shapes = [shapes[0]] * (self.depth - 2) + [self.group_size, self.sub_space_size]
        rate = [rate[0]] * self.depth

        # self.to_logvar = Sequential('x, edge_index', [
        #     (GCNConv(self.group_size, self.group_size * 4), 'x, edge_index -> x'), nn.LeakyReLU(True),
        #     (GCNConv(self.group_size * 4, self.group_size * 4), 'x, edge_index -> x'), nn.LeakyReLU(True),
        #     nn.Linear(self.group_size * 4, self.sub_space_size),
        # ])
        # self.to_mean = Sequential('x, edge_index', [
        #     (GCNConv(self.group_size, self.group_size * 4), 'x, edge_index -> x'), nn.LeakyReLU(True),
        #     (GCNConv(self.group_size * 4, self.group_size * 4), 'x, edge_index -> x'), nn.LeakyReLU(True),
        #     nn.Linear(self.group_size * 4, self.sub_space_size),
        # ])

        # if self.cond_disen:
        #     self.z2t_conv = SGAT(size, self.group_size, self.sub_space_size)
            # self.z2t_pool = TopKPooling(self.sub_space_size, rate[0])
            # self.z2t = Sequential('x, edge_index', [
            #     (GCNConv(self.group_size, self.group_size * 4), 'x, edge_index -> x'), nn.LeakyReLU(True),
            #     (GCNConv(self.group_size * 4, self.group_size * 4), 'x, edge_index -> x'), nn.LeakyReLU(True),
            #     nn.Linear(self.group_size * 4, self.sub_space_size),
            # ])

        self.to_mean2 = SGAT(size, self.sub_space_size, self.sub_space_size)
        self.to_logvar2 = SGAT(size, self.sub_space_size, self.sub_space_size)

        self.gcn_t_i = torch.nn.ModuleList()

        for i in range(self.sub_space_size):
            gcn = SAGEConv(self.group_size, self.sub_space_size)
            self.gcn_t_i.append(gcn)

        # encoder
        conv = SGAT(size, input_size, shapes[0])
        self.down_list.append(conv)
        for i in range(self.depth - 1):
            pool = TopKPooling(shapes[i], rate[i])
            self.pool_list.append(pool)
            if i < self.depth - 2:
                conv = SGAT(size, shapes[i], shapes[i + 1])
                self.down_list.append(conv)
        pool = TopKPooling(shapes[-1], rate[-1])
        self.pool_list.append(pool)

        # decoder
        # shapes.insert(0, shapes[0])
        # shapes = shapes[:-1]
        shapes[-1] = self.group_size
        for i in range(self.depth - 1):
            conv = SAGEConv(shapes[self.depth - i - 1], shapes[self.depth - i - 2])
            self.up_list.append(conv)
        conv = SAGEConv(shapes[0], input_size)
        self.up_list.append(conv)

        self.mlp1 = Linear(self.sub_space_size ** 2, self.sub_space_size)
        self.mlp2 = Linear(self.group_size*2, self.group_size)

        # Lie algebra
        self.lie_alg_basis_ls = nn.ParameterList([])
        self.mat_dim = int(math.sqrt(self.group_size))
        for _ in range(self.sub_space_size):
            lie_alg_tmp, _ = self.init_alg_basis(mat_dim=self.mat_dim, lie_alg_init_scale=0.001)
            self.lie_alg_basis_ls.append(lie_alg_tmp)

    def get_hidden_feature(self):
        return self.feat_list

    def forward(self, data):
        self.feat_list = []
        x, edge_index, y, batch = data.x, data.edge_index, data.y, data.batch
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])

        self.edge_list = []
        self.perm_list = []
        self.shape_list = []
        edge_weight = x.new_ones(edge_index.size(1), device=self.device)
        means, logvars = None, None

        f, e, b = x.to(self.device), edge_index.to(self.device), batch.to(self.device)
        for i in range(self.depth):
            self.edge_list.append(e)
            if i == self.depth - 1:
                t_i = [gcn(f, e) for gcn in self.gcn_t_i]
                feats = torch.stack(t_i).view(-1, self.sub_space_size**2)
                agg1 = self.mlp1(feats).squeeze()
                agg_t_i = F.leaky_relu(agg1).to(self.device)
                means, attn1 = self.to_mean2(agg_t_i, e, self.direction)
                logvars, attn2 = self.to_logvar2(agg_t_i, e, self.direction)
                means, logvars = F.leaky_relu(means), F.leaky_relu(logvars)
                t, attn_ = self.reparametrise(means, logvars).to(self.device), attn1 + attn2
                t = F.leaky_relu(t).to(self.device)
                lie_subgroup = self.train_exp(t, self.lie_alg_basis_ls, self.mat_dim) if self.training else self.val_exp(t, self.lie_alg_basis_ls)
                lie = lie_subgroup.view(-1, self.group_size).to(self.device)
                if self.cond_disen:
                    lie = F.leaky_relu(self.mlp2(torch.stack([f, lie]).view(lie.shape[0], self.group_size*2))).to(self.device)
                self.shape_list.append(lie.shape)
                self.group_feat_G = lie
                f, _, _, _, perm, _ = self.pool_list[i](lie, e, edge_weight, b, (attn_.to(self.device)))
                self.perm_list.append(perm)
            else:
                f, attn = self.down_list[i](f, e, self.direction)
                self.shape_list.append(f.shape)
                f = F.leaky_relu(f).to(self.device)
                f, e, _, b, perm, _ = self.pool_list[i](f, e, edge_weight, b, attn.to(self.device))
                if i < self.depth - 1:
                    e, edge_weight = self.augment_adj(e, edge_weight, f.shape[0])
                self.perm_list.append(perm)
                if i == self.depth - 2:
                    self.group_feats_E = f  # 900x49
                    latent_x, latent_edge, latent_b = f, e, b

        z = self.decode(f)

        # self.edge_list.clear()
        # self.perm_list.clear()
        # self.shape_list.clear()

        group_loss = self.group_loss(self.group_feats_E, self.group_feat_G, self.lie_alg_basis_ls)
        if self.loss == "bce":
            rec_loss = self.bce(z, data.x.to(self.device))
        elif self.loss == "l2":
            rec_loss = (z.sigmoid() - data.x.to(self.device)).pow(2).sum() / data.x.to(self.device).shape[0]
        else:
            rec_loss = torch.nn.MSELoss()(z, data.x.to(self.device))
        total_kl, _, _ = self.compute_kl(means, logvars)
        beta_kl = self.control_capacity(total_kl, self.global_step)
        loss = group_loss + rec_loss + beta_kl
        loss_dic = {"total": loss, "rec": rec_loss, "group": group_loss, "kl": beta_kl}

        self.global_step += 1
        return z, latent_x, latent_edge, b, loss_dic

    def decode(self, f):
        z = f
        for i in range(self.depth):
            index = self.depth - i - 1
            shape = self.shape_list[index]
            up = torch.zeros(shape, device=self.device)
            p = self.perm_list[index]
            up[p] = z
            z = self.up_list[i](up, self.edge_list[index])
            if i < self.depth - 1:
                z = torch.relu(z)
            self.feat_list.append(z.detach().cpu().numpy())
        return z

    def bce(self, z, x):
        return F.binary_cross_entropy_with_logits(z.view(z.size(0), -1), x.view(x.size(0), -1), reduction='sum') / x.shape[0]

    def control_capacity(self, total_kl, global_step, anneal=1):
        if self.capacity is not None:
            leadin = 1e5 if self.capacity_leadin is None else self.capacity_leadin
            delta = torch.tensor((self.capacity / leadin) * global_step).clamp(max=self.capacity)
            return (total_kl - delta).abs().clamp(min=0) * self.beta * (anneal ** global_step)
        else:
            return total_kl*self.beta

    @staticmethod
    def compute_kl(mu, logvar, mean=False):
        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if mean:
            reduce = lambda x: torch.mean(x, 1)
        else:
            reduce = lambda x: torch.sum(x, 1)

        total_kld = reduce(klds).mean(0, True)
        dimension_wise_kld = klds.mean(0)
        mean_kld = reduce(klds).mean(0, True)
        return total_kld, dimension_wise_kld, mean_kld

    def reparametrise(self, mu, lv):
        """ VAE Reparameterization trick """
        if self.training:
            std = lv.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
            # std = torch.exp(0.5 * lv)
            # eps = torch.randn_like(std)
            # return mu + std * eps
        else:
            return mu

    @staticmethod
    def train_exp(x, lie_alg_basis_ls, mat_dim):
        """maps 10 -> 10 x 10"""
        lie_alg_basis_ls = [p * 1. for p in lie_alg_basis_ls]  # For torch.cat, convert param to tensor.
        lie_alg_basis = torch.cat(lie_alg_basis_ls, dim=0)[np.newaxis, ...]  # [1, lat_dim, mat_dim, mat_dim]
        lie_group = torch.eye(mat_dim, dtype=x.dtype).to(x.device)[np.newaxis, ...]  # [1, mat_dim, mat_dim]
        lie_alg = 0.
        latents_in_cut_ls = [x]
        for masked_latent in latents_in_cut_ls:
            lie_alg_sum_tmp = torch.sum(masked_latent[..., np.newaxis, np.newaxis] * lie_alg_basis,dim=1)
            lie_alg += lie_alg_sum_tmp  # [b, mat_dim, mat_dim]
            lie_group_tmp = torch.matrix_exp(lie_alg_sum_tmp)
            lie_group = torch.matmul(lie_group, lie_group_tmp)  # [b, mat_dim, mat_dim]
        return lie_group

    @staticmethod
    def val_exp(x, lie_alg_basis_ls):
        """exp map for eval settings"""
        lie_alg_basis_ls = [p * 1. for p in lie_alg_basis_ls]  # For torch.cat, convert param to tensor.
        lie_alg_basis = torch.cat(lie_alg_basis_ls, dim=0)[np.newaxis, ...]  # [1, lat_dim, mat_dim, mat_dim]
        lie_alg_mul = x[..., np.newaxis, np.newaxis] * lie_alg_basis  # [b, lat_dim, mat_dim, mat_dim]
        lie_alg = torch.sum(lie_alg_mul, dim=1)  # [b, mat_dim, mat_dim]
        lie_group = torch.matrix_exp(lie_alg)  # [b, mat_dim, mat_dim]
        return lie_group

    def group_loss(self, group_feats_E, group_feats_G, lie_alg_basis_ls):
        """compute group loss"""
        b_idx = 0
        hessian_loss = 0.
        commute_loss = 0.
        subgroup_sizes_ls = [self.group_size]
        subspace_sizes_ls = [self.sub_space_size]

        for i, subspace_size in enumerate(subspace_sizes_ls):
            e_idx = b_idx + subspace_size
            if subspace_size > 1:
                mat_dim = int(math.sqrt(subgroup_sizes_ls[i]))
                assert list(lie_alg_basis_ls[b_idx].size())[-1] == mat_dim
                lie_alg_basis_mul_ij = self.calc_basis_mul_ij(lie_alg_basis_ls[b_idx:e_idx])  # XY
                hessian_loss += self.calc_hessian_loss(lie_alg_basis_mul_ij, i)
                commute_loss += self.calc_commute_loss(lie_alg_basis_mul_ij, i)
            b_idx = e_idx
        rec_loss = torch.mean(
            torch.sum(torch.square(group_feats_E - group_feats_G), dim=1))
        rec_loss *= self.hy_rec
        hessian_loss *= self.hy_hes
        commute_loss *= self.hy_commute
        loss = hessian_loss + commute_loss + rec_loss
        return loss

    @staticmethod
    def calc_hessian_loss(lie_alg_basis_mul_ij, i):
        hessian_loss = torch.mean(
            torch.sum(torch.square(lie_alg_basis_mul_ij), dim=[2, 3]))
        return hessian_loss

    @staticmethod
    def calc_commute_loss(lie_alg_basis_mul_ij, i):
        lie_alg_commutator = lie_alg_basis_mul_ij - lie_alg_basis_mul_ij.permute(0, 1, 3, 2)
        commute_loss = torch.mean(torch.sum(torch.square(lie_alg_commutator), dim=[2, 3]))
        return commute_loss

    @staticmethod
    def calc_basis_mul_ij(lie_alg_basis_ls_param):
        lie_alg_basis_ls = [alg_tmp * 1. for alg_tmp in lie_alg_basis_ls_param]
        lie_alg_basis = torch.cat(lie_alg_basis_ls, dim=0)[np.newaxis, ...]  # [1, lat_dim, mat_dim, mat_dim]
        _, lat_dim, mat_dim, _ = list(lie_alg_basis.size())
        lie_alg_basis_col = lie_alg_basis.view(lat_dim, 1, mat_dim, mat_dim)
        lie_alg_basis_outer_mul = torch.matmul(
            lie_alg_basis,
            lie_alg_basis_col)  # [lat_dim, lat_dim, mat_dim, mat_dim]
        hessian_mask = 1. - torch.eye(
            lat_dim, dtype=lie_alg_basis_outer_mul.dtype
        )[:, :, np.newaxis, np.newaxis].to(lie_alg_basis_outer_mul.device)
        lie_alg_basis_mul_ij = lie_alg_basis_outer_mul * hessian_mask  # XY
        return lie_alg_basis_mul_ij

    @staticmethod
    def init_alg_basis(mat_dim, lie_alg_init_scale):
        """init lie alg basis"""
        lie_alg_tmp = nn.Parameter(torch.normal(mean=torch.zeros(1, mat_dim, mat_dim), std=lie_alg_init_scale),
                                   requires_grad=True)
        var_tmp = nn.Parameter(torch.normal(torch.zeros(1, 1), lie_alg_init_scale))
        return lie_alg_tmp, var_tmp

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        # edge_index, edge_weight = coalesce(edge_index, edge_weight, num_nodes, num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        return edge_index, edge_weight

