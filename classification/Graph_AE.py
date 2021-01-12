from abc import ABC

from torch.autograd import Variable
from torch.nn import Linear
from torch_geometric.data import Data

from graph_ae.Layer import SGAT
from torch_sparse import spspmm, coalesce
from graph_ae.SAGEConv import SAGEConv
from torch_geometric.nn import TopKPooling
from torch_geometric.utils import sort_edge_index, add_remaining_self_loops, to_networkx
import torch.nn.functional as F
import torch


class Net(torch.nn.Module, ABC):

    def __init__(self, input_size, kernels, depth, rate, shapes, device):
        super(Net, self).__init__()
        size = kernels
        self.device = device
        self.depth = depth
        self.direction = 1
        self.down_list = torch.nn.ModuleList()
        self.up_list = torch.nn.ModuleList()
        self.pool_list = torch.nn.ModuleList()

        shapes = [shapes[0]]*self.depth
        # encoder
        conv = SGAT(size, input_size, shapes[0])
        self.down_list.append(conv)
        for i in range(self.depth - 1):
            pool = TopKPooling(shapes[i], rate[i])
            self.pool_list.append(pool)
            conv = SGAT(size, shapes[i], shapes[i + 1])
            self.down_list.append(conv)
        pool = TopKPooling(shapes[-1], rate[-1])
        self.pool_list.append(pool)

        # decoder
        for i in range(self.depth - 1):
            conv = SAGEConv(shapes[self.depth - i - 1], shapes[self.depth - i - 2])
            self.up_list.append(conv)
        conv = SAGEConv(shapes[0], input_size)
        self.up_list.append(conv)

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        # edge_index, edge_weight = coalesce(edge_index, edge_weight, num_nodes, num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        return edge_index, edge_weight

    def get_hidden_feature(self):
        return self.feat_list

    def encode(self, data):
        x, edge_index, y, batch = data.x, data.edge_index, data.y, data.batch
        self.x = x
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])

        self.edge_list = []
        self.perm_list = []
        self.shape_list = []
        edge_weight = x.new_ones(edge_index.size(1), device=self.device)

        f, e, b = x.to(self.device), edge_index.to(self.device), batch.to(self.device)
        for i in range(self.depth):
            if i < self.depth:
                self.edge_list.append(e)
            f, attn = self.down_list[i](f, e, self.direction)
            self.shape_list.append(f.shape)
            f = F.leaky_relu(f).to(self.device)
            f, e, _, b, perm, _ = self.pool_list[i](f.to(self.device), e, edge_weight, b, attn.to(self.device))
            if i < self.depth - 1:
                e, edge_weight = self.augment_adj(e, edge_weight, f.shape[0])
            self.perm_list.append(perm)
        latent_x, latent_edge = f, e
        return latent_x, latent_edge, b

    def decode(self, f):
        self.feat_list = []
        z = f
        for i in range(self.depth):
            index = self.depth - i - 1
            shape = self.shape_list[index]
            up = torch.zeros(shape, device=self.device)  # .to(self.device)
            p = self.perm_list[index]
            up[p] = z
            z = self.up_list[i](up, self.edge_list[index])
            if i < self.depth - 1:
                z = torch.relu(z)
            self.feat_list.append(z.detach().cpu().numpy())

        # self.edge_list.clear()
        # self.perm_list.clear()
        # self.shape_list.clear()

        loss = torch.nn.MSELoss()(z, self.x.to(self.device))
        loss_dic = {"total": loss, "rec": loss, "kl": torch.tensor(0), "group": torch.tensor(0)}

        return z, loss_dic

    def forward(self, data):
        latent_x, latent_edge, b = self.encode(data)
        z, loss_dic = self.decode(latent_x)
        return z, latent_x, latent_edge, b, loss_dic




