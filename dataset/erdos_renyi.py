import os

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx


class ErdosRenyi(InMemoryDataset):
    """PyG object for ErdosRenyi graphs dataset with disen properties"""

    def __init__(self, root, prob=50, logvar=10, mean=50, num_nodes=20, num_feat=64, transform=None, pre_transform=None,
                 device="cpu", task="train"):
        self.prob = prob
        self.logvar = logvar
        self.mean = mean
        self.num_nodes = num_nodes
        self.num_feat = num_feat
        self.device = device

        super(ErdosRenyi, self).__init__(root, transform, pre_transform)
        map_location = {'cuda:%d' % 0: 'cuda:%d' % device} if torch.cuda.is_available() else device
        self.data, self.slices = torch.load(self.processed_paths[0], map_location)

        self.latents_sizes = np.array([self.prob, self.logvar, self.mean])
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:], np.array([1, ])))

    @property
    def raw_file_names(self):
        return ['graph_node.npy', 'graph.npy']

    @property
    def processed_file_names(self):
        return ['processed.pt']

    def download(self):
        pass

    def process(self):
        """
        create WS dataset
        """
        data_list = []
        # classes, latents = [], []
        num_node = self.num_nodes
        num_features = self.num_feat
        prob_range = list(np.linspace(0.05, 0.95, self.prob))
        logvar_range = list(range(2, self.logvar + 2))
        mean_range = list(np.linspace(2, 20, self.mean))

        for ix_a, p in enumerate(prob_range):
            for ix_b, lvar in enumerate(logvar_range):
                for ix_c, m in enumerate(mean_range):
                    ws = nx.generators.random_graphs.erdos_renyi_graph(num_node, p)
                    data = from_networkx(ws)
                    edge_index = data.edge_index
                    x = torch.normal(mean=m, std=lvar, size=(num_node, num_features))
                    pyg = Data(edge_index=edge_index, x=x)
                    data_list.append(pyg)
                    # classes.append([ix_a, ix_b, ix_c])
                    # latents.append([p, lvar, m])

        # np.save(self.processed_paths[1], np.array(classes))
        # np.save(self.processed_paths[2], np.array(latents))
        data_, slices = self.collate(data_list)
        if self.device == 0 or self.device == 'cpu':
            torch.save((data_, slices), self.processed_paths[0])

    def get_img_by_latent(self, latent_code):
        """
        Returns the image defined by the latent code

        Args:
            latent_code (:obj:`list` of :obj:`int`): Latent code of length 6 defining each generative factor
        Returns:
            Image defined by given code
        """
        idx = self.latent_to_index(latent_code)
        res = self.__getitem__(idx).to(self.device)
        return res

    def latent_to_index(self, latents):
        """maps vector of latent factor into index"""
        return np.dot(latents, self.latents_bases).astype(int)

    def sample_latent(self):
        """creates a latent vector by sampling a value for each latent factor"""
        f = []
        for factor in self.latents_sizes:
            f.append(np.random.randint(0, factor - 1))
        return np.array(f)


if __name__ == '__main__':
    ds = ErdosRenyi("../data/ErdosRenyi/", 10, 10, 10)
    # ds.process()
    latent = ds.sample_latent()
    graph = ds.get_img_by_latent(latent)
    # ds.process()
    i0 = ds.__getitem__(0)
    i1 = ds.__getitem__(1)
    g1 = ds.get_img_by_latent([0, 0, 1])
    print(g1)

