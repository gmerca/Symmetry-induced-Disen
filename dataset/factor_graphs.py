import os
import random

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx


class FactorGraphs(InMemoryDataset):
    """PyG object for ErdosRenyi graphs dataset with disen properties"""

    def __init__(self, root, num_graphs=30, num_nodes=20, num_feat=64, transform=None, pre_transform=None,
                 device="cpu", task="train"):
        self.num_nodes = num_nodes
        self.num_feat = num_feat
        self.device = device
        self.num_graphs = num_graphs

        super(FactorGraphs, self).__init__(root, transform, pre_transform)
        map_location = {'cuda:%d' % 0: 'cuda:%d' % device} if torch.cuda.is_available() else device
        self.data, self.slices = torch.load(self.processed_paths[0], map_location)

        # self.latents_sizes = np.array([self.prob, self.logvar, self.mean])
        # self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:], np.array([1, ])))

    @property
    def raw_file_names(self):
        return ['graph_node.npy', 'graph.npy']

    @property
    def processed_file_names(self):
        return ['processed.pt']

    def download(self):
        pass

    @staticmethod
    def get_graph_list(num_factors):
        graph_list = []
        # 2, 3 bipartite graph
        g = nx.turan_graph(n=5, r=2)
        graph_list.append(nx.to_numpy_array(g))

        g = nx.house_x_graph()
        graph_list.append(nx.to_numpy_array(g))

        g = nx.balanced_tree(r=3, h=2)
        graph_list.append(nx.to_numpy_array(g))

        g = nx.grid_2d_graph(m=3, n=3)
        graph_list.append(nx.to_numpy_array(g))

        g = nx.hypercube_graph(n=3)
        graph_list.append(nx.to_numpy_array(g))

        g = nx.octahedral_graph()
        graph_list.append(nx.to_numpy_array(g))

        graph_list.append(nx.to_numpy_array(nx.diamond_graph()))
        graph_list.append(nx.to_numpy_array(nx.lollipop_graph(3, 3)))
        graph_list.append(nx.to_numpy_array(nx.icosahedral_graph()))
        graph_list.append(nx.to_numpy_array(nx.barbell_graph(3, 3)))

        return graph_list[:num_factors]

    def gen_union_graph(self, graph_size=15, num_graph=30):
        # if os.path.isfile(self.saved_file):
        #     print(f"load synthetic graph cls data from {self.saved_file}")
        #     with open(self.saved_file, 'rb') as f:
        #         return pickle.load(f)

        graph_list = self.get_graph_list(8)
        samples = []
        data_list = []

        for _ in range(num_graph):
            union_adj = np.zeros((graph_size, graph_size))
            factor_adjs = []
            labels = np.zeros((1, len(graph_list)))

            id_index = list(range(len(graph_list)))
            random.shuffle(id_index)

            for i in range((len(id_index) + 1) // 2):  # get random half adj
                id = id_index[i]
                labels[0, id] = 1

                single_adj = graph_list[id]
                padded_adj = np.zeros((graph_size, graph_size))
                padded_adj[:single_adj.shape[0], :single_adj.shape[0]] = single_adj

                random_index = np.arange(padded_adj.shape[0])
                np.random.shuffle(random_index)
                padded_adj = padded_adj[random_index]
                padded_adj = padded_adj[:, random_index]

                union_adj += padded_adj
                factor_adjs.append((padded_adj, id))

            # g = dgl.DGLGraph()
            g = from_networkx(nx.DiGraph(union_adj))
            # g = dgl.transform.add_self_loop(g)
            x = torch.FloatTensor(union_adj)
            labels = torch.tensor(labels)
            pyg = Data(edge_index=g.edge_index, x=x)
            data_list.append(pyg)
            samples.append((g, labels, factor_adjs))
        # with open(self.saved_file, 'wb') as f:
        #     pickle.dump(samples, f)
        #     print(f"dataset saved to {self.saved_file}")

        data_, slices = self.collate(data_list)
        if self.device == 0 or self.device == "cpu":
            torch.save((data_, slices), self.processed_paths[0])

        return

    def process(self):
        """
        create WS dataset
        """
        data_list = []
        # classes, latents = [], []

        self.gen_union_graph(num_graph=self.num_graphs)
        return

        # num_node = self.num_nodes
        # num_features = self.num_feat
        # prob_range = list(np.linspace(0.05, 0.95, self.prob))
        # logvar_range = list(range(2, self.logvar + 2))
        # mean_range = list(np.linspace(2, 20, self.mean))
        #
        # for ix_a, p in enumerate(prob_range):
        #     for ix_b, lvar in enumerate(logvar_range):
        #         for ix_c, m in enumerate(mean_range):
        #             ws = nx.generators.random_graphs.erdos_renyi_graph(num_node, p)
        #             data = from_networkx(ws)
        #             edge_index = data.edge_index
        #             x = torch.normal(mean=m, std=lvar, size=(num_node, num_features))
        #             pyg = Data(edge_index=edge_index, x=x)
        #             data_list.append(pyg)
        #             # classes.append([ix_a, ix_b, ix_c])
        #             # latents.append([p, lvar, m])
        #
        # # np.save(self.processed_paths[1], np.array(classes))
        # # np.save(self.processed_paths[2], np.array(latents))
        # data_, slices = self.collate(data_list)
        # if self.device == 0:
        #     torch.save((data_, slices), self.processed_paths[0])

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
    ds = FactorGraphs("../data/FactorGraphs/", num_graphs=10, num_nodes=10, num_feat=32)
    # ds.process()
    # latent = ds.sample_latent()
    # graph = ds.get_img_by_latent(latent)
    # # ds.process()
    # i0 = ds.__getitem__(0)
    # i1 = ds.__getitem__(1)
    # g1 = ds.get_img_by_latent([0, 0, 1])
    # print(g1)

