"""
Alchemy data for Disen
based on alchemy repo
"""
import json
import os.path
import os.path as osp
import random

import numpy as np
from numpy.random import choice
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import networkx as nx
import pathlib
import pandas as pd


class TencentAlchemyDataset(InMemoryDataset):
    """PyG dataset for Alchemy"""
    fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

    def __init__(self,
                 root,
                 mode='dev',
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.mode = mode

        self.a_type_ls, self.aromatic_ls, self.hyb_ls, self.num_h_ls, self.n_atoms_ = [], [], [], [], []

        # prob obtained as : [round(el/94968*100,2) for el in df.aromatic_ls.value_counts().to_list()]
        # num_atoms = choice(self.df2.natom.unique(), p=[el/3951 for el in self.df2.natom.value_counts().to_list()])
        atoms = ["H", "C", "O", "N", "S", "F", "Cl"]
        self.atom_dic = dict(zip(atoms, [str(el) for el in range(len(atoms))]))
        self.atom_probs = [0.532, 0.362, 0.051, 0.0394, 0.0151, 0.0004, 0.0001]

        hyb = ["S", "SP3", "SP2", "SP"]
        self.hyb_dic = dict(zip(hyb, [str(el) for el in range(len(hyb))]))
        self.hyb_probs = [0.53, 0.27, 0.17, 0.03]

        arom = ["F", "T"]
        self.arom_dic = dict(zip(arom, [str(el) for el in range(len(arom))]))
        self.arom_probs = [0.9, 0.1]

        hyn = [0]
        self.hyn_dic = dict(zip(hyn, [el for el in range(len(hyn))]))

        super(TencentAlchemyDataset, self).__init__(root, transform,
                                                    pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        with open(self.processed_paths[1], "r") as f:
            self.ix2lat = json.load(f)
        with open(self.processed_paths[2], "r") as f:
            self.lat2ix = json.load(f)
        with open(self.processed_paths[5], "r") as f:
            self.ix2gix = json.load(f)
        self.df = pd.read_csv(self.processed_paths[3])
        self.df2 = pd.read_csv(self.processed_paths[4])

        self.latents_sizes = np.array([7, 2, 4, 4])


    @property
    def raw_file_names(self):
        subset = "valid" if torch.cuda.is_available() else "valid"
        return [f"/{subset}/sdf", os.path.join(self.root, f"/{subset}/valid_target.csv")]

    @property
    def processed_file_names(self):
        return ['TencentAlchemy_%s.pt' % self.mode, 'ix2lat.json', 'lat2ix.json', 'atoms.csv', "n_atoms.csv", "ix2gix.json"]

    def download(self):
        pass
        # raise NotImplementedError('please download and unzip dataset from %s, and put it at %s' % (_urls[self.mode], self.raw_dir))

    def alchemy_nodes(self, g):
        feat = []
        for n, d in g.nodes(data=True):
            h_t = []
            # Atom type (One-hot H, C, N, O F)
            h_t += [
                int(d['a_type'] == x)
                for x in ['H', 'C', 'N', 'O', 'F', 'S', 'Cl']
            ]
            # Atomic number
            h_t.append(d['a_num'])
            # Acceptor
            h_t.append(d['acceptor'])
            # Donor
            h_t.append(d['donor'])
            # Aromatic
            h_t.append(int(d['aromatic']))
            # Hybradization
            h_t += [int(d['hybridization'] == x) \
                    for x in (Chem.rdchem.HybridizationType.SP, \
                        Chem.rdchem.HybridizationType.SP2,
                        Chem.rdchem.HybridizationType.SP3)]
            h_t.append(d['num_h'])
            feat.append((n, h_t))
        feat.sort(key=lambda item: item[0])
        node_attr = torch.FloatTensor([item[1] for item in feat])
        return node_attr

    def alchemy_edges(self, g):
        e = {}
        for n1, n2, d in g.edges(data=True):
            e_t = [int(d['b_type'] == x)
                    for x in (Chem.rdchem.BondType.SINGLE, \
                            Chem.rdchem.BondType.DOUBLE, \
                            Chem.rdchem.BondType.TRIPLE, \
                            Chem.rdchem.BondType.AROMATIC)]
            e[(n1, n2)] = e_t

        edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
        edge_attr = torch.FloatTensor(list(e.values()))
        return edge_index, edge_attr

    def get_img_by_latent(self, latent_code):
        """
        Returns the graph defined by the latent code
        """
        idx = self.latent_to_index(latent_code)
        res = self.__getitem__(idx)
        return res

    def latent_to_index(self, latents):
        """maps vector of latent factor into index"""
        latstr = "".join(latents)
        ix = self.lat2ix[latstr]
        gix = self.ix2gix[str(ix)]
        return gix
        # return np.dot(latents, self.latents_bases).astype(int)

    def sample_latent(self):
        """
        check if sampled code is acutal molecule
        repeat until valid molecule is found
        should return np array of latents
        """
        # code = self.sample_codes()
        ix = random.randrange(len(self.ix2lat))
        code = list(self.ix2lat.values())[ix]
        mol = list(code[:4 * 25])
        bonds = ["e" + s for s in code[4 * 25:].split("e")[1:]]
        full = mol + bonds
        return full

    def sample_codes_(self, to_avoid):
        """sample codes excluding those already not working"""
        type_ = choice(list(self.atom_dic.values()), p=self.atom_probs)
        aromatic = choice(list(self.arom_dic.values()), p=self.arom_probs)
        hyb = choice(list(self.hyb_dic.values()), p=self.hyb_probs)
        hyn = list(self.hyb_dic.values())[0]
        st_i = "".join([type_, aromatic, hyb, hyn])
        if st_i not in to_avoid:
            return st_i
        else:
            return self.sample_codes_(to_avoid)

    def update_subset(self, subset, i, code, c=0):
        """ select molecules matching the sampled code """
        output_subset = []
        for k in subset:
            st_part = k[i * 4: i * 4 + 4]
            if code == st_part:
                c += 1
                output_subset.append(k)
        if c == 0:
            return subset, c
        else:
            return output_subset, c

    def sample_codes(self):
        """
        return a string like OF30 x n of atoms (e.g. 20)
        bonds: ex1_x2
        """
        st, i, c, to_avoid = "", 0, 0, []
        subset = list(self.lat2ix.keys())
        while c != 1:
            st_i = self.sample_codes_(to_avoid)
            subset, c = self.update_subset(subset, i, st_i)
            if c > 1:
                st += st_i
                i += 1
                to_avoid = []
            if c == 0:
                to_avoid = st_i
        return st

    def compute_gf_string(self, atom_i, i):
        """concat atom properties in a string"""
        a_type = self.atom_dic[atom_i.GetSymbol()]
        hyb = self.hyb_dic[str(atom_i.GetHybridization())]
        aromatic = self.arom_dic[str(atom_i.GetIsAromatic())[0]]
        num_h = self.hyn_dic[atom_i.GetTotalNumHs()]

        els = [a_type, aromatic, hyb, num_h]
        lss = [self.a_type_ls, self.aromatic_ls, self.hyb_ls, self.num_h_ls]
        res = ""
        for el, ls in zip(els, lss):
            el = str(el)
            ls.append(el)
            res += el
        return res
        # return f"atm{i}" + "".join(res)

    def sdf_graph_reader(self, sdf_file):
        """sdf file reader for Alchemy dataset"""
        gf_string = ""

        with open(sdf_file, 'r') as f:
            sdf_string = f.read()
        mol = Chem.MolFromMolBlock(sdf_string, removeHs=False)
        if mol is None:
            print("rdkit can not parsing", sdf_file)
            return None
        feats = self.chem_feature_factory.GetFeaturesForMol(mol)

        g = nx.DiGraph()

        # for training set, we store its target
        # otherwise, we store its molecule id
        l = torch.FloatTensor(self.target.loc[int(sdf_file.stem)].tolist()).unsqueeze(0) \
                if self.mode == 'dev' else torch.LongTensor([int(sdf_file.stem)])

        n_atoms = mol.GetNumAtoms()
        if n_atoms != 25:
            return None
        self.n_atoms_.append(n_atoms)
        # Create nodes
        assert len(mol.GetConformers()) == 1
        geom = mol.GetConformers()[0].GetPositions()
        for i in range(n_atoms):
            atom_i = mol.GetAtomWithIdx(i)
            g.add_node(i,
                       a_type=atom_i.GetSymbol(),
                       a_num=atom_i.GetAtomicNum(),
                       acceptor=0,
                       donor=0,
                       aromatic=atom_i.GetIsAromatic(),
                       hybridization=atom_i.GetHybridization(),
                       num_h=atom_i.GetTotalNumHs())
            gf_string += self.compute_gf_string(atom_i, i)

        for i in range(len(feats)):
            if feats[i].GetFamily() == 'Donor':
                node_list = feats[i].GetAtomIds()
                for i in node_list:
                    g.nodes[i]['donor'] = 1
                    # gf_string += "d" + str(i)
            elif feats[i].GetFamily() == 'Acceptor':
                node_list = feats[i].GetAtomIds()
                for i in node_list:
                    g.nodes[i]['acceptor'] = 1
                    # gf_string += "a" + str(i)
        # Read Edges
        ed = 0
        for i in range(mol.GetNumAtoms()):
            for j in range(mol.GetNumAtoms()):
                e_ij = mol.GetBondBetweenAtoms(i, j)
                if e_ij is not None and ed < 50:
                    g.add_edge(i, j, b_type=e_ij.GetBondType())
                    gf_string += "e" + str(i) + "_" + str(j)
                    ed += 1
        # print(f"edges:{ed}")

        if self.lat2ix.get(gf_string) is None:
            self.lat2ix[gf_string] = int(sdf_file.stem)
        else:
            print('error lat2ix')

        key = int(sdf_file.stem)
        if self.ix2lat.get(key) is None:
            self.ix2lat[key] = gf_string
        else:
            print('error ix2lat')

        self.gix += 1
        self.ix2gix[key] = self.gix

        node_attr = self.alchemy_nodes(g)
        edge_index, edge_attr = self.alchemy_edges(g)
        data = Data(
            x=node_attr,
            pos=torch.FloatTensor(geom),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=l,
        )
        return data

    def process(self):
        """process raw data into graph"""
        self.lat2ix = {}
        self.ix2lat = {}
        self.ix2gix = {}
        self.gix = 0

        if self.mode == 'dev':
            usecols = ['gdb_idx', ] + ['property_%d' % x for x in range(12)]
            path = self.raw_dir + self.raw_paths[1]
            self.target = pd.read_csv(path, index_col=0, names=usecols, skiprows=1)
            self.target = self.target[['property_%d' % x for x in range(12)]]
        sdf_dir = pathlib.Path(self.raw_dir + self.raw_paths[0])
        data_list = []
        for sdf_file in sdf_dir.glob("**/*.sdf"):
            alchemy_data = self.sdf_graph_reader(sdf_file)
            if alchemy_data is not None:
                data_list.append(alchemy_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        with open(self.processed_paths[1], "w") as f:
            json.dump(self.ix2lat, f)
        with open(self.processed_paths[2], "w") as f:
            json.dump(self.lat2ix, f)
        with open(self.processed_paths[5], "w") as f:
            json.dump(self.ix2gix, f)

        df = pd.DataFrame({"a_type_ls": self.a_type_ls, "aromatic_ls": self.aromatic_ls, "hyb_ls": self.hyb_ls, "num_h_ls": self.num_h_ls})
        df.to_csv(self.processed_paths[3])
        df2 = pd.DataFrame({"natom": self.n_atoms_})
        df2.to_csv(self.processed_paths[4])

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    ds = TencentAlchemyDataset(root="../data/Alchemy/", mode='dev')
    # ds.process()
    lats = ds.sample_latent()
    g = ds.get_img_by_latent(lats)
    # print(g)
    # i0 = ds.__getitem__(0)



