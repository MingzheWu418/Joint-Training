from copy import deepcopy as dc
import dgl
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from dataclasses import dataclass
import new_knn_graph
from dgl import backend as B
import time
# from pynvml import *

# torch.set_printoptions(profile="full")
@dataclass
class LightGraph:

    in_cuda: bool                   # if cuda graph
    n: int                          # number of nodes
    k: int                          # number of clusters
    m: list                         # number of nodes in each cluster
    batch_size: int                 # batch graph size
    ndata: dict                     # node attributes
    edata: dict                     # edge attributes
    kcut_value: torch.tensor        # current K-cut value
    # extend_info: tuple

    def number_of_nodes(self):
        return self.n

    def number_of_edges(self):
        return self.n * (self.n - 1)


def to_cuda(G_, copy=True):
    if copy:
        G = dc(G_)
    else:
        G = G_
    for node_attr in G.ndata.keys():
        if node_attr in ('adj', 'label'):
            G.ndata[node_attr] = G.ndata[node_attr].cuda()
    # for edge_attr in G.edata.keys():
    #     G.edata[edge_attr] = G.edata[edge_attr].cuda()
    return G


class GraphGenerator:

    def __init__(self, k, m, ajr, style='plain'):
        self.k = k
        self.m = m
        if isinstance(m, list):
            self.n = sum(m)
            assert len(m) == k
            self.cut = 'unequal'
            self.init_label = []
            for i in range(k):
                self.init_label.extend([i]*m[i])
        else:
            self.n = k * m
            self.cut = 'equal'
        self.ajr = ajr
        self.nonzero_idx = [i for i in range(self.n ** 2) if i % (self.n + 1) != 0]
        self.adj_mask = torch.tensor(range(0, self.n ** 2, self.n)).unsqueeze(1).expand(self.n, ajr + 1)
        self.style = style

    def generate_graph(self, x=None, index=None, batch_size=1, style=None, seed=None, cuda_flag=True, label_input=None):

        k = self.k
        m = self.m
        n = self.n
        ajr = self.ajr
        if style is not None:
            style = style
        else:
            style = self.style

        if style.startswith('er'):
            p = float(style.split('-')[1])
            G = [nx.erdos_renyi_graph(n, p) for _ in range(batch_size)]
            adj_matrices = torch.cat([torch.tensor(nx.adjacency_matrix(g).todense()).float() for g in G])

        elif style.startswith('ba'):
            _m = int(style.split('-')[1])
            G = [nx.barabasi_albert_graph(n, _m) for _ in range(batch_size)]
            adj_matrices = torch.cat([torch.tensor(nx.adjacency_matrix(g).todense()).float() for g in G])

        # init batch graphs
        bg = LightGraph(cuda_flag, self.n, self.k, self.m, batch_size, {}, {}, torch.zeros(batch_size))

        # assign 2-d coordinates 'x'
        if x is None:
            # if style is plain, we generate points uniformly
            if style == 'plain':
                if seed is not None:
                    np.random.seed(seed)
                    bg.ndata['x'] = torch.tensor(np.random.rand(batch_size * n, 2)).float()
                else:
                    bg.ndata['x'] = torch.rand((batch_size * n, 2))
            # if style is cluster, we generate clustered points
            # that are normally distributed round the mean
            elif style.startswith('cluster'):
                mean = torch.rand((batch_size * k, 1, 805)).repeat(1, m, 1)
                std = (torch.rand((batch_size * k, 1, 805)).repeat(1, m, 1) + 1) * 0.1
                bg.ndata['x'] = torch.normal(mean, std).view(batch_size * n, 805)
        else:
            bg.ndata['x'] = x
            
        # assigning initial label
        if label_input is not None:
            bg.ndata['label'] = label_input
        else:
            if self.cut == 'equal':
                label = torch.tensor(range(k)).unsqueeze(1).repeat(batch_size, m).view(-1)
            else:
                label = torch.tensor(self.init_label).repeat(batch_size)
            batch_mask = torch.tensor(range(0, n * batch_size, n)).unsqueeze(1).expand(batch_size, n).flatten()
            if seed is not None:
                perm_idx = torch.cat([torch.tensor(np.random.permutation(n)) for _ in range(batch_size)]) + batch_mask
            else:
                # perm_idx = torch.cat([torch.randperm(n) for _ in range(batch_size)]) + batch_mask 

                # use this if batch size is larger than one
                perm_idx = torch.cat(\
                    [torch.flatten(torch.cat([(m*torch.randperm(k)+i).unsqueeze(1) \
                    for i in range(m)], dim=1)) \
                    for _ in range(batch_size)]).squeeze() 
                    + batch_mask

            label = label[perm_idx].view(batch_size, n)
            bg.ndata['label'] = torch.nn.functional.one_hot(label, k).float().view(batch_size * n, k)

        # d/adj mat
        if style.startswith('er') or style.startswith('ba'):
            bg.edata['d'] = adj_matrices.view(batch_size, -1, 1)[:, self.nonzero_idx, :].view(-1, 1)
        else:

            # Below is training on euclidean distances
            # Disabling this part to test if non-linear transformation is causing differences between GA and RL
            
            """
            _, neighbor_idx, square_dist_matrix = new_knn_graph.knn_graph(bg.ndata['x'].view(batch_size, n, -1), ajr + 1, extend_info=True)

            square_dist_matrix = F.relu(square_dist_matrix, inplace=True)  # numerical error could result in NaN in sqrt. value

            bg.ndata['adj'] = torch.sqrt(square_dist_matrix).view(bg.n * bg.batch_size, -1) 

            """

            # Below is training on correlation coefficient
            
            # neighbor_idx = []
            if index is not None:
                bg.ndata['index'] = index
                # can do slicing without considering batch size, since we are going to reshape it anyway
                embedding = bg.ndata['x'].reshape(-1, m, 86)[index, :, :].reshape(-1, 86)
            else:
                embedding = bg.ndata['x']
            if index is not None:
                CUDA_LAUNCH_BLOCKING=1
                coef_all = torch.unsqueeze(1-torch.corrcoef(bg.ndata['x']), 0)
                sensor_idx = ((m*index).repeat_interleave(m)+torch.arange(m).repeat(batch_size * k)).reshape(batch_size, m*k)
                coef = torch.cat([coef_all[:, idx, :][:, :, idx] for idx in sensor_idx])
                bg.ndata['adj'] = coef.view(n * batch_size, -1)
            else:
                coef = torch.cat([1-torch.corrcoef(embedding.view(batch_size, n, -1)[i]).view(1, n, -1) for i in range(batch_size)], dim=0)
                bg.ndata['adj'] = coef.view(n * batch_size, -1)

            neighbor_idx = B.argtopk(coef, ajr+1, 2, descending=False).cpu()
            # del first_coef, 
            # torch.cuda.empty_cache()
            
            """ 
            scale d (maintain avg=0.5):
            Updated from bg.ndata['adj'].shape[0]**2 to bg.ndata['adj'].shape[0]*bg.ndata['adj'].shape[1], 
            since the adjacency matrices for multiple graphs are not square.
            """
            # if style != 'plain':
            #     bg.ndata['adj'] /= (bg.ndata['adj'].sum() / (bg.ndata['adj'].shape[0]*bg.ndata['adj'].shape[1]) / 0.5)

            # bg.edata['d'] = bg.ndata['adj'].view(batch_size, -1, 1)[:, self.nonzero_idx, :].view(-1, 1).cpu()
            bg.edata['d'] = coef.view(batch_size, -1, 1)[:, self.nonzero_idx, :].view(-1, 1).cpu()

        # e_type, adding edge information to the graph
        group_matrix = torch.bmm(bg.ndata['label'].view(batch_size, n, -1), bg.ndata['label'].view(batch_size, n, -1).transpose(1, 2)).view(batch_size, -1)[:, self.nonzero_idx].view(-1, 1).cpu()

        if style.startswith('er') or style.startswith('ba'):
            bg.edata['e_type'] = torch.cat([bg.edata['d'], group_matrix], dim=1)
        else:
            neighbor_idx -= torch.tensor(range(0, batch_size * n, n)).view(batch_size, 1, 1).repeat(1, n, ajr + 1) \
                            - torch.tensor(range(0, batch_size * n * n, n * n)).view(batch_size, 1, 1).repeat(1, n,
                                                                                                              ajr + 1)
            adjacent_matrix = torch.zeros((batch_size * n * n, 1))
            adjacent_matrix[neighbor_idx + self.adj_mask.repeat(batch_size, 1, 1)] = 1
            adjacent_matrix = adjacent_matrix.view(batch_size, n * n, 1)[:, self.nonzero_idx, :].view(-1, 1)
            bg.edata['e_type'] = torch.cat([adjacent_matrix, group_matrix], dim=1)

        if cuda_flag:
            to_cuda(bg, copy=False)
        # calculating kcut value
        bg.kcut_value = calc_S(bg)
        return bg


def calc_S(states, mode='complete'):
    S = states.edata['e_type'][:, 1].clone() * states.edata['d'][:, 0]
    if mode != 'complete':
        S *= states.edata['e_type'][:, 0]
    return S.view(states.batch_size, -1).sum(dim=1) / 2


def make_batch(graphs):
    # make graphs into a batch, to increase time complexity
    bg = LightGraph(graphs[0].in_cuda
                    , graphs[0].number_of_nodes()
                    , graphs[0].k
                    , graphs[0].m
                    , len(graphs) * graphs[0].batch_size
                    , {}, {}, torch.zeros(len(graphs)))

    for node_attr in graphs[0].ndata.keys():
        if node_attr == "x":
            bg.ndata[node_attr] = graphs[0].ndata[node_attr]
        else:
            bg.ndata[node_attr] = torch.cat([g.ndata[node_attr] for g in graphs])
    for edge_attr in graphs[0].edata.keys():
        bg.edata[edge_attr] = torch.cat([g.edata[edge_attr] for g in graphs])
    bg.kcut_value = torch.cat([g.kcut_value for g in graphs])

    return bg


def un_batch(graphs, copy=True):
    n = graphs.number_of_nodes()
    e = graphs.number_of_edges()
    batch_size = graphs.batch_size

    ndata = {}.fromkeys(graphs.ndata.keys())
    edata = {}.fromkeys(graphs.edata.keys())
    if copy:
        kcut_value = graphs.kcut_value.clone()
    else:
        kcut_value = graphs.kcut_value

    for node_attr in graphs.ndata.keys():
        if copy:
            if node_attr == "x":
                ndata[node_attr] = [graphs.ndata[node_attr].clone() for i in range(batch_size)]
            elif node_attr == "index":
                ndata[node_attr] = [torch.unsqueeze(graphs.ndata[node_attr][i, :].clone(), 0) for i in range(batch_size)]
            else:
                ndata[node_attr] = [graphs.ndata[node_attr][i*n:(i+1)*n, :].clone() for i in range(batch_size)]
        else:
            if node_attr == "x":
                ndata[node_attr] = [graphs.ndata[node_attr].clone() for i in range(batch_size)]
            elif node_attr == "index":
                ndata[node_attr] = [torch.unsqueeze(graphs.ndata[node_attr][i, :], 0) for i in range(batch_size)]
            else:
                ndata[node_attr] = [graphs.ndata[node_attr][i * n:(i + 1) * n, :] for i in range(batch_size)]
    for edge_attr in graphs.edata.keys():
        if copy:
            edata[edge_attr] = [graphs.edata[edge_attr][i*e:(i+1)*e, :].clone() for i in range(batch_size)]
        else:
            edata[edge_attr] = [graphs.edata[edge_attr][i * e:(i + 1) * e, :] for i in range(batch_size)]

    graph_list = [LightGraph(graphs.in_cuda
                             , n
                             , graphs.k
                             , graphs.m
                             , 1
                             , dict([(n_attr, ndata[n_attr][i]) for n_attr in graphs.ndata.keys()])
                             , dict([(e_attr, edata[e_attr][i]) for e_attr in graphs.edata.keys()])
                             , kcut_value[i:i+1]) for i in range(batch_size)]

    return graph_list


def perm_weight(graphs, eps=0.1):
    a = F.relu(torch.ones(graphs.edata['d'].shape).cuda() + eps * torch.randn(graphs.edata['d'].shape).cuda())
    graphs.edata['d'] *= a.to('cpu')


def reset_label(graphs, label, compute_S=True, rewire_edges=True):

    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label)
    label = graphs.ndata['label'] = torch.nn.functional.one_hot(label, graphs.k).float()
    if graphs.in_cuda:
        graphs.ndata['label'] = label.cuda()
    else:
        graphs.ndata['label'] = label

    if rewire_edges:
        nonzero_idx = [i for i in range(graphs.n ** 2) if i % (graphs.n + 1) != 0]
        graphs.edata['e_type'][:, 1] = torch.bmm(graphs.ndata['label'].view(graphs.batch_size, graphs.n, -1)
                                             , graphs.ndata['label'].view(graphs.batch_size, graphs.n, -1).transpose(1, 2)) \
                                       .view(graphs.batch_size, -1)[:, nonzero_idx].view(-1)
    if compute_S:
        graphs.kcut_value = calc_S(graphs)
    return graphs
