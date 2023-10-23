import torch.nn as nn
from envs import *
from read_data import save_fig, save_fig_with_result
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import numpy
import torch

# torch.set_printoptions(profile="full")
class GCN(nn.Module):
    def __init__(self, k, n, hidden_dim, activation=F.relu):
        super(GCN, self).__init__()
        self.n = n
        self.l1 = nn.Linear(k, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(1, hidden_dim)
        self.activ = activation

    def forward(self, graphs, feature, use_label=True, use_edge=False):  # use_edge=True
        b = graphs.batch_size
        n = graphs.n
        l = graphs.ndata['label']  # node label [l]  (bn, k)
        # print(l)
        adjM = graphs.ndata['adj']
        n1_h = torch.bmm(adjM.view(b, n, n), feature.view(b, n, -1)).view(b * n, -1) / n  # (bn, h)

        if use_edge:
            n2_h = self.activ(self.l4(adjM.view(b, n, n, 1))).sum(dim=-2).view(b * n, -1) / n  # (bn, h)
            h = self.activ(self.l1(torch.cat([l], dim=1)) + self.l2(n1_h) + self.l3(n2_h), inplace=True)
        else:
            h = self.activ(self.l1(torch.cat([l], dim=1)) + self.l2(n1_h), inplace=True)
        # print(h)
        return h


def bPtAP(P, A, b, n):
    return torch.bmm(torch.bmm(P.transpose(1, 2), A.view(b, n, n)), P)


def batch_trace(A, b, k, cuda_flag):
    if cuda_flag:
        return (A * torch.eye(k).repeat(b, 1, 1).cuda()).sum(dim=2).sum(dim=1) / 2
    else:
        return (A * torch.eye(k).repeat(b, 1, 1)).sum(dim=2).sum(dim=1) / 2


class DQNet(nn.Module):
    def __init__(self, k, n, hidden_dim):
        super(DQNet, self).__init__()
        self.k = k
        self.n = n
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([GCN(k, n, hidden_dim)])
        # baseline
        self.t5 = nn.Linear(self.hidden_dim, 1)
        self.t5_ = nn.Linear(self.hidden_dim + 1, 1)
        self.t6 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.t7 = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.t7_0 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.t7_1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.t8 = nn.Linear(self.hidden_dim + self.k, self.hidden_dim)
        self.t9 = nn.Linear(2 * self.hidden_dim, self.hidden_dim)

        self.Wa = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.l1 = nn.Linear(self.hidden_dim, self.k)
        self.l2 = nn.Linear(self.k, self.k)
        self.l3 = nn.Linear(1, 1)

        self.l4 = nn.Linear(self.hidden_dim, self.n)
        self.l5 = nn.Linear(self.n, self.n)

        self.L1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.L2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.L3 = nn.Linear(2, 1)
        self.L4 = nn.Linear(self.hidden_dim, 2)
        self.L5 = nn.Linear(2, 1)

    def forward_prop(self, graphs, actions=None, action_type='swap', gnn_step=3, top_ratio=0.1):

        n = graphs.n
        b = graphs.batch_size
        bn = b * n
        num_action = actions.shape[0] // b

        A = graphs.ndata['adj']

        h = torch.zeros((bn, self.hidden_dim))

        if graphs.in_cuda:
            h = h.cuda(device=A.device)

        for _ in range(gnn_step):
            h = self.layers[0].forward(graphs, h)  # (bn, h)

        action_mask = torch.tensor(range(0,bn,n)).unsqueeze(1).expand(b,2).repeat(1,num_action).view(num_action*b,-1)

        if graphs.in_cuda:
            action_mask = action_mask.cuda()
        actions_ = actions + action_mask
        
        # type_constraint = (actions_[:, 1] - actions_[:, 0])%4 == 0
        # actions_ = actions_[type_constraint]

        state_embedding = h.view(b, n, -1).mean(dim=1)

        # Action proposal network: 2-layer MLP
        proto_a = F.relu(self.L2(F.relu(self.L1(state_embedding.detach())))).unsqueeze(1)  # (b, 1, h)
        h_a = self.L4(h.view(b, n, -1) * proto_a).view(b*n, 2)  # (b, n, h) -> (b, n, 2) -> (bn, 2)
        h_a_0 = h_a[actions_[:, 0], :]
        h_a_1 = h_a[actions_[:, 1], :]
        prop_a_score = self.L5(F.relu((h_a_0 + h_a_1 + self.L3(h_a_0 * h_a_1)))).view(b, -1).softmax(dim=1)  # (b, num_action)
        topk_action_num = int(top_ratio * num_action)
        prop_action_indices = torch.multinomial(prop_a_score, topk_action_num).view(-1)  # (b, topk) -> (b * topk)

        topk_mask = torch.tensor(range(0, num_action*b,num_action)).unsqueeze(1).repeat(1,topk_action_num).view(-1)
        if graphs.in_cuda:
            topk_mask = topk_mask.cuda()
        prop_action_indices_ = prop_action_indices + topk_mask
        prop_actions = actions[prop_action_indices_, :]

        return prop_a_score.view(b * num_action, 1)[prop_action_indices_, :].view(b, -1), prop_actions

    def forward(self, graphs, actions=None, action_type='swap', gnn_step=3, use_attention=False, leak=False, save_figure=False):

        n = graphs.n
        b = graphs.batch_size
        m = graphs.m
        bn = b * n
        num_action = actions.shape[0] // b

        L = graphs.ndata['label']
        A = graphs.ndata['adj']
        
        h = torch.zeros((bn, self.hidden_dim))

        if graphs.in_cuda:
            h = h.cuda(device=A.device)

        for _ in range(gnn_step):
            h = self.layers[0].forward(graphs, h)  # (bn, hidden_dim)
            
        action_mask = torch.tensor(range(0, bn, n))\
            .unsqueeze(1).expand(b, 2)\
            .repeat(1, num_action)\
            .view(num_action * b, -1)

        if graphs.in_cuda:
            action_mask = action_mask.cuda()
            
        actions_ = actions + action_mask
    
        # one action [6,10]
        Ha = (self.t7_0(h[actions_[:, 0], :]) + self.t7_1(h[actions_[:, 1], :])).view(b, num_action, -1)

        if use_attention:
            # (b, k, h) L * H
            Hc = torch.bmm(L.view(b, n, -1).transpose(1, 2), h.view(b, n, -1))
            # weight (b, k, na)
            wi = torch.bmm(Hc, self.Wa(Ha).transpose(1, 2))
            # (b, k, h) * (b, k, na) -> (b, na, h)
            state_embedding = torch.bmm(wi.transpose(1, 2), Hc)
        else:
            state_embedding = h.view(b, n, -1).mean(dim=1)
        if action_type == 'swap':
            if leak:
                immediate_rewards = peek_greedy_reward(states=graphs, actions=actions,
                                                       action_type=action_type,
                                                       m=m).unsqueeze(1)
                Q_sa = self.t5_(torch.cat(
                    [F.relu(
                        (
                                self.t6(state_embedding).view(b, -1, self.hidden_dim) + Ha
                        ).view(b * num_action, -1), inplace=True), immediate_rewards], dim=1)
                ).squeeze()
            else:
                Q_sa = self.t5(F.relu(
                    (
                        self.t6(state_embedding).view(b, -1, self.hidden_dim) + Ha
                    ).view(b * num_action, -1), inplace=True)
                ).squeeze()
        elif action_type == 'flip':
            if leak:
                immediate_rewards = peek_greedy_reward(states=graphs, actions=actions,
                                                       action_type=action_type,
                                                       m=m).unsqueeze(1)
                Q_sa = self.t5_(torch.cat(
                    [F.relu(
                        (
                            self.t6(state_embedding).view(b, 1, -1)
                            + (self.t8(torch.cat([h[actions_[:, 0], :], torch.nn.functional.one_hot(actions[:, 1], self.k).float()], axis=1))).view(b, num_action,
                                                                                                           -1)
                        ).view(b * num_action, -1), inplace=True), immediate_rewards ], dim=1)
                ).squeeze()
            else:
                Q_sa = self.t5(F.relu(
                        (
                                self.t6(state_embedding).view(b, 1, -1)
                                # self.t6(cluster_embedding).view(b, 1, -1)
                                + (self.t8(torch.cat(
                            [h[actions_[:, 0], :], torch.nn.functional.one_hot(actions[:, 1], self.k).float()],
                            axis=1))).view(b, num_action,
                                           -1)
                        ).view(b * num_action, -1), inplace=True)
                ).squeeze()
                
        return 0, 0, h.view(b, n, -1), Q_sa
