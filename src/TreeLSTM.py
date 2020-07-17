import torch
import torch.nn as nn
import dgl
import faulthandler
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
faulthandler.enable()
import math
import numpy as np
def plot_tree(g):
    # this plot requires pygraphviz package
    pos = nx.nx_agraph.graphviz_layout(g, prog='dot')
    nx.draw(g, pos, with_labels=True, node_size=50,
            node_color=[[.5, .5, .5]], arrowsize=4)
    plt.show()

class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.W_f = nn.Linear(x_size, h_size)
        self.b_f = nn.Parameter(torch.zeros(1, h_size))
        self.U_f = nn.Linear(h_size, h_size)


    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        # concatenate h_jl for equation (1), (2), (3), (4)
        h_cat = nodes.mailbox['h'].view(nodes.mailbox['h'].size(0), -1)
        f = torch.sigmoid(nodes.data["Wfx"] + self.U_f(h_cat) + self.b_f).view(*nodes.mailbox['h'].size())
        # second term of equation (5)
        c = torch.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_cat), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['Wx'] + self.b_iou + nodes.data["iou"]
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        # equation (5)
        c = i * u + nodes.data['c']
        # equation (6)
        h = o * torch.tanh(c)
        return {'h': h, 'c': c}


# %%
class TreeLSTM(nn.Module):
    def __init__(self,
                 x_size,
                 h_size,
                 dropout,
                 device=torch.device("cpu")):
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        self.dropout = nn.Dropout(dropout)
        self.cell = TreeLSTMCell(x_size, h_size)
        self.linear = nn.Sequential(
            nn.Linear(h_size, 2 * h_size),
            nn.Linear(2 * h_size, h_size),
            nn.Linear(h_size, 1)
        )
        # self.linear = nn.Linear(h_size, 1)
        self.smax = nn.Softmax(dim=0)
        self.device = device

    def forward(self, g, h, c, iou):
        """Compute tree-lstm prediction given a batch.

        Parameters
        ----------
        batch : dgl.data.SSTBatch
            The data batch.
        h : Tensor
            Initial hidden state.
        c : Tensor
            Initial cell state.

        Returns
        -------
        logits : Tensor
            The prediction of each node.
        """
        g.register_message_func(self.cell.message_func)
        g.register_reduce_func(self.cell.reduce_func)
        g.register_apply_node_func(self.cell.apply_node_func)
        # feed embedding
        features = g.ndata["feature"]
        features = features.to(device=self.device)
        g.ndata['Wx'] = self.cell.W_iou(features)
        g.ndata["Wfx"] = self.cell.W_f(features)
        g.ndata["iou"] = iou
        g.ndata['h'] = h
        g.ndata['c'] = c
        g.to(self.device)
        # propagate
        dgl.prop_nodes_topo(g, reverse=True)
        dgl.prop_nodes_topo(g)
        # compute logits
        h = g.ndata['h']
        ids = g.ndata["node_id"][(g.ndata["node_id"] * g.ndata["in_queue"]).nonzero()]
        tas = self.linear(h).squeeze(0)
        vals = tas * torch.autograd.Variable(g.ndata["in_queue"].unsqueeze(dim=1))
        vals = vals.squeeze(dim=1)
        nonZeroed = vals[vals.nonzero()]
        probs = nonZeroed.squeeze(dim=1)
        return probs, ids
    #Leaf to the root and then go back to leaf
    #Supervised learning to initialize
    #Use dagger sampling to improve

class ShallowLib(nn.Module):
    def __init__(self, in_dim):
        super(ShallowLib, self).__init__()
        self.fc1 = nn.Linear(in_dim, 2 * in_dim)
        self.fc2 = nn.Linear(2 * in_dim, in_dim)
        self.fc3 = nn.Linear(in_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)

class LinLib(nn.Module):
    def __init__(self, in_dim):
        super(LinLib, self).__init__()
        self.fc1 = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.fc1(x)




class TreeLSTMBranch(nn.Module):
    def __init__(self,
                 x_size,
                 h_size,
                 dropout,
                 device=torch.device("cpu")):
        super(TreeLSTMBranch, self).__init__()
        self.x_size = x_size
        self.dropout = nn.Dropout(dropout)
        self.cell = TreeLSTMCell(x_size, h_size)
        self.linear = nn.Sequential(
            nn.Linear(h_size, 1)
        )
        # self.linear = nn.Linear(h_size, 1)
        self.smax = nn.Softmax(dim=0)
        self.device = device
        self.mu = .5
    def forward(self, g, h, c, iou, branch_cands):
        """Compute tree-lstm prediction given a batch.

        Parameters
        ----------
        batch : dgl.data.SSTBatch
            The data batch.
        h : Tensor
            Initial hidden state.
        c : Tensor
            Initial cell state.

        Returns
        -------
        logits : Tensor
            The prediction of each node.
        """
        g.register_message_func(self.cell.message_func)
        g.register_reduce_func(self.cell.reduce_func)
        g.register_apply_node_func(self.cell.apply_node_func)
        # feed embedding
        features = g.ndata["feature"]
        features = features.to(device=self.device)
        g.ndata['Wx'] = self.cell.W_iou(features)
        g.ndata["Wfx"] = self.cell.W_f(features)
        g.ndata["iou"] = iou
        g.ndata['h'] = h
        g.ndata['c'] = c
        g.to(self.device)
        # propagate
        dgl.prop_nodes_topo(g, reverse=True)
        dgl.prop_nodes_topo(g)
        # compute logits
        h = self.linear(g.ndata['h']).squeeze(dim=1)
        down_scores = h * g.ndata['scaled_improvement_down']
        up_scores = h * g.ndata["scaled_improvement_up"]
        vars = g.ndata["variable_chosen"]
        max_score = -1 * math.inf
        best_var = None
        scores = torch.tensor([], dtype=torch.float)
        for i in range(len(branch_cands[0])):
            var = branch_cands[0][i].getIndex()
            history = (vars == var).type(torch.float)
            if len(history.nonzero()) == 0:
                score = torch.tensor([0], dtype=torch.float)
            else:
                pseudodown = torch.dot(history,down_scores)/torch.sum(history)
                pseudoup = torch.dot(history,up_scores)/torch.sum(history)
                score = (1 - self.mu) * min(pseudodown, pseudodown) + self.mu * max(pseudodown, pseudoup)
                score = score.unsqueeze(dim=0)
            scores = torch.cat((scores, score), dim = 0)
            if score > max_score:
                best_var = branch_cands[0][i]
                max_score = score

        print(max_score)
        return best_var, scores
