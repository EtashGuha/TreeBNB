import torch
# import utilities as ut
# import utilities_gnn as ut_gnn
import os
import argparse
import time
import pickle
import torch.sparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
import torch.sparse
import itertools
from treelib import Tree
import networkx as nx
import dgl
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
import sys
import faulthandler
from TreeLSTM import TreeLSTMCell, TreeLSTM
from NodeSel import MyNodesel
from nodeutil import getListOptimalID
from pyscipopt import Model, Heur, quicksum, multidict, SCIP_RESULT, SCIP_HEURTIMING, SCIP_PARAMSETTING, Sepa, \
    Branchrule, Nodesel
import glob
torch.set_printoptions(precision=10)
class Dagger():
    def __init__(self, selector, problem_dir, device):
        self.policy = selector
        self.problems = glob.glob(problem_dir + "/*.lp")
        self.model = Model("setcover")
        self.sfeature_list = []
        self.soracle = []
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.device = device
        self.prev = None

    def train(self):
        self.policy.train()
        for problem in self.problems:
            print(problem)
            model = Model("setcover")
            model.setIntParam('separating/maxroundsroot', 0)
            model.setBoolParam('conflict/enable', False)
            ourNodeSel = MyNodesel(model, self.policy)
            model.readProblem(problem)
            model.includeNodesel(ourNodeSel, "nodesel", "My node selection", 999999, 999999)
            model.optimize()
            optimal_node = None
            modelObjVal = model.getObjVal()

            #             ourNodeSel.tree.show()
            for node in ourNodeSel.tree.leaves():
                curr_node = node.data.node
                isOptimal = True
                variables = node.data.variables
                bound_types = node.data.bound_types
                branch_bounds = node.data.branch_bounds
                for i in range(len(variables)):
                    optval = model.getVal(variables[i])
                    if ((bound_types[i] == 0 and optval < branch_bounds[i]) or (
                            bound_types[i] == 1 and optval > branch_bounds[i])):
                        isOptimal = False
                        break
                if isOptimal:
                    optimal_node = node
                    break

            if optimal_node is None:
                continue

            optimal_ids = getListOptimalID(optimal_node.identifier, ourNodeSel.tree)

            model = Model("setcover")
            model.readProblem(problem)
            ourNodeSel = MyNodesel(model, self.policy, train=True, dataset=self.sfeature_list, oracle=self.soracle,
                                   optimal_ids=optimal_ids)
            model.includeNodesel(ourNodeSel, "nodesel", "My node selection", 999999, 999999)
            model.optimize()

            samples = list(zip(self.sfeature_list, self.soracle))
            #             s_loader = DataLoader(samples, batch_size=32, shuffle=True, collate_fn=collate)
            for (bg, label) in samples:
                self.optimizer.zero_grad()
                g = bg
                n = g.number_of_nodes()
                h_size = 7
                h = torch.zeros((n, h_size))
                c = torch.zeros((n, h_size))
                output, _ = self.policy(g, h, c)
                output = output.unsqueeze(0)
                label = label.unsqueeze(0)
                loss = self.loss(output, label)
                loss.backward()
                self.optimizer.step()