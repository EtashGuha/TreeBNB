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
import os
faulthandler.enable()
torch.set_printoptions(precision=10)
class Dagger():
    def __init__(self, selector, problem_dir, device, num_train=None, num_epoch = 1):
        self.policy = selector
        self.problems = glob.glob(problem_dir + "/*.lp")
        if num_train is None:
            self.num_train = len(self.problems)
        else:
            self.num_train = num_train
        self.model = Model("setcover")
        self.sfeature_list = []
        self.soracle = []
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.device = device
        self.prev = None
        self.num_epoch = num_epoch

    def train(self):
        self.policy.train()
        counter = 0
        for epoch in range(self.num_epoch):
            for problem in self.problems:
                if counter > self.num_train:
                    break
                counter += 1

                temp_features = []

                print(problem)
                print(counter)
                model = Model("setcover")
                model.setIntParam('separating/maxroundsroot', 0)
                model.setBoolParam('conflict/enable', False)
                step_ids = []
                ourNodeSel = MyNodesel(model, self.policy, dataset=temp_features, step_ids=step_ids)
                model.readProblem(problem)
                model.includeNodesel(ourNodeSel, "nodesel", "My node selection", 999999, 999999)
                model.optimize()

                optimal_node = None
                ourNodeSel.tree.show()
                for node in ourNodeSel.tree.leaves():
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

                for i in range(len(temp_features)):
                    queue_contains_optimal = False
                    optimal_id = None
                    idlist = step_ids[i].flatten().tolist()
                    for id in idlist:
                        if id in optimal_ids:
                            queue_contains_optimal = True
                            optimal_id = id
                            break
                    if queue_contains_optimal:
                        self.soracle.append((step_ids[i]== optimal_id).type(torch.uint8).nonzero()[0][0])
                        self.sfeature_list.append(temp_features[i])

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

                os.remove("/Users/etashguha/Documents/TreeBNB/lstmFeature.pt")
                torch.save(self.policy.state_dict(), "/Users/etashguha/Documents/TreeBnB/lstmFeature.pt")
