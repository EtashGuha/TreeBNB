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
from NodeSel import MyNodesel, LinNodesel
from nodeutil import getListOptimalID, checkIsOptimal
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
                # ourNodeSel.tree.show(data_property="nodeid")
                if len(ourNodeSel.tree.all_nodes()) < 2:
                    continue
                for node in ourNodeSel.tree.leaves():
                    if checkIsOptimal(node, model, ourNodeSel.tree):
                        optimal_node = node
                        print("FOUND OPTIMal")

                if optimal_node is None:
                    continue

                optimal_ids = getListOptimalID(optimal_node.identifier, ourNodeSel.tree)
                print(optimal_ids)
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

                # print(optimal_ids)
                #             s_loader = DataLoader(samples, batch_size=32, shuffle=True, collate_fn=collate)
                for (bg, label) in samples:
                    self.optimizer.zero_grad()
                    g = bg
                    n = g.number_of_nodes()
                    h_size = 14
                    h = torch.zeros((n, h_size))
                    c = torch.zeros((n, h_size))
                    output, _ = self.policy(g, h, c)
                    output = output.unsqueeze(0)
                    label = label.unsqueeze(0)
                    loss = self.loss(output, label)
                    loss.backward()
                    self.optimizer.step()
                if os.path.exists("/Users/etashguha/Documents/TreeBNB/lstmFeature.pt"):
                    os.remove("/Users/etashguha/Documents/TreeBNB/lstmFeature.pt")
                torch.save(self.policy.state_dict(), "/Users/etashguha/Documents/TreeBnB/lstmFeature.pt")


class LinDagger():
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
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.device = device
        self.prev = None
        self.num_epoch = num_epoch
        self.listNNodes = []

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
                ourNodeSel = LinNodesel(model, self.policy, dataset=temp_features)
                model.readProblem(problem)
                model.includeNodesel(ourNodeSel, "nodesel", "My node selection", 999999, 999999)
                try:
                    model.optimize()
                except:
                    continue

                self.listNNodes.append(model.getNNodes())
                print(self.listNNodes)
                optimal_node = None
                # ourNodeSel.tree.show(data_property="variables")
                for node in ourNodeSel.tree.leaves():
                    if checkIsOptimal(node, model, ourNodeSel.tree):
                        optimal_node = node
                        print("FOUND OPTIMal")


                if optimal_node is None:
                    continue

                optimal_ids = getListOptimalID(optimal_node.identifier, ourNodeSel.tree)

                for idToFeature in temp_features:
                    ids = idToFeature.keys()
                    for id in ids:
                        if id in optimal_ids:
                            for otherid in ids:
                                if id == otherid:
                                    continue
                                self.sfeature_list.append(idToFeature[id] - idToFeature[otherid])
                                self.soracle.append(torch.tensor([1], dtype=torch.float32));
                                self.sfeature_list.append(idToFeature[otherid] - idToFeature[id])
                                self.soracle.append(torch.tensor([-1], dtype=torch.float32));

                samples = list(zip(self.sfeature_list, self.soracle))

                # print(optimal_ids)
                s_loader = DataLoader(samples, batch_size=1, shuffle=True)
                for epoch in range(10):
                    running_loss = 0.0
                    for i, (feature, label) in enumerate(s_loader):
                        self.optimizer.zero_grad()
                        output= self.policy(feature)
                        loss = self.loss(output, label)
                        loss.backward()
                        self.optimizer.step()
                        running_loss += loss.item()
                    print('[%d] loss: %.3f' %
                          (epoch + 1, running_loss/len(s_loader)))
                    running_loss = 0.0

                if os.path.exists("/Users/etashguha/Documents/TreeBNB/lstmFeature.pt"):
                    os.remove("/Users/etashguha/Documents/TreeBNB/lstmFeature.pt")
                torch.save(self.policy.state_dict(), "/Users/etashguha/Documents/TreeBnB/lstmFeature.pt")
