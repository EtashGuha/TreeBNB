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
from TreeLSTM import TreeLSTMBranch
from utilities import init_scip_params, init_scip_params_haoran
from brancher import TreeBranch

import os
faulthandler.enable()
torch.set_printoptions(precision=10)
def plot_tree(g):
    # this plot requires pygraphviz package
    pos = nx.nx_agraph.graphviz_layout(g, prog='dot')
    nx.draw(g, pos, with_labels=True, node_size=50,
            node_color=[[.5, .5, .5]], arrowsize=4)
    plt.show()

def collate(samples):
    graphs, labels, debug = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


def size_splits(tensor, split_sizes, dim=0):
    """Splits the tensor according to chunks of split_sizes.

    Arguments:
        tensor (Tensor): tensor to split.
        split_sizes (list(int)): sizes of chunks
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()

    dim_size = tensor.size(dim)
    if dim_size != torch.sum(torch.Tensor(split_sizes)):
        raise KeyError("Sum of split sizes exceeds tensor dim")

    splits = torch.cumsum(torch.Tensor([0] + split_sizes), dim=0)[:-1]

    return tuple(tensor.narrow(int(dim), int(start), int(length))
                 for start, length in zip(splits, split_sizes))

class Dagger():
    def __init__(self, selector, problem_dir, device, loss, num_train=None, num_epoch = 3, num_repeat=1, batch_size=5, save_path=None):
        print(num_repeat)
        self.policy = selector
        self.save_path = save_path
        self.problems = glob.glob(problem_dir + "/*.lp")
        if num_train is None:
            self.num_train = len(self.problems)
        else:
            self.num_train = num_train
        self.model = Model("setcover")
        self.sfeature_list = []
        self.soracle = []
        self.loss = loss
        self.optimizer = optim.Adam(self.policy.parameters(), lr= 1e-3)
        self.device = device
        self.prev = None
        self.num_epoch = num_epoch
        self.num_train = num_train
        self.listNNodes = []
        self.debug = []
        self.batch_size = batch_size
        self.num_repeat = num_repeat
        self.num_features = 0

    def isScippable(self):
        if self.num_features == len(self.sfeature_list):
            return True
        else:
            return False
        self.num_features = len(self.sfeature_list)

    def test(self, problems, MyNodesel):
        with torch.no_grad():
            real_problems = glob.glob(problems + "/*.lp")
            num_nodes = []
            default = []
            for problem in real_problems:
                print(problem)
                model = Model("setcover")
                ourNodeSel = MyNodesel(model, self.policy)
                model.readProblem(problem)
                # model.setRealParam('limits/time', self.time_limit)
                model.includeNodesel(ourNodeSel, "nodesel", "My node selection", 999999, 999999)
                init_scip_params_haoran(model, 10)
                model.optimize()
                num_nodes.append(model.getNNodes())
            for problem in real_problems:
                print(problem)
                model = Model("setcover")
                model.setRealParam('limits/time', self.time_limit)
                model.readProblem(problem)
                model.optimize()
                default.append(model.getNNodes())
        return num_nodes, default

class RankDagger(Dagger):
    def __init__(self, selector, problem_dir, device, num_train=None, num_epoch=1, time_limit=200, save_path=None, num_repeat=1):
        super().__init__(selector, problem_dir, device, nn.MSELoss(), num_train=num_train, num_epoch=num_epoch, save_path=save_path, num_repeat=num_repeat)
        self.nodesel = LinNodesel
        self.time_limit = time_limit
    def train(self):
        self.policy.train()
        counter = 0
        for epoch in range(self.num_repeat):
            for problem in self.problems:
                if counter > self.num_train:
                    break
                counter += 1

                temp_features = []

                model = Model("setcover")
                model.setRealParam('limits/time', self.time_limit)
                ourNodeSel = self.nodesel(model, self.policy, dataset=temp_features)
                model.includeNodesel(ourNodeSel, "nodesel", "My node selection", 999999, 999999)
                model.readProblem(problem)
                print(model.getStatus())
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
                    print("SCIPPING HERE")
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

                samples = list(zip(self.sfeature_list, self.soracle))[-1500:]
                if self.isScippable():
                    continue
                s_loader = DataLoader(samples, batch_size=1, shuffle=True)
                for epoch in range(self.num_epoch):
                    running_loss = 0.0
                    for i, (feature, label) in enumerate(s_loader):
                        self.optimizer.zero_grad()
                        output = self.policy(feature)
                        loss = self.loss(output, label)
                        loss.backward()
                        self.optimizer.step()
                        running_loss += loss.item()
                    print('[%d] loss: %.3f' %
                          (epoch + 1, running_loss / len(s_loader)))
                    running_loss = 0.0

                if os.path.exists(self.save_path):
                    os.remove(self.save_path)
                torch.save(self.policy.state_dict(), self.save_path)
    def test(self, problems):
        return super().test(problems, self.nodesel)

class TreeDagger(Dagger):
    def __init__(self, selector, problem_dir, device, num_train=None, num_epoch = 1, batch_size=5, save_path=None, num_repeat=1):
        print(num_repeat)
        super().__init__(selector, problem_dir, device, nn.CrossEntropyLoss(), num_train, num_epoch, batch_size, save_path=save_path)
        self.nodesel = MyNodesel
        self.num_repeat = num_repeat
        self.time_limit = 600
    def train(self):
        self.policy.train()
        counter = 0
        for epoch in range(self.num_repeat):
            for problem in self.problems:
                if counter > self.num_train:
                    break
                counter += 1
                print(problem)
                temp_features = []
                torch.autograd.set_detect_anomaly(True)
                model = Model("setcover")
                step_ids = []
                ourNodeSel = self.nodesel(model, self.policy, dataset=temp_features, step_ids=step_ids)
                init_scip_params_haoran(model, 10)
                model.readProblem(problem)
                model.setRealParam('limits/time', self.time_limit)
                model.includeNodesel(ourNodeSel, "nodesel", "My node selection", 999999, 999999)
                model.optimize()

                self.listNNodes.append(model.getNNodes())
                print(self.listNNodes)

                optimal_node = None
                if len(ourNodeSel.tree.all_nodes()) < 2:
                    continue
                for node in ourNodeSel.tree.leaves():
                    if checkIsOptimal(node, model, ourNodeSel.tree):
                        optimal_node = node
                        break

                if optimal_node is not None:
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
                            self.debug.append((optimal_id, step_ids))
                            oracle_val = (step_ids[i]== optimal_id).type(torch.uint8).nonzero()[0][0]
                            self.soracle.append(oracle_val)
                            self.sfeature_list.append(temp_features[i])
                if self.isScippable():
                    continue
                samples = list(zip(self.sfeature_list, self.soracle, self.debug))[-1500:]

                s_loader = DataLoader(samples, batch_size=self.batch_size, shuffle=True, collate_fn=collate)
                print('Number of datapoints: %d' % (len(samples)))
                for epoch in range(self.num_epoch):
                    running_loss = 0.0
                    number_right = 0

                    for (bg, labels) in s_loader:
                        self.optimizer.zero_grad()
                        unbatched = dgl.unbatch(bg)
                        sizes = [torch.sum(unbatched[i].ndata['in_queue']) for i in range(len(unbatched))]
                        g = bg
                        n = g.number_of_nodes()
                        h_size = 14
                        h = torch.zeros((n, h_size))
                        c = torch.zeros((n, h_size))
                        iou = torch.zeros((n, 3 * h_size))

                        outputs, _ = self.policy(g, h, c, iou)
                        outputs = size_splits(outputs, sizes)
                        total_loss = None
                        for i in range(len(unbatched)):
                            output = outputs[i]
                            label = labels[i]
                            _, indices = torch.max(output, 0)
                            if indices.item() == label.item():
                                number_right += 1
                            output = output.unsqueeze(0)
                            label = label.unsqueeze(0)
                            loss = self.loss(output, label.to(device=self.device))
                            if total_loss == None:
                                total_loss = loss
                            else:
                                total_loss = total_loss + loss
                            running_loss += loss.item()
                        self.optimizer.zero_grad()
                        total_loss.backward()
                        self.optimizer.step()

                    print('[%d] loss: %.3f accuracy: %.3f number right: %d' %
                          (epoch + 1, running_loss / len(samples), number_right/len(samples), number_right))
                    running_loss = 0.0

                if os.path.exists(self.save_path):
                    os.remove(self.save_path)
                torch.save(self.policy.state_dict(), self.save_path)


    def test(self, problems):
        return super().test(problems, self.nodesel)

    def testAccuracy(self, problems):
        real_problems = glob.glob(problems + "/*.lp")

        with torch.no_grad():
            counter = 0
            for problem in real_problems:
                if counter > self.num_train:
                    break
                counter += 1
                print(problem)
                temp_features = []
                torch.autograd.set_detect_anomaly(True)
                model = Model("setcover")
                step_ids = []
                ourNodeSel = self.nodesel(model, self.policy, dataset=temp_features, step_ids=step_ids)
                # init_scip_params_haoran(model, 10)
                model.readProblem(problem)
                model.setRealParam('limits/time', self.time_limit)
                model.includeNodesel(ourNodeSel, "nodesel", "My node selection", 999999, 999999)
                model.optimize()

                self.listNNodes.append(model.getNNodes())
                print(self.listNNodes)

                optimal_node = None
                if len(ourNodeSel.tree.all_nodes()) < 2:
                    continue
                for node in ourNodeSel.tree.leaves():
                    if checkIsOptimal(node, model, ourNodeSel.tree):
                        optimal_node = node
                        break

                if optimal_node is not None:
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
                            self.debug.append((optimal_id, step_ids))
                            oracle_val = (step_ids[i] == optimal_id).type(torch.uint8).nonzero()[0][0]
                            self.soracle.append(oracle_val)
                            self.sfeature_list.append(temp_features[i])
        samples = list(zip(self.sfeature_list, self.soracle, self.debug))

        s_loader = DataLoader(samples, batch_size=self.batch_size, shuffle=True, collate_fn=collate)
        print('Number of datapoints: %d' % (len(samples)))

        running_loss = 0.0
        number_right = 0

        for (bg, labels) in s_loader:
            self.optimizer.zero_grad()
            unbatched = dgl.unbatch(bg)
            sizes = [torch.sum(unbatched[i].ndata['in_queue']) for i in range(len(unbatched))]
            g = bg
            n = g.number_of_nodes()
            h_size = 14
            h = torch.zeros((n, h_size))
            c = torch.zeros((n, h_size))
            iou = torch.zeros((n, 3 * h_size))

            outputs, _ = self.policy(g, h, c, iou)
            outputs = size_splits(outputs, sizes)
            total_loss = None
            for i in range(len(unbatched)):
                output = outputs[i]
                label = labels[i]
                _, indices = torch.max(output, 0)
                if indices.item() == label.item():
                    number_right += 1


        print('[%d] loss: %.3f accuracy: %.3f number right: %d' %
              (0, running_loss / len(samples), number_right / len(samples), number_right))


class branchDagger(Dagger):
    def __init__(self, selector, problem_dir, device, time_limit=1500, num_train=None, num_epoch = 1, batch_size=5, save_path=None, num_repeat=1):
        super().__init__(selector, problem_dir, device, nn.MSELoss(), num_train, num_epoch, batch_size, save_path=save_path)
        self.time_limit = time_limit

    def train(self):
        for epoch in range(self.num_repeat):
            for problem in self.problems:
                print(problem)
                model = Model("setcover")
                model.readProblem(problem)
                model.setRealParam('limits/time', self.time_limit)
                myBranch = TreeBranch(model, self.policy, dataset = self.sfeature_list, train=True)
                init_scip_params(model, 100, False, False, False, False, False, False)

                model.setBoolParam("branching/vanillafullstrong/donotbranch", True)
                model.setBoolParam('branching/vanillafullstrong/idempotent', True)
                model.includeBranchrule(myBranch, "ImitationBranching", "Policy branching on variable",
                                        priority=99999, maxdepth=-1, maxbounddist=1)
                model.optimize()

                if self.isScippable():
                    continue
                for epoch in range(self.num_epoch):
                    running_loss = 0.0
                    number_right = 0
                    print(len(self.sfeature_list))
                    for (dgltree, branch_cand, default_gains, label_index, label) in self.sfeature_list:
                        self.optimizer.zero_grad()

                        h_size = 14
                        n = dgltree.number_of_nodes()
                        h = torch.zeros((n, h_size))
                        c = torch.zeros((n, h_size))
                        iou = torch.zeros((n, 3 * h_size))

                        best_var, tree_scores, down_scores, up_scores, vars = self.policy(dgltree, h, c, iou, branch_cand, default_gains)
                        best_values = torch.tensor([100000 if label_index == var else 0 for var in vars], dtype=torch.float)

                        best_in = np.argmax(tree_scores.detach().numpy())
                        if label == best_in:
                            number_right += 1


                        tree_scores = tree_scores.unsqueeze(0)


                        loss = self.loss(down_scores, best_values)

                        loss.backward()

                        self.optimizer.step()

                        running_loss += loss.item()

                    print('[%d] loss: %.3f accuracy: %.3f number right: %d' %
                          (epoch + 1, running_loss / len(dataset), number_right / len(dataset), number_right))

                #
                # if self.train and not pre_calculated:
                #     best_in = np.argmax(tree_scores.detach().numpy())
                #     if label == best_in:
                #         self.num_right += 1
                #     self.optimizer.zero_grad()
                #     tree_scores = tree_scores.unsqueeze(0)
                #     label = torch.tensor(label).unsqueeze(0)
                #     loss = self.loss(tree_scores, label)
                #
                #     self.total_loss *= self.num_example
                #     self.total_loss += loss.item()
                #     self.num_example += 1
                #     self.total_loss = self.total_loss / self.num_example
                #     loss.backward()
                #     self.optimizer.step()
                #
                # if self.train:
                #     if os.path.exists(self.save_path):
                #         os.remove(self.save_path)
                #     torch.save(self.policy.state_dict(), self.save_path)

    def test(self, problems):
        with torch.no_grad():
            real_problems = glob.glob(problems + "/*.lp")
            num_nodes = []
            default = []
            for problem in real_problems:
                print(problem)
                model = Model("setcover")
                ourNodeSel = MyNodesel(model, self.policy)
                model.readProblem(problem)
                myBranch = TreeBranch(model, self.policy, train=False)
                init_scip_params(model, 100, False, False, False, False, False, False)

                model.setBoolParam("branching/vanillafullstrong/donotbranch", True)
                model.setBoolParam('branching/vanillafullstrong/idempotent', True)
                model.includeBranchrule(myBranch, "ImitationBranching", "Policy branching on variable",
                                        priority=99999, maxdepth=-1, maxbounddist=1)
                model.optimize()
                num_nodes.append(model.getNNodes())
            for problem in real_problems:
                print(problem)
                model = Model("setcover")
                model.setRealParam('limits/time', self.time_limit)
                model.readProblem(problem)
                model.optimize()
                default.append(model.getNNodes())
        return num_nodes, default 