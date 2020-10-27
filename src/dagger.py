import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import pickle
import torch.sparse
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.sparse
import networkx as nx
import dgl
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
from NodeSel import MyNodesel, LinNodesel
from utilities.nodeutil import getListOptimalID, checkIsOptimal
from pyscipopt import Model, Heur, quicksum, multidict, SCIP_RESULT, SCIP_HEURTIMING, SCIP_PARAMSETTING, Sepa, \
    Branchrule, Nodesel
import glob
from utilities.utilities import init_scip_params, init_scip_params_haoran, personalize_scip
from brancher import TreeBranch

import os

os.chdir("../")
torch.set_printoptions(precision=10)
def intersperse(lst, item):
    result = [item] * (len(lst) * 2 - 1)
    result[0::2] = lst
    return result

def plot_tree(g):
    # this plot requires pygraphviz package
    pos = nx.nx_agraph.graphviz_layout(g, prog='dot')
    nx.draw(g, pos, with_labels=True, node_size=50,
            node_color=[[.5, .5, .5]], arrowsize=4)
    plt.show()

def collate(samples):
    graphs, labels, _, weight = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels), weight

def collate_undebug(samples):
    graphs, labels, weight = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels), weight

def position_in_array(count, arr):
    for i in range(len(arr)):
        if count < sum(arr[0:i+1]):
            return i

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
    def __init__(self, selector, problem_dir, device, loss, num_train=None, num_epoch = 3, num_repeat=1, batch_size=5, save_path=None, problem_type="lp"):
        self.policy = selector
        self.save_path = save_path
        self.problem_dir = os.path.join(os.getcwd() , problem_dir)
        self.problem_type = problem_type

        self.problems = glob.glob(os.path.join(problem_dir, "/*." + self.problem_type))
        if num_train is None:
            self.num_train = len(self.problems)
        else:
            self.num_train = num_train
        self.model = Model("setcover")
        self.sfeature_list = []
        self.weights = []
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
        self.description = None

    def setDescription(self, text):
        self.description = text

    def isScippable(self):
        if self.num_features == len(self.sfeature_list):
            return True
        else:
            return False

        self.num_features = len(self.sfeature_list)

    def test(self, problems, MyNodesel):
        with torch.no_grad():
            real_problems = glob.glob(problems + "/*." + self.problem_type)
            num_nodes = []
            solving_times_us = []
            solving_times_def = []
            default = []
            for problem in real_problems:
                print(problem)
                model = Model("setcover")
                ourNodeSel = MyNodesel(model, self.policy)
                model.readProblem(problem)
                model.setIntParam('timing/clocktype', 2)
                model.setRealParam('limits/time', self.time_limit)
                model.includeNodesel(ourNodeSel, "nodesel", "My node selection", 999999, 999999)
                personalize_scip(model, 10)
                model.optimize()
                solving_times_us.append(model.getSolvingTime())
                num_nodes.append(model.getNNodes())
            for problem in real_problems:
                print(problem)
                model = Model("setcover")
                model.setIntParam('timing/clocktype', 2)
                model.setRealParam('limits/time', self.time_limit)
                model.readProblem(problem)
                personalize_scip(model, 10)
                model.optimize()
                default.append(model.getNNodes())
                solving_times_def.append(model.getSolvingTime())

        self.write_to_log_file("Test", self.problem_dir, -1, -1, def_nodes=default)

        return num_nodes, default, solving_times_us, solving_times_def

    def write_to_log_file(self, type, data_path, accuracy, loss, def_nodes=None):

        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(now)
        if self.description is not None:
            log = ("%s: Type: %s, Model Name: %s, Data Path: %s, Accuracy: %.2f, Loss: %.2f, Description: %s" % (
            dt_string, type, self.model_name, data_path, 100 * accuracy, loss, self.description))
        else:
            log = ("%s: Type: %s, Model Name: %s, Data Path: %s, Accuracy: %.2f, Loss: %.2f \n" % (dt_string, type, self.model_name, data_path, 100 * accuracy, loss))
        if self.listNNodes is not None and len(self.listNNodes) > 0:

            log = log + ", NumNodes: " + ''.join(intersperse([str(v) for v in self.listNNodes], ","))
        if def_nodes is not None:
            log += ", Default: " + ''.join(intersperse([str(v) for v in def_nodes], ","))
        log = log + "\n"
        file_object = open('log/log.txt', 'a')
        file_object.write(log)
        file_object.close()

    def test(self, problems):
        with torch.no_grad():
            real_problems = glob.glob(problems + "/*." + self.problem_type)
            num_nodes = []
            default = []
            for problem in real_problems:
                print(problem)
                self.solveModel(problem, to_train=False)
                num_nodes.append(self.model.getNNodes())
                self.listNNodes = num_nodes
            for problem in real_problems:
                print(problem)
                self.solveModel(problem, to_train=False, default=True)
                default.append(self.model.getNNodes())
        self.write_to_log_file("Test", problems, -1, -1, def_nodes=default)

        return num_nodes, default



class RankDagger(Dagger):
    def __init__(self, selector, problem_dir, device, num_train=None, num_epoch=1, time_limit=200, save_path=None, num_repeat=1):
        super().__init__(selector, problem_dir, device, nn.MSELoss(), num_train=num_train, num_epoch=num_epoch, save_path=save_path, num_repeat=num_repeat)
        self.nodesel = LinNodesel
        self.time_limit = time_limit
        self.model_name = "RankDagger"
    def solveModel(self,problem, train=True, default=False):
        temp_features = []
        self.model = Model("setcover")
        self.model.setRealParam('limits/time', self.time_limit)
        if not default:
            self.ourNodeSel = self.nodesel(self.model, self.policy, dataset=temp_features)
            self.model.includeNodesel(self.ourNodeSel, "nodesel", "My node selection", 999999, 999999)
        self.model.readProblem(problem)
        self.model.optimize()
        return temp_features

    def addTreeData(self, temp_features,num_past = 1500):
        num_right = 0
        num_cases = 0
        optimal_node = None
        # ourNodeSel.tree.show(data_property="variables")
        for node in self.ourNodeSel.tree.leaves():
            if checkIsOptimal(node, self.model, self.ourNodeSel.tree):
                optimal_node = node
                print("FOUND OPTIMal")

        if optimal_node is None:
            return None, 0, 0

        optimal_ids = getListOptimalID(optimal_node.identifier, self.ourNodeSel.tree)

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

            for_accuracy_feat = [idToFeature[id] for id in ids]
            labels = [1 if id in optimal_ids else 0 for id in ids]
            if 1 not in labels:
                continue
            best_node = labels.index(1)

            with torch.no_grad():
                vals = [self.policy(feature) for feature in for_accuracy_feat]
                true_best_node = np.argmax(vals)
            if best_node == true_best_node:
                num_right += 1
            num_cases += 1
            samples = list(zip(self.sfeature_list, self.soracle))[-1 * num_past:]
        return samples, num_right, num_cases

    def train(self):
        self.policy.train()
        self.feature_set_for_accuracy = []
        counter = 0
        total_num_cases = 0
        total_num_right = 0
        total_num_predict = 0
        average_loss = 0
        for epoch in range(self.num_repeat):
            for problem in self.problems:
                temp_features = self.solveModel(problem)
                self.listNNodes.append(self.model.getNNodes())
                print(self.listNNodes)
                samples, num_right, num_cases_predict = self.addTreeData(temp_features)
                if samples is None:
                    continue
                total_num_right += num_right
                total_num_predict += num_cases_predict
                if self.isScippable():
                    continue
                s_loader = DataLoader(samples, batch_size=1, shuffle=True)
                for epoch in range(self.num_epoch):
                    running_loss = 0.0
                    for i, (feature, label) in enumerate(s_loader):
                        self.optimizer.zero_grad()
                        output = self.policy(feature)
                        loss = self.loss(output, label.to(device=self.device))
                        loss.backward()
                        self.optimizer.step()
                        running_loss += loss.item()
                    print('[%d] loss: %.3f' %
                          (epoch + 1, running_loss / len(s_loader)))
                    average_loss += running_loss
                    total_num_cases += len(samples)

                if os.path.exists(self.save_path):
                    os.remove(self.save_path)
                torch.save(self.policy.state_dict(), self.save_path)
        self.write_to_log_file("Train", self.problem_dir, total_num_right/total_num_predict, average_loss/total_num_cases)


    def testAccuracy(self, problems):
        real_problems = glob.glob(problems + "/*." + self.problem_type)
        total_num_cases = 0
        total_num_right = 0
        for problem in real_problems:
            temp_features = self.solveModel(problem)
            self.listNNodes.append(self.model.getNNodes())
            print(self.listNNodes)
            samples, num_right, num_cases = self.addTreeData(temp_features, num_past=0)
            total_num_cases += num_cases
            total_num_right += num_right
        s_loader = DataLoader(samples, batch_size=1, shuffle=True)

        running_loss = 0.0
        for i, (feature, label) in enumerate(s_loader):
            output = self.policy(feature)
            loss = self.loss(output, label.to(device=self.device))
            running_loss += loss.item()
        print('[%d] loss: %.3f' %
              (0, running_loss / len(s_loader)))

        self.write_to_log_file("Test", problems, total_num_right/total_num_cases, running_loss / len(s_loader))

class TreeDagger(Dagger):
    def __init__(self, selector, problem_dir, device, val_dir, num_train=None, num_epoch = 1, batch_size=5, save_path=None, num_repeat=1, problem_type="lp"):
        super().__init__(selector, problem_dir, device, nn.CrossEntropyLoss(), num_train, num_epoch, batch_size, save_path=save_path, problem_type=problem_type)
        self.nodesel = MyNodesel
        self.num_repeat = num_repeat
        self.time_limit = 60
        self.model_name = "TreeDagger"
        self.val_dir = val_dir

    def validate(self):
        real_problems = glob.glob(self.val_dir + "/*." + self.problem_type)
        number_right = 0
        num_problems = 0
        nodes_needed = 0
        with torch.no_grad():
            for problem in real_problems:

                temp_features, step_ids, ourNodeSel = self.solveModel(problem)
                self.listNNodes.append(self.model.getNNodes())
                nodes_needed += self.model.getNNodes()
                if len(ourNodeSel.tree.all_nodes()) < 2:
                    continue

                samples = self.addTreeData(ourNodeSel, temp_features, step_ids)

                if self.isScippable():
                    continue

                s_loader = DataLoader(samples, batch_size=self.batch_size, shuffle=False, collate_fn=collate)
                num_problems += 1

                for (bg, labels, weights) in s_loader:
                    self.optimizer.zero_grad()
                    unbatched, outputs = self.compute(bg)
                    for i in range(len(unbatched)):
                        output = outputs[i]
                        label = labels[i]
                        _, indices = torch.max(output, 0)
                        if indices.item() == label.item():
                            number_right += 1 / len(samples)
        return number_right/num_problems, nodes_needed


    def solveModel(self, problem, default=False, to_train=True):
        temp_features = []
        torch.autograd.set_detect_anomaly(True)
        self.model = Model("indset")
        self.model.hideOutput()
        step_ids = []
        ourNodeSel = None

        if not default:
            if to_train:
                ourNodeSel = self.nodesel(self.model, self.policy, dataset=temp_features, step_ids=step_ids)
                self.model.includeNodesel(ourNodeSel, "nodesel", "My node selection", 999999, 999999)
            else:
                ourNodeSel = self.nodesel(self.model, self.policy)
                self.model.includeNodesel(ourNodeSel, "nodesel", "My node selection", 999999, 999999)

        self.model.setRealParam('limits/time', self.time_limit)
        personalize_scip(self.model, 10)

        self.model.readProblem(problem)
        self.model.optimize()

        torch.cuda.empty_cache()

        return temp_features, step_ids, ourNodeSel

    def addTreeData(self, ourNodeSel, temp_features, step_ids):
        self.debug = []
        self.soracle = []
        self.sfeature_list = []
        self.weights = []

        optimal_node = None
        for node in ourNodeSel.tree.leaves():
            if checkIsOptimal(node, self.model, ourNodeSel.tree):
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
                    self.weights.append(1/len(temp_features))

        for i in range(len(self.weights)):
            self.weights[i] = 1/len(self.weights)

        samples = list(zip(self.sfeature_list, self.soracle, self.debug, self.weights))

        return samples

    def compute(self, bg):
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

        return unbatched, outputs

    def train(self):
        self.policy.train()
        torch.cuda.empty_cache()
        counter = 0
        problems = glob.glob(self.problem_dir + "/*." + self.problem_type)
        print(self.problem_dir)
        print(problems)
        for total_epoch in range(self.num_repeat):
            for problem in problems:
                torch.cuda.empty_cache()

                counter += 1
                try:
                    temp_features, step_ids, ourNodeSel = self.solveModel(problem)
                except:
                    continue
                self.listNNodes.append(self.model.getNNodes())

                if len(ourNodeSel.tree.all_nodes()) < 2:
                    continue

                samples = self.addTreeData(ourNodeSel, temp_features, step_ids)

                if len(samples) == 0:
                    continue

                s_loader = DataLoader(samples, batch_size=self.batch_size, shuffle=True, collate_fn=collate)
                for epoch in range(self.num_epoch):
                    for (bg, labels, weights) in s_loader:
                        self.optimizer.zero_grad()

                        unbatched, outputs = self.compute(bg)
                        total_loss = None
                        for i in range(len(unbatched)):
                            output = outputs[i]
                            label = labels[i]

                            _, indices = torch.max(output, 0)
                            output = output.unsqueeze(0)
                            label = label.unsqueeze(0)
                            loss = self.loss(output, label.to(device=self.device))
                            if total_loss == None:
                                total_loss = loss
                            else:
                                total_loss = total_loss + loss

                        self.optimizer.zero_grad()
                        total_loss.backward()
                        self.optimizer.step()
                    torch.cuda.empty_cache()

                if os.path.exists(self.save_path):
                    os.remove(self.save_path)
                torch.save(self.policy.state_dict(), self.save_path)
                if counter % 10 == 0:
                    val_accuracy, nodes_needed = self.validate()
                    print('[%d] loss: %.3f accuracy: %.3f nodes needed: %d' %
                          (total_epoch + 1, 0, val_accuracy, nodes_needed))

        self.write_to_log_file("Train", self.problem_dir, val_accuracy, 0)

    def testAccuracy(self, problems):
        real_problems = glob.glob(problems + "/*." + self.problem_type)
        number_right = 0
        num_problems = 0
        with torch.no_grad():
            for problem in real_problems:
                print(problem)

                temp_features, step_ids, ourNodeSel = self.solveModel(problem)
                self.listNNodes.append(self.model.getNNodes())
                print(self.listNNodes)

                if len(ourNodeSel.tree.all_nodes()) < 2:
                    continue

                samples = self.addTreeData(ourNodeSel, temp_features, step_ids, num_past=0)

                if self.isScippable():
                    continue

                s_loader = DataLoader(samples, batch_size=self.batch_size, shuffle=False, collate_fn=collate)
                num_problems += 1
                for (bg, labels, weights) in s_loader:
                    self.optimizer.zero_grad()
                    unbatched, outputs = self.compute(bg)
                    for i in range(len(unbatched)):
                        output = outputs[i]
                        label = labels[i]
                        _, indices = torch.max(output, 0)
                        if indices.item() == label.item():
                            number_right += 1/samples

            print('Accuracy %.2f' % (100 * number_right/num_problems))
        self.write_to_log_file("Test", problems, number_right/num_problems, 0)


class branchDagger(Dagger):
    def __init__(self, selector, problem_dir, device, time_limit=1500, num_train=None, num_epoch = 1, batch_size=5, save_path=None, num_repeat=1):
        super().__init__(selector, problem_dir, device, nn.MSELoss(), num_train, num_epoch, batch_size, save_path=save_path)
        self.time_limit = time_limit
        self.model_name = "BranchDagger"
        self.num_repeat = num_repeat

    def solveModel(self, problem, to_train=True, default=False):
        self.model = Model("setcover")
        self.model.readProblem(problem)
        self.model.setRealParam('limits/time', self.time_limit)
        if not default:
            myBranch = TreeBranch(self.model, self.policy, dataset=self.sfeature_list, train=to_train)
            self.model.includeBranchrule(myBranch, "ImitationBranching", "Policy branching on variable",
                                         priority=99999, maxdepth=-1, maxbounddist=1)
        personalize_scip(self.model, 100, False, False, False, False, False, False)

        self.model.setBoolParam("branching/vanillafullstrong/donotbranch", True)
        self.model.setBoolParam('branching/vanillafullstrong/idempotent', True)

        self.model.optimize()

    def compute(self, dgltree, branch_cand, default_gains, label_index):
        h_size = 14
        n = dgltree.number_of_nodes()
        h = torch.zeros((n, h_size))
        c = torch.zeros((n, h_size))
        iou = torch.zeros((n, 3 * h_size))

        best_var, tree_scores, down_scores, up_scores, vars = self.policy(dgltree, h, c, iou, branch_cand,
                                                                          default_gains)
        best_values = torch.tensor([100000 if label_index == var else 0 for var in vars], dtype=torch.float)

        best_in = np.argmax(tree_scores.detach().numpy())
        return best_values, best_in, down_scores, up_scores,

    def train(self):
        total_num_cases = 0
        total_num_right = 0
        average_loss = 0

        for epoch in range(self.num_repeat):
            for problem in self.problems:
                print(problem)
                self.solveModel(problem)
                self.listNNodes.append(self.model.getNNodes())
                print(self.listNNodes)
                if self.isScippable():
                    continue
                for epoch in range(self.num_epoch):
                    running_loss = 0.0
                    number_right = 0
                    for (dgltree, branch_cand, default_gains, label_index, label) in self.sfeature_list:
                        self.optimizer.zero_grad()
                        best_values, best_in, down_scores, up_scores = self.compute(dgltree, branch_cand, default_gains, label_index)
                        if label == best_in:
                            number_right += 1

                        loss = self.loss(down_scores, best_values)
                        loss.backward()
                        self.optimizer.step()
                        running_loss += loss.item()
                    average_loss += running_loss
                    total_num_cases += len(self.sfeature_list)
                    total_num_right += number_right
                    torch.cuda.empty_cache()
                    print('[%d] loss: %.3f accuracy: %.3f number right: %d' %
                          (epoch + 1, running_loss / len(self.sfeature_list), number_right / len(self.sfeature_list), number_right))

        self.write_to_log_file("train", self.problem_dir, total_num_right/total_num_cases, average_loss/total_num_cases)

    def testAccuracy(self, problems):
        with torch.no_grad():
            real_problems = glob.glob(problems + "/*." + self.problem_type)
            self.sfeature_list = []
            for problem in real_problems:
                print(problem)
                self.solveModel(problem)
                self.listNNodes.append(self.model.getNNodes())
                print(self.listNNodes)
                if self.isScippable():
                    continue
                running_loss = 0.0
                number_right = 0
            with torch.no_grad():
                for (dgltree, branch_cand, default_gains, label_index, label) in self.sfeature_list:

                    best_values, best_in, down_scores, up_scores = self.compute(dgltree, branch_cand, default_gains,
                                                                                label_index)
                    if label == best_in:
                        number_right += 1

                    loss = self.loss(down_scores, best_values)
                    running_loss += loss.item()

                print('[%d] loss: %.3f accuracy: %.3f number right: %d' %
                      (0, running_loss / len(self.sfeature_list), number_right / len(self.sfeature_list),
                       number_right))
                self.write_to_log_file("Test", problems, number_right / len(self.sfeature_list), running_loss / len(self.sfeature_list))

class tree_offline(TreeDagger):
    def __init__(self, selector, problem_dir, device, data_path, val_dir, num_train=None, num_epoch=1, batch_size=5, save_path=None,
                 num_repeat=1, ):
        super().__init__(selector, problem_dir, device, nn.CrossEntropyLoss(), num_train, num_epoch, batch_size,
                         )
        self.num_epoch = num_epoch
        self.save_path = save_path
        self.data_path = data_path
        self.model_name = "TreeOffline"
        self.val_dir = val_dir
    def compute(self, bg):
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
        torch.cuda.empty_cache()
        return unbatched, outputs

    def train(self):
        total_num_cases = 0
        total_num_right = 0
        average_loss = 0
        # s_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_undebug)
        for epoch in range(self.num_epoch):
            running_loss = 0.0
            number_right = 0
            total_weight = 0
            pickles = glob.glob(self.problem_dir + "/*.pkl")
            for sample in pickles:
                self.dataset = pickle.load(open( sample, "rb" ))
                if len(self.dataset) == 0:
                    continue
                try:
                    s_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_undebug)
                except:
                    continue

                for (bg, labels, weights) in s_loader:
                    self.optimizer.zero_grad()

                    unbatched, outputs = self.compute(bg)
                    total_loss = None
                    for i in range(len(unbatched)):
                        output = outputs[i]
                        label = labels[i]
                        weight = weights[i]
                        total_weight += weight
                        _, indices = torch.max(output, 0)
                        if indices.item() == label.item():
                            number_right += 1 * weight
                        output = output.unsqueeze(0)
                        label = label.unsqueeze(0)
                        loss = self.loss(output, label.to(device=self.device))
                        if total_loss == None:
                            total_loss = loss
                        else:
                            total_loss = total_loss + loss
                        running_loss += loss.item()
                        total_weight += weight
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()
                torch.cuda.empty_cache()
                average_loss += total_loss.item()
                total_num_cases += len(self.dataset)
                total_num_right += number_right
            val_accuracy = self.validate()
            print('[%d] loss: %.3f accuracy: %.3f number right: %.3f' %
                      (epoch + 1, running_loss/total_num_cases, val_accuracy, number_right))
            if os.path.exists(self.save_path):
                os.remove(self.save_path)
            torch.save(self.policy.state_dict(), self.save_path)

        self.write_to_log_file("Train", self.problem_dir, val_accuracy, average_loss/total_num_cases)

    def validate(self):
        pickles = glob.glob(self.val_dir + "/*.pkl")
        number_right = 0
        total_weight = 0
        for sample in pickles:
            self.dataset = pickle.load(open(sample, "rb"))
            if len(self.dataset) == 0:
                continue
            try:
                s_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                      collate_fn=collate_undebug)
            except:
                continue
            for (bg, labels, weights) in s_loader:
                self.optimizer.zero_grad()
                unbatched, outputs = self.compute(bg)
                total_loss = None
                for i in range(len(unbatched)):
                    output = outputs[i]
                    label = labels[i]
                    weight = weights[i]
                    total_weight += weight
                    _, indices = torch.max(output, 0)
                    if indices.item() == label.item():
                        number_right += 1 * weight
                    total_weight += weight
            torch.cuda.empty_cache()
        return number_right/total_weight
class rankOffline(RankDagger):
    def __init__(self, selector, problem_dir, device, num_train=None, num_epoch=7, time_limit=200, save_path=None, num_repeat=1):
        super().__init__(selector, problem_dir, device, num_train=num_train, num_epoch=num_epoch, save_path=save_path, num_repeat=num_repeat)
        self.nodesel = LinNodesel
        self.time_limit = time_limit
        self.model_name = "RankOffline"

    def train(self):
        counter = 0
        total_num_cases = 0
        total_num_right = 0
        total_num_predict = 0
        average_loss = 0

        samples = pickle.load(open("../data/instances/setcover/train_500r_1000c_0.05d_100mc_0se/sample_rank.pkl", "rb"))
        s_loader = DataLoader(samples, batch_size=1, shuffle=True)
        for epoch in range(self.num_epoch):
            running_loss = 0
            for i, (feature, label) in enumerate(s_loader):
                self.optimizer.zero_grad()
                output = self.policy(feature)
                loss = self.loss(output, label.to(device=self.device))
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print('[%d] loss: %.3f' %
                  (epoch + 1, running_loss / len(s_loader)))
        average_loss += running_loss
        total_num_cases += len(samples)

        if os.path.exists(self.save_path):
            os.remove(self.save_path)
        torch.save(self.policy.state_dict(), self.save_path)
        self.write_to_log_file("Train", self.problem_dir, total_num_right/total_num_predict, average_loss/total_num_cases)

