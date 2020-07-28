from pyscipopt import Model, Heur, quicksum, multidict, SCIP_RESULT, SCIP_HEURTIMING, SCIP_PARAMSETTING, Sepa, \
    Branchrule
import numpy as np
from treelib import Tree
import networkx as nx
import dgl
import copy
from nodeutil import nodeData, getNodeFeature, _build_tree, setRoot
import faulthandler
faulthandler.enable()
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np

class TreeBranch(Branchrule):
    def __init__(self, model, policy, loss=nn.CrossEntropyLoss(), device=torch.device('cpu'), dataset=None, mu = .5, train=True):
        self.model = model
        self.tree = Tree()
        self.nodeToBounds = {}
        self.mu = mu
        self.policy = policy
        self.indexToVar = {}
        self.dataset = dataset
        self.goneDown = {}
        self.root_id = None
        self.loss = loss
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.device = device
        self.total_loss = 0
        self.num_example = 0
        self.save_path = "branch_lstm.pt"
        self.num_right = 0
        self.train = train
    def branchinit(self):
        self.root_buffer = {}

    def calcLSTMFeatures(self, g):
        h_size = 14
        n = g.number_of_nodes()
        h = torch.zeros((n, h_size))
        c = torch.zeros((n, h_size))
        iou = torch.zeros((n, 3 * h_size))

        return self.policy(g, h, c, iou, self.model.getLPBranchCands())
    def branchexeclp(self, allowaddcons):
        assert allowaddcons
        branch_cands = self.model.getLPBranchCands()
        nbranch_cand = len(branch_cands[0])
        curr_node = self.model.getCurrentNode()


        curr_tree_node = None

        if curr_node != None:
            number = curr_node.getNumber()
            if self.tree.size() == 0 or curr_node.getParent() == None:
                self.tree = Tree()
                self.tree.create_node(number, number, data=nodeData(curr_node, self.model.getLPObjVal(), self.model,
                                                                    lp_obj_val=self.model.getLPObjVal()))
                self.nodeToBounds[curr_node] = ([], [], [])
                self.root_id = number
            else:
                variables, branch_bounds, bound_types = curr_node.getParentBranchings()
                parent_node = curr_node.getParent()
                parent_num = parent_node.getNumber()
                parent_variables, parent_bb, parent_bt = self.nodeToBounds[parent_node]
                curr_variables = list(parent_variables) + variables
                curr_bb = list(parent_bb) + branch_bounds
                curr_bt = list(parent_bt) + bound_types
                self.nodeToBounds[curr_node] = (curr_variables, curr_bb, curr_bt)

                self.tree.create_node(number, number, parent=parent_num,
                                      data=nodeData(curr_node, self.model.getLPObjVal(), self.model,
                                                    variables=variables,
                                                    bound_types=bound_types, branch_bounds=branch_bounds,
                                                    lp_obj_val=self.model.getLPObjVal()))

            curr_tree_node = self.tree.get_node(number)

        if self.tree.size() != 0:
            setRoot(self.tree, curr_node.getNumber())

            g = _build_tree(self.tree, self.model)
            dgltree = dgl.DGLGraph()
            dgltree.from_networkx(g, node_attrs=["feature", "node_id", "in_queue", "variable_chosen",
                                                 "scaled_improvement_down", "scaled_improvement_up"])
            best_var, tree_scores = self.calcLSTMFeatures(dgltree)

        else:
            best_var = branch_cands[0][0]


        curr_tree_node.data.variable_chosen = best_var.getIndex()
        if curr_node.getParent() != None:
            parent_node = self.tree.get_node(curr_node.getParent().getNumber())
            if self.goneDown[curr_node.getNumber()] == True:
                parent_node.data.calc_down_improvements(self.model.getLPObjVal(), best_var)
            else:
                parent_node.data.calc_up_improvements(self.model.getLPObjVal(), best_var)

        if self.train:
            self.model.executeBranchRule('vanillafullstrong', allowaddcons)
            cands_, scores, npriocands, label, result = self.model.getVanillafullstrongData()

        self.branching(best_var)

        if self.train:
            best_in = np.argmax(tree_scores.detach().numpy())
            if label == best_in:
                self.num_right += 1
            self.optimizer.zero_grad()
            tree_scores = tree_scores.unsqueeze(0)
            label = torch.tensor(label).unsqueeze(0)
            loss = self.loss(tree_scores, label)

            self.total_loss *= self.num_example
            self.total_loss += loss.item()
            self.num_example += 1
            self.total_loss = self.total_loss / self.num_example
            loss.backward()
            self.optimizer.step()

        if self.train:
            if os.path.exists(self.save_path):
                os.remove(self.save_path)
            torch.save(self.policy.state_dict(), self.save_path)
        return {'result': SCIP_RESULT.BRANCHED}


    def branching(self, variable):
        down, eq, up = self.model.branchVarVal(variable, variable.getLPSol())
        self.goneDown[down.getNumber()] = True
        self.goneDown[up.getNumber()] = False
        self.model.chgVarLbNode(up, variable, int(np.ceil(variable.getLPSol())))
        self.model.chgVarUbNode(down, variable, int(np.floor(variable.getLPSol())))


