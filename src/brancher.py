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
from utilities import probing_features_extraction
from copy import deepcopy
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
        self.optimizer = optim.Adam(self.policy.parameters(), lr= 1e-3)
        self.device = device
        self.total_loss = 0
        self.num_example = 0
        self.save_path = "branch_lstm.pt"
        self.num_right = 0
        self.default_gains = None
        self.train = train
        self.counter = 0
        self.node_numbers = set()
        self.num_cutoff = 0
        self.num_reddom = 0
    def branchinit(self):
        self.root_buffer = {}

    def calcLSTMFeatures(self, g):
        h_size = 14
        n = g.number_of_nodes()
        h = torch.zeros((n, h_size))
        c = torch.zeros((n, h_size))
        iou = torch.zeros((n, 3 * h_size))

        return self.policy(g, h, c, iou, self.cand_indeces, self.default_gains)
    def branchexeclp(self, allowaddcons):

        assert allowaddcons
        self.counter += 1
        branch_cands = self.model.getLPBranchCands()
        self.cand_indeces = [cand.getIndex() for cand in branch_cands[0]]
        nbranch_cand = len(branch_cands[0])
        curr_node = self.model.getCurrentNode()
        pre_calculated = False

        if self.counter > 2 and self.counter < 5 :
            best_var, self.default_gains = self.getFullStrong()
            pre_calculated = True
        else:
            self.default_gains = [(1,1)] * nbranch_cand
        curr_tree_node = None

        variables_values = {}
        for var in self.model.getVars():
            variables_values[var.getIndex()] = (var.getLbLocal(), var.getUbLocal())

        if curr_node != None and not pre_calculated:
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
            self.node_numbers.add(number)
        if self.tree.size() != 0 and not pre_calculated:
            setRoot(self.tree, curr_node.getNumber())

            g = _build_tree(self.tree, self.model)
            dgltree = dgl.DGLGraph()
            dgltree.from_networkx(g, node_attrs=["feature", "node_id", "in_queue", "variable_chosen",
                                                 "scaled_improvement_down", "scaled_improvement_up"])
            best_var, tree_scores, _, _, _ = self.calcLSTMFeatures(dgltree)

        else:
            best_var = 0

        if not pre_calculated:
            curr_tree_node.data.variable_chosen = branch_cands[0][best_var].getIndex()
            if curr_node.getParent() != None:
                parent_node = self.tree.get_node(curr_node.getParent().getNumber())
                if self.goneDown[curr_node.getNumber()] == True:
                    parent_node.data.calc_down_improvements(self.model.getLPObjVal(), branch_cands[0][best_var])
                else:
                    parent_node.data.calc_up_improvements(self.model.getLPObjVal(), branch_cands[0][best_var])

        if self.train and not pre_calculated:
            self.model.executeBranchRule('vanillafullstrong', allowaddcons)
            cands_, scores, npriocands, label, result = self.model.getVanillafullstrongData()
            self.dataset.append(
                (dgltree, deepcopy(self.cand_indeces), deepcopy(self.default_gains), branch_cands[0][label].getIndex(), label))

        self.branching(branch_cands[0][best_var])
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
        return {'result': SCIP_RESULT.BRANCHED}

    def getFullStrong(self):
        best_score = -1 * self.model.infinity()
        result = -1 * self.model.infinity()
        self.branch_cand = self.model.getLPBranchCands()
        self.nprio_lpcands = self.branch_cand[-2]
        lp_objval = self.model.getLPObjVal()

        cols = self.model.getLPColsData()
        cons = self.model.getConss()

        # initialize the sampling data
        up_sols = torch.zeros((len(cols), 1))
        down_sols = torch.zeros((len(cols), 1))
        up_var_UBs = torch.zeros((len(cols), 1))
        up_var_LBs = torch.zeros((len(cols), 1))
        down_var_UBs = torch.zeros((len(cols), 1))
        down_var_LBs = torch.zeros((len(cols), 1))
        branch_scores = []
        best_cand = None
        data = []
        total_gains = []
        for i in range(self.nprio_lpcands):

            variable = self.branch_cand[0][i]
            curr_node = self.model.getCurrentNode()
            up_gain, up_score, up_sol, up_var_UB, up_var_LB = \
                probing_features_extraction(self.model, i, self.branch_cand, rounding_direction='up')

            down_gain, down_score, down_sol, down_var_UB, down_var_LB = \
                probing_features_extraction(self.model, i, self.branch_cand, rounding_direction='down')

            # frac_data = model.getVarStrongbranchFrac(self.branch_cand[0][i], 1)

            # cutOff and domainRed
            # if up_score == self.model.infinity() or down_score == self.model.infinity():
            #     if up_score == self.model.infinity() and down_score == self.model.infinity():
            #         # cutoff
            #         self.num_cutoff += 1
            #         break
            #     elif down_score == self.model.infinity():
            #         self.model.tightenVarLb(self.branch_cand[0][i], int(np.ceil(self.branch_cand[1][i])), force=True)
            #         self.num_reddom += 1
            #         break
            #     else:
            #         self.model.tightenVarUb(self.branch_cand[0][i], int(np.floor(self.branch_cand[1][i])), force=True)
            #         self.num_reddom += 1
            #         break
            gains = [down_gain, up_gain, 0]
            scaled_gains = [down_gain/(variable.getLPSol() - int(np.floor(variable.getLPSol()))), up_gain/(int(np.ceil(variable.getLPSol())) - variable.getLPSol()), 0]
            total_gains.append(scaled_gains)
            branch_score = self.model.getBranchScoreMultiple(self.branch_cand[0][i], 2, gains)

            if branch_score > best_score:
                best_cand = i
                best_score = branch_score

            # Take strong branching feature here
            up_sols = torch.cat((up_sols, up_sol), 1)
            down_sols = torch.cat((down_sols, down_sol), 1)
            up_var_UBs = torch.cat((up_var_UBs, up_var_UB), 1)
            up_var_LBs = torch.cat((up_var_LBs, up_var_LB), 1)
            down_var_LBs = torch.cat((down_var_LBs, down_var_LB), 1)
            down_var_UBs = torch.cat((down_var_UBs, down_var_UB), 1)
            branch_scores.append(branch_score)
            data.append((variable,i,scaled_gains))

        for (variable, i, scaled_gains) in data:
            curr_node = self.model.getCurrentNode()
            if i != best_cand:
                number = min(self.node_numbers) - 1
                self.node_numbers.add(number)
            else:
                number = curr_node.getNumber()
                self.node_numbers.add(number)

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
                                                    lp_obj_val=self.model.getLPObjVal(),
                                                    variable_chosen=variable.getIndex(),
                                                    scaled_improvement_down=scaled_gains[0],
                                                    scaled_improvement_up=scaled_gains[1],
                                      ))


        return best_cand, total_gains


    def branching(self, variable):
        down, eq, up = self.model.branchVarVal(variable, variable.getLPSol())
        self.goneDown[down.getNumber()] = True
        self.goneDown[up.getNumber()] = False
        self.model.chgVarLbNode(up, variable, int(np.ceil(variable.getLPSol())))
        self.model.chgVarUbNode(down, variable, int(np.floor(variable.getLPSol())))


