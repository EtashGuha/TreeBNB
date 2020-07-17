from pyscipopt import Model, Heur, quicksum, multidict, SCIP_RESULT, SCIP_HEURTIMING, SCIP_PARAMSETTING, Sepa, \
    Branchrule
import numpy as np
import random
from treelib import Tree
import networkx as nx
import dgl
import copy
from nodeutil import nodeData, getNodeFeature, _build_tree
import faulthandler
faulthandler.enable()
import matplotlib.pyplot as plt
import torch
class TreeBranch(Branchrule):
    def __init__(self, model, policy, mu = .5):
        self.model = model
        self.tree = Tree()
        self.nodeToBounds = {}
        self.mu = mu
        self.policy = policy
        self.indexToVar = {}
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
        branch_index = random.randint(0, nbranch_cand - 1)
        curr_node = self.model.getCurrentNode()

        # for index in range(len(branch_cands[0])):
        #     self.indexToVar[branch_cands[0][index].getIndex()] = branch_cands[0][index]

        if self.tree.size() != 0:
            g = _build_tree(self.tree, self.model)
            dgltree = dgl.DGLGraph()
            dgltree.from_networkx(g, node_attrs=["feature", "node_id", "in_queue", "variable_chosen",
                                                 "scaled_improvement_down", "scaled_improvement_up"])
            best_var, scores = self.calcLSTMFeatures(dgltree)
            scaled_improvement_down, scaled_improvement_up = self.branching(best_var)
        else:
            branch_index = random.randint(0, nbranch_cand - 1)
            scaled_improvement_down, scaled_improvement_up = self.branching(branch_cands[0][branch_index])





        if curr_node != None:
            number = curr_node.getNumber()
            if self.tree.size() == 0 or curr_node.getParent() == None:
                self.tree = Tree()
                self.tree.create_node(number, number, data=nodeData(curr_node, self.model.getLPObjVal(), self.model,
                                                                    variable_chosen=branch_cands[0][branch_index].getIndex(),
                                                                    scaled_improvement_down=scaled_improvement_down,
                                                                    scaled_improvement_up=scaled_improvement_up))
                self.nodeToBounds[curr_node] = ([], [], [])
            else:
                variables, branch_bounds, bound_types = curr_node.getParentBranchings()
                parent_node = curr_node.getParent()
                parent_num = parent_node.getNumber()
                parent_variables, parent_bb, parent_bt = self.nodeToBounds[parent_node]
                curr_variables = list(parent_variables) + variables
                curr_bb = list(parent_bb) + branch_bounds
                curr_bt = list(parent_bt) + bound_types
                self.nodeToBounds[curr_node] = (curr_variables, curr_bb, curr_bt)

                try:
                    self.tree.create_node(number, number, parent=parent_num,
                                          data=nodeData(curr_node, self.model.getLPObjVal(), self.model,
                                                        variables=variables,
                                                        bound_types=bound_types, branch_bounds=branch_bounds,
                                                        variable_chosen=branch_cands[0][branch_index].getIndex(),
                                                        scaled_improvement_down=scaled_improvement_down,
                                                        scaled_improvement_up=scaled_improvement_up))
                except:
                    pass

        return {'result': SCIP_RESULT.BRANCHED}
    def branching(self, variable):
        currObj = self.model.getCurrentNode().getEstimate()
        down, eq, up = self.model.branchVarVal(variable, variable.getLPSol())
        downObj = down.getEstimate()
        upObj = up.getEstimate()

        scaled_improvement_up = (upObj - currObj)/(int(np.ceil(variable.getLPSol())) - variable.getLPSol())
        scaled_improvement_down = (downObj - currObj)/(variable.getLPSol() - int(np.floor(variable.getLPSol())))



        self.model.chgVarLbNode(up, variable, int(np.ceil(variable.getLPSol())))
        self.model.chgVarUbNode(down, variable, int(np.floor(variable.getLPSol())))

        return scaled_improvement_down, scaled_improvement_up