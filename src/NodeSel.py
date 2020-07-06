from pyscipopt import Model, Heur, quicksum, multidict, SCIP_RESULT, SCIP_HEURTIMING, SCIP_PARAMSETTING, Sepa, \
    Branchrule, Nodesel

import torch
import torch.sparse
import torch.sparse
import itertools
from treelib import Tree
import networkx as nx
import dgl
import copy
from nodeutil import nodeData, getNodeFeature, _build_tree
import faulthandler
faulthandler.enable()
import matplotlib.pyplot as plt





class MyNodesel(Nodesel):
    def __init__(self, model, policy, dataset=None, step_ids=None):
        self.model = model
        self.policy = policy
        self.tree = Tree()
        self.dataset = dataset
        self.step_ids = step_ids
        self.tempdgltree = dgl.DGLGraph()
        self.cou = 1
        self.probs = None
        self.ids = None

    def calcLSTMFeatures(self, g):
        h_size = 14
        n = g.number_of_nodes()
        h = torch.zeros((n, h_size))
        c = torch.zeros((n, h_size))
        iou = torch.zeros((n, 3 * h_size))

        return self.policy(g, h, c, iou)

    def nodeselect(self):
        '''first method called in each iteration in the main solving loop. '''
        # this method needs to be implemented by the user
        listOfNodes = list(itertools.chain.from_iterable(self.model.getOpenNodes()))
        curr_node = self.model.getCurrentNode()
        if curr_node != None:
            number = curr_node.getNumber()
            if self.tree.size() == 0 or curr_node.getParent() == None:
                self.tree = Tree()
                self.tree.create_node(number, number, data=nodeData(curr_node, self.model.getLPObjVal(), self.model))
            else:
                variables, branch_bounds, bound_types = curr_node.getParentBranchings()
                parent_node = curr_node.getParent()
                parent_num = parent_node.getNumber()
                try:
                    self.tree.create_node(number, number, parent=parent_num,
                                          data=nodeData(curr_node, self.model.getLPObjVal(), self.model, variables=variables,
                                                        bound_types=bound_types, branch_bounds=branch_bounds))
                except:
                    pass

        if self.tree.size() != 0:
            g = _build_tree(self.tree, self.model)
        else:
            g = nx.DiGraph()

        self.id_to_node = dict()
        for node in listOfNodes:

            num = node.getNumber()
            parent = node.getParent()
            g.add_node(node.getNumber(), feature=getNodeFeature(node, self.model), node_id=node.getNumber(), in_queue=1)
            self.id_to_node[node.getNumber()] = node
            if parent is not None:
                g.add_edge(parent.getNumber(), num)

        dgltree = dgl.DGLGraph()
        dgltree.from_networkx(g, node_attrs=["feature", "node_id", "in_queue"])

        # To calculate node features via lstm
        with torch.no_grad():
            probs, ids = self.calcLSTMFeatures(dgltree)

        self.probs = probs
        self.ids = ids
        if self.step_ids is not None and self.dataset is not None:
            self.step_ids.append(ids)
            self.dataset.append(dgltree)

        if len(probs) != 0:

            _, indices = torch.max(probs, 0)
            best_id = ids[indices]
            return {"selnode": self.id_to_node[best_id[0].item()]}
        else:
            return {"selnode": None}

    def nodecomp(self, node1, node2):
        '''
        compare two leaves of the current branching tree

        It should return the following values:

        value < 0, if node 1 comes before (is better than) node 2
        value = 0, if both nodes are equally good
        value > 0, if node 1 comes after (is worse than) node 2.
        '''
        node1Idx = (self.ids == node1.getNumber()).nonzero()[0][0]
        node2Idx = (self.ids == node2.getNumber()).nonzero()[0][0]
        return self.probs[node2Idx] - self.probs[node1Idx]


class LinNodesel(Nodesel):
    def __init__(self, model, policy, dataset=None):
        self.policy = policy
        self.model = model
        self.dataset = dataset
        self.nodeToParent = {}
        self.tree = Tree()


    def nodeselect(self):
        listOfNodes = list(itertools.chain.from_iterable(self.model.getOpenNodes()))
        if len(listOfNodes) == 0:
            return {"selnode": None}
        optimalNode = listOfNodes[0]
        optimalVal = self.policy(getNodeFeature(optimalNode, self.model))
        for i in range(1,len(listOfNodes)):
            challengerVal = self.policy(getNodeFeature(listOfNodes[i], self.model))
            if challengerVal > optimalVal:
                optimalVal = challengerVal
                optimalNode = listOfNodes[i]

        curr_node = self.model.getCurrentNode()
        if curr_node != None :
            number = curr_node.getNumber()
            if self.tree.size() == 0 or curr_node.getParentBranchings() == None or curr_node.getNumber() == 1:
                self.tree = Tree()
                self.tree.create_node(number, number, data=nodeData(curr_node, self.model.getLPObjVal(), self.model))
            else:
                variables, branch_bounds, bound_types = curr_node.getParentBranchings()
                parent_node = curr_node.getParent()
                parent_num = parent_node.getNumber()
                try:
                    self.tree.create_node(number, number, parent=parent_num,
                                          data=nodeData(curr_node, self.model.getLPObjVal(), self.model,
                                                        variables=variables,
                                                        bound_types=bound_types, branch_bounds=branch_bounds))
                except:
                    pass

        idToFeature = {}
        for node in listOfNodes:
            idToFeature[node.getNumber()] = getNodeFeature(node, self.model)
        if self.dataset is not None:
            self.dataset.append(idToFeature)

        return {"selnode": optimalNode }



    def nodecomp(self, node1, node2):
        with torch.no_grad():
            valOne = self.policy(getNodeFeature(node1, self.model))
            valTwo = self.policy(getNodeFeature(node2, self.model))

        return(valTwo - valOne)



