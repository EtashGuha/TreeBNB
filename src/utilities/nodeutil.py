import torch
import networkx as nx
import faulthandler
import numpy as np
faulthandler.enable()
from treelib import Tree, Node

class nodeData():
    def __init__(self, node, val, model, conflict_score=None, inference_score = None, variables=None, branch_bounds=None, bound_types=None, variable_chosen=-1, lp_obj_val=None, scaled_improvement_up=1, scaled_improvement_down=1):
        self.node = node
        self.feature = getNodeFeature(node, model)
        self.nodeid = node.getNumber()
        self.val = val
        self.variables = variables
        self.branch_bounds = branch_bounds
        self.bound_types = bound_types
        self.variable_chosen=variable_chosen
        self.scaled_improvement_up = scaled_improvement_up
        self.conflict_score = conflict_score
        self.inference_score = inference_score
        self.scaled_improvement_down = scaled_improvement_down
        self.lp_obj_val = lp_obj_val
    def calc_up_improvements(self, upObj, variable):
        self.scaled_improvement_up = (upObj - self.lp_obj_val)/(int(np.ceil(variable.getLPSol())) - variable.getLPSol())
    def calc_down_improvements(self, downObj, variable):
        self.scaled_improvement_down = (downObj - self.lp_obj_val) / (
                    variable.getLPSol() - int(np.floor(variable.getLPSol())))


def checkIsOptimal(node, model, tree):
    if tree.parent(node.tag) is not None and not checkIsOptimal(tree.parent(node.tag), model, tree):
        return False
    if tree.parent(node.tag) is None:
        return True
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

    return isOptimal

def getNodeGap(node, model):
    variables = node.data.variables
    bound_types = node.data.bound_types
    branch_bounds = node.data.branch_bounds
    listOfSols = model.getSols()
    for sol in listOfSols:
        for i in range(len(variables)):
            optval = sol[variables[i]]
            if ((bound_types[i] == 0 and optval < branch_bounds[i]) or (
                    bound_types[i] == 1 and optval > branch_bounds[i])):
                isOptimal = False
                break
        if isOptimal:
            return model.getSolObjVal(sol) - model.getObjVal()
    return -1

# %%
def getNodeFeature(node, model):
    toReturn = None
    if node.getParentBranchings() != None:
        variables, branch_bounds, bound_types = node.getParentBranchings()
        toReturn =  torch.Tensor(
            [node.getDepth(), node.getLowerbound(), node.getEstimate(), node.getType(), model.getPrimalbound(),
             model.getDualbound(), node.getNDomchg()[0], node.getNDomchg()[1], node.getNDomchg()[2],
             node.getNAddedConss(),
             node.isActive(), node.isPropagatedAgain(), model.getGap(), model.getVariablePseudocost(variables[0])])
    else:
        toReturn = torch.Tensor(
            [node.getDepth(), node.getLowerbound(), node.getEstimate(), node.getType(), model.getPrimalbound(),
             model.getDualbound(), node.getNDomchg()[0], node.getNDomchg()[1], node.getNDomchg()[2],
             node.getNAddedConss(),
             node.isActive(), node.isPropagatedAgain(), model.getGap(), 0])
    return toReturn
    #Shallow Neural Network
# %%
def getListOptimalID(initial_id, tree):
    optimal_nodes = {initial_id}
    curr_id = initial_id
    curr_node = tree.get_node(initial_id)
    while not curr_node.is_root():
        curr_node = tree.parent(curr_id)
        curr_id = curr_node.tag
        optimal_nodes.add(curr_id)
    return optimal_nodes


def setRoot(tree, nid):
    queue = list(tree.rsearch(nid))
    queue.reverse()
    if len(queue) == 1:
        return
    queue = queue[1:]
    for id in queue:
        curr_node = tree.get_node(id)
        parent = tree.get_node(curr_node.predecessor(tree.identifier))
        curr_node.update_successors(parent.identifier, mode=Node.ADD, tree_id=tree.identifier)
        parent.update_successors(id, mode=Node.DELETE, tree_id=tree.identifier)
        curr_node.set_predecessor(None, tree.identifier)
        parent.set_predecessor(id, tree.identifier)
        tree.root = id

def _build_tree(tree, model, collect_variable_values=False):
    root = tree.get_node(tree.root)
    g = nx.DiGraph()

    def _rec_build(nid, node):
        children = tree.children(node.identifier)
        for child in children:
            cid = child.identifier
            g.add_node(cid, feature=child.data.feature,
                       node_id=cid, in_queue=0,
                       variable_chosen=child.data.variable_chosen,
                       scaled_improvement_up=child.data.scaled_improvement_up,
                       scaled_improvement_down=child.data.scaled_improvement_down)
            if not child.is_leaf():
                _rec_build(cid, child)
            g.add_edge(nid, cid)

    g.add_node(root.identifier, feature=root.data.feature, node_id=root.identifier, in_queue=0,
               variable_chosen=root.data.variable_chosen,
               scaled_improvement_up=root.data.scaled_improvement_up,
               scaled_improvement_down=root.data.scaled_improvement_down)
    _rec_build(root.identifier, root)
    return g
# %%