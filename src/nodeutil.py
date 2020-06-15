import torch
import networkx as nx
import faulthandler
faulthandler.enable()

class nodeData():
    def __init__(self, node, val, model, variables=None, branch_bounds=None, bound_types=None):
        self.node = node
        self.feature = getNodeFeature(node, model)
        self.nodeid = node.getNumber()
        self.val = val
        self.variables = variables
        self.branch_bounds = branch_bounds
        self.bound_types = bound_types

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
# %%
def getNodeFeature(node, model):

    if node.getParentBranchings() != None:
        variables, branch_bounds, bound_types = node.getParentBranchings()
        return torch.Tensor(
            [node.getDepth(), node.getLowerbound(), node.getEstimate(), node.getType(), model.getPrimalbound(),
             model.getDualbound(), node.getNDomchg()[0], node.getNDomchg()[1], node.getNDomchg()[2],
             node.getNAddedConss(),
             node.isActive(), node.isPropagatedAgain(), model.getGap(), model.getVariablePseudocost(variables[0])])
    else:
        return torch.Tensor(
            [node.getDepth(), node.getLowerbound(), node.getEstimate(), node.getType(), model.getPrimalbound(),
             model.getDualbound(), node.getNDomchg()[0], node.getNDomchg()[1], node.getNDomchg()[2],
             node.getNAddedConss(),
             node.isActive(), node.isPropagatedAgain(), model.getGap(), 0])

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

def _build_tree(tree, model):
    root = tree.get_node(tree.root)
    g = nx.DiGraph()

    def _rec_build(nid, node):
        children = tree.children(node.identifier)
        for child in children:
            cid = child.identifier
            g.add_node(cid, feature=child.data.feature, node_id=cid, in_queue=0)
            if not child.is_leaf():
                _rec_build(cid, child)
            g.add_edge(nid, cid)

    g.add_node(root.identifier, feature=root.data.feature, node_id=root.identifier, in_queue=0)
    _rec_build(root.identifier, root)
    return g
# %%