import torch
import networkx as nx
import faulthandler
faulthandler.enable()

class nodeData():
    def __init__(self, node, val, model, variables=None, branch_bounds=None, bound_types=None, variable_chosen=None, scaled_improvement_up=None, scaled_improvement_down=None):
        self.node = node
        self.feature = getNodeFeature(node, model)
        self.nodeid = node.getNumber()
        self.val = val
        self.variables = variables
        self.branch_bounds = branch_bounds
        self.bound_types = bound_types
        self.variable_chosen=variable_chosen
        self.scaled_improvement_up = scaled_improvement_up
        self.scaled_improvement_down = scaled_improvement_down

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

def _build_tree(tree, model):
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