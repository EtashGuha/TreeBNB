import torch
import networkx as nx
import faulthandler
faulthandler.enable()

class nodeData():
    def __init__(self, node, val, variables=None, branch_bounds=None, bound_types=None):
        self.node = node
        self.val = val
        self.variables = variables
        self.branch_bounds = branch_bounds
        self.bound_types = bound_types


# %%
def getNodeFeature(node, model):
    # if node.getParentBranchings() != None:
    #     variables, branch_bounds, bound_types = node.getParentBranchings()
    #     return torch.Tensor(
    #         [node.getDepth(), node.getLowerbound(), node.getEstimate(), node.getType(), model.getPrimalbound(),
    #          model.getDualbound(), model.getVariablePseudocost(variables[0])])
    # else:
    #     return torch.Tensor(
    #         [node.getDepth(), node.getLowerbound(), node.getEstimate(), node.getType(), model.getPrimalbound(),
    #          model.getDualbound(), 0])
    return torch.Tensor(
        [node.getDepth(), node.getLowerbound(), node.getEstimate(), node.getType(), model.getPrimalbound(),
         model.getDualbound(), 0])

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
            g.add_node(cid, feature=getNodeFeature(node.data.node, model), node_id=cid, in_queue=0)
            if not child.is_leaf():
                _rec_build(cid, child)
            g.add_edge(nid, cid)

    g.add_node(root.identifier, feature=getNodeFeature(root.data.node, model), node_id=root.identifier, in_queue=0)
    _rec_build(root.identifier, root)
    return g
# %%