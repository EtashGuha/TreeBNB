import torch.sparse
import faulthandler
from TreeLSTM import TreeLSTMCell, TreeLSTM, LinLib, ShallowLib
from dagger import TreeDagger, RankDagger
import pickle
import os
import matplotlib.pyplot as plt
from TreeLSTM import TreeLSTMBranch
from pyscipopt import Model, Heur, quicksum, multidict, SCIP_RESULT, SCIP_HEURTIMING, SCIP_PARAMSETTING, Sepa, \
    Branchrule, Nodesel
from brancher import TreeBranch
import torch
from utilities import init_scip_params
device = torch.device('cpu')

# hyper parameters
x_size = 14
h_size = 14
dropout = 0.5
lr = 0.05
weight_decay = 1000000000000
epochs = 10

# create the model
# lstmFeature = TreeLSTM(x_size,
#                        h_size,
#                        dropout,
#                        device=device)
# lstmFeature.to(device)
# lstmFeature.cell.to(device)
# if os.path.exists("../lstmFeature.pt"):
#     lstmFeature.load_state_dict(torch.load("../lstmFeature.pt"))
#
# my_dagger = TreeDagger(lstmFeature, "../data/instances/setcover/train_200r_400c_0.1d_0mc_10se", device, num_train = 1000, num_epoch=4, save_path="../lstmFeature.pt")
# tree_vals, def_vals = my_dagger.test("../data/instances/setcover/test_200r_400c_0.1d_0mc_10se")
# with open('answer.pkl', 'wb') as f:
#     pickle.dump([tree_vals, def_vals], f)
# print(tree_vals)
# print(def_vals)




lstmFeature = TreeLSTMBranch(x_size,
                       h_size,
                       dropout,
                       device=device)
lstmFeature.load_state_dict(torch.load("branch_lstm.pt"))


problem = "../data/instances/setcover/train_200r_400c_0.1d_0mc_10se/instance_92.lp"
model = Model("setcover")
model.readProblem(problem)
model.setRealParam('limits/time', 300)
myBranch = TreeBranch(model, lstmFeature,  train=False)
init_scip_params(model, 100, False, False, False, False, False, False)

model.setBoolParam("branching/vanillafullstrong/donotbranch", True)
model.setBoolParam('branching/vanillafullstrong/idempotent', True)

model.includeBranchrule(myBranch, "ImitationBranching", "Policy branching on variable",
                                    priority=99999, maxdepth=-1, maxbounddist=1)
model.optimize()
withTree = model.getNNodes()

model = Model("setcover")
model.readProblem(problem)
model.setRealParam('limits/time', 300)
init_scip_params(model, 100, False, False, False, False, False, False)

myBranch = TreeBranch(model, lstmFeature,  train=False)
model.setBoolParam("branching/vanillafullstrong/donotbranch", True)
model.setBoolParam('branching/vanillafullstrong/idempotent', True)

model.includeBranchrule(myBranch, "ImitationBranching", "Policy branching on variable",
                                    priority=99999, maxdepth=-1, maxbounddist=1)
model.optimize()
withoutTree = model.getNNodes()
print(withTree)
print(withoutTree)
# How many nodes to get last primal bound change
# Why is the feature shallow very unstable?
# How to use to create a structure  by thhe relationship of variables
# aggrregate upper bound and lower bound of each variable: use variable bounds as a feature of the nodes
# Once two nodes have the same variable bounds, create an edge between the nodes?
# use vector indicating variable bounds
# aggreaget variable bounds along tree structure
# try with more datapoints to reduce overfitting of shallow
# try to fix the SCIP's branching policy


#random seed for scip
#random seed for python
#random seed for pytorch
#disable heuristic, probing, propagation heuristic, init_scip look in source code


#Use oracle from wenbo's code
#Talk to haoran about tree flipping maybe
#Conflict score, inferrence score
#pseudoscore