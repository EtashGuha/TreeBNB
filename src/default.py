from pyscipopt import Model, Heur, quicksum, multidict, SCIP_RESULT, SCIP_HEURTIMING, SCIP_PARAMSETTING, Sepa, \
    Branchrule, Nodesel
from brancher import TreeBranch
import torch
from TreeLSTM import TreeLSTMBranch
from nodeutil import checkIsOptimal
# hyper parameters
x_size = 14
h_size = 14
dropout = 0.5
lr = 0.05
weight_decay = 1000000000000
epochs = 10

device= torch.device('cpu')
lstmFeature = TreeLSTMBranch(x_size,
                       h_size,
                       dropout,
                       device=device)
problem = "../realsingle/instance_9.lp"

model = Model("setcover")
model.readProblem(problem)
myBranch = TreeBranch(model, lstmFeature)
model.includeBranchrule(myBranch, "ImitationBranching", "Policy branching on variable",
                                    priority=99999, maxdepth=-1, maxbounddist=1)
model.optimize()

