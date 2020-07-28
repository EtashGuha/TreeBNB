import torch.sparse
import faulthandler
from TreeLSTM import TreeLSTMCell, TreeLSTM, LinLib, ShallowLib
from dagger import TreeDagger, RankDagger
import os
import matplotlib.pyplot as plt
from pyscipopt import Model, Heur, quicksum, multidict, SCIP_RESULT, SCIP_HEURTIMING, SCIP_PARAMSETTING, Sepa, \
    Branchrule, Nodesel
from brancher import TreeBranch
import torch
from TreeLSTM import TreeLSTMBranch
from utilities import init_scip_params
device = torch.device('cpu')
import glob
# hyper parameters
x_size = 14
h_size = 14
dropout = 0.5
lr = 0.05
weight_decay = 1000000000000
epochs = 10

# # create the model
# lstmFeature = TreeLSTM(x_size,
#                        h_size,
#                        dropout,
#                        device=device)
# lstmFeature.to(device)
# lstmFeature.cell.to(device)
#
# my_dagger = TreeDagger(lstmFeature, "../data/instances/setcover/train_200r_400c_0.1d_0mc_10se", device, num_train = 1000, num_epoch=4, save_path="../lstmFeature.pt")
# my_dagger.train()
#
# print(my_dagger.listNNodes)

device= torch.device('cpu')

lstmFeature = TreeLSTMBranch(x_size,
                       h_size,
                       dropout,
                       device=device)
# lstmFeature.load_state_dict(torch.load("branch_lstm.pt"))
problem = "../data/instances/setcover/train_200r_400c_0.1d_0mc_10se/instance_66.lp"
losses = []
listNNodes = []
problem_dir = "../data/instances/setcover/train_200r_400c_0.1d_0mc_10se"
problems = glob.glob(problem_dir + "/*.lp")
problems = ["../data/instances/setcover/train_200r_400c_0.1d_0mc_10se/instance_91.lp"]
for problem in problems:
    for i in range(1):
        print(problem )
        model = Model("setcover")
        model.readProblem(problem)
        model.setRealParam('limits/time', 300)
        myBranch = TreeBranch(model, lstmFeature,  train=True)
        init_scip_params(model, 100, False, False, False, False, False, False)

        model.setBoolParam("branching/vanillafullstrong/donotbranch", True)
        model.setBoolParam('branching/vanillafullstrong/idempotent', True)
        model.includeBranchrule(myBranch, "ImitationBranching", "Policy branching on variable",
                                            priority=99999, maxdepth=-1, maxbounddist=1)
        model.optimize()
        losses.append((myBranch.total_loss))
        listNNodes.append(model.getNNodes())
        print(listNNodes)
        print(losses)
        if myBranch.num_example == 0:
            continue
        print('accuracy %.3f' % (myBranch.num_right/myBranch.num_example))