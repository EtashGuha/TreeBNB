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
from dagger import branchDagger



device = torch.device('cuda:0')

x_size = 14
h_size = 14
dropout = 0.5
lr = 0.05
weight_decay = 1000000000000
epochs = 10

lstmFeature = TreeLSTM(x_size,
                       h_size,
                       dropout,
                       device=device)
lstmFeature.to(device)
lstmFeature.cell.to(device)

my_dagger = TreeDagger(lstmFeature, "../data/instances/setcover/train_100r_200c_0.1d_5mc_10se/", device, num_train = 1000, num_epoch=4, save_path="../lstmFeature.pt")
my_dagger.train()

print(my_dagger.listNNodes)

device= torch.device('cpu')

lstmFeature = TreeLSTMBranch(x_size,
                       h_size,
                       dropout,
                       device=device)

my_dagger = branchDagger(lstmFeature, "../realsingle", device, num_train = 1000, num_epoch=4, save_path="../lstmFeature.pt")
my_dagger.train()