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
from dagger import branchDagger,  tree_offline
import sys


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

x_size = 14
h_size = 14
dropout = 0.5
lr = 0.05
weight_decay = 1000000000000
epochs = 10
mode = sys.argv[1]
if mode == "tree":
    lstmFeature = TreeLSTM(x_size,
                           h_size,
                           dropout,
                           device=device)
    lstmFeature.to(device)
    lstmFeature.cell.to(device)

    my_dagger = TreeDagger(lstmFeature, "../data/instances/setcover/train_100r_200c_0.1d_5mc_10se/", device, "../data/instances/setcover/valid_500r_1000c_0.05d_100mc_0se", num_repeat=1, num_train = 1000, num_epoch= 4, save_path="../lstmFeature.pt")
    my_dagger.setDescription("Training single instance to get debug accuracy")
    my_dagger.train()

    print(my_dagger.listNNodes)
elif mode == "baseline":
    lstmFeature = LinLib(x_size, device)
    lstmFeature.to(device)

    my_dagger = RankDagger(lstmFeature, "../data/instances/setcover/train_100r_200c_0.1d_5mc_10se/", device, num_repeat=1,
                           num_train=1000, num_epoch=1, save_path="../lstmFeatureRank.pt")
    my_dagger.setDescription("First Run")

    my_dagger.train()
elif mode == "tree_super":
    lstmFeature = TreeLSTM(x_size,
                           h_size,
                           dropout,
                           device=device)
    lstmFeature.to(device)
    lstmFeature.cell.to(device)

    offline =  tree_offline(lstmFeature, "../data/instances/setcover/train_500r_1000c_0.05d_100mc_0se", device, "../data/instances/setcover/valid_500r_1000c_0.05d_100mc_0se",  "../data/instances/setcover/valid_500r_1000c_0.05d_100mc_0se", num_repeat=1,
                           num_train=1000, num_epoch=7, save_path="../lstmFeature.pt")
    offline.setDescription("supervised")
    offline.train()
