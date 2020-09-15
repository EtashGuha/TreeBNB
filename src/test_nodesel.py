import torch.sparse
import faulthandler
from TreeLSTM import TreeLSTMCell, TreeLSTM, LinLib, ShallowLib
from dagger import TreeDagger, RankDagger, branchDagger
import pickle
import os
import matplotlib.pyplot as plt
from TreeLSTM import TreeLSTMBranch
from pyscipopt import Model, Heur, quicksum, multidict, SCIP_RESULT, SCIP_HEURTIMING, SCIP_PARAMSETTING, Sepa, \
    Branchrule, Nodesel
from brancher import TreeBranch
import torch
import sys
from dagger import branchDagger,  tree_offline
from utilities import init_scip_params
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# hyper parameters
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
    if os.path.exists("../lstmFeature.pt"):
        lstmFeature.load_state_dict(torch.load("../lstmFeature.pt"))
    my_dagger = TreeDagger(lstmFeature, "../data/instances/setcover/train_100r_200c_0.1d_5mc_10se/", device, num_train = 1000, num_epoch=4, save_path="../lstmFeature.pt")
    tree_vals, def_vals = my_dagger.test("../data/instances/setcover/test_100r_200c_0.1d_5mc_10se")
    print(tree_vals)
    print(def_vals)
# create the model


elif mode == "baseline":
    lstmFeature = LinLib(x_size, device)
    lstmFeature.to(device)

    if os.path.exists("../lstmFeatureRank.pt"):
        lstmFeature.load_state_dict(torch.load("../lstmFeatureRank.pt"))

    my_dagger = RankDagger(lstmFeature, "../data/instances/setcover/train_100r_200c_0.1d_5mc_10se/", device,
                           num_train=1000, num_epoch=4, save_path="../lstmFeatureRank.pt")
    my_dagger.testAccuracy("../realsingle")

    # tree_vals , def_vals = my_dagger.test("../data/instances/setcover/test_100r_200c_0.1d_5mc_10se")

    # print(tree_vals)
    # print(def_vals)
elif mode == "tree_super":
    lstmFeature = TreeLSTM(x_size,
                           h_size,
                           dropout,
                           device=device)
    lstmFeature.to(device)
    lstmFeature.cell.to(device)
    if os.path.exists("../lstmFeature.pt"):
        lstmFeature.load_state_dict(torch.load("../lstmFeature.pt"))


    offline =  tree_offline(lstmFeature, "../data/instances/setcover/train_500r_1000c_0.05d_100mc_0se", device, "../data/instances/setcover/100_200samples/100_200.pkl", "../data/instances/setcover/valid_500r_1000c_0.05d_100mc_0se", num_repeat=1,
                           num_train=1000, num_epoch=4, save_path="../lstmFeatureRank.pt")
    offline.setDescription("supervised")
    tree_vals , def_vals = offline.test("../data/instances/setcover/test_100r_200c_0.1d_5mc_10se")
    print(tree_vals)
    print(def_vals)
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
# my_dagger.testAccuracy("../data/instances/setcover/test_100r_200c_0.1d_5mc_10se")
# with open('answer.pkl', 'wb') as f:
#     pickle.dump([tree_vals, def_vals], f)
# print(tree_vals)
# print(def_vals)


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