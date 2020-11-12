import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TreeLSTM import TreeLSTMCell, TreeLSTM, LinLib, ShallowLib
from dagger import TreeDagger, RankDagger
import torch
from dagger import branchDagger, tree_offline
import sys

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

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

    my_dagger = TreeDagger(lstmFeature, "data/instances/setcover/train_250r_500c_0.05d_10mc_0se", device,
                           "data/instances/setcover/valid_250r_500c_0.05d_10mc_0se", num_repeat=1, num_train=1000, num_epoch=4,
                           save_path="models/lstmFeatureHalf.pt", problem_type="lp")
    my_dagger.setDescription("Removing half of step")
    my_dagger.train()

    print(my_dagger.listNNodes)
    print(my_dagger.description)
elif mode == "baseline":
    lstmFeature = LinLib(x_size, device)
    lstmFeature.to(device)

    my_dagger = RankDagger(lstmFeature, "../data/instances/setcover/train_100r_200c_0.1d_5mc_10se/", device,
                           num_repeat=1,
                           num_train=1000, num_epoch=1, save_path="models/lstmFeatureRank.pt")
    my_dagger.setDescription("First Run")

    my_dagger.train()
elif mode == "tree_super":
    lstmFeature = TreeLSTM(x_size,
                           h_size,
                           dropout,
                           device=device)
    lstmFeature.to(device)
    lstmFeature.cell.to(device)

    offline = tree_offline(lstmFeature, "../data/instances/setcover/train_500r_1000c_0.05d_100mc_0se", device,
                           "../data/instances/setcover/valid_500r_1000c_0.05d_100mc_0se",
                           "../data/instances/setcover/valid_500r_1000c_0.05d_100mc_0se", num_repeat=1,
                           num_train=1000, num_epoch=7, save_path="models/lstmFeature.pt")
    offline.setDescription("supervised")
    offline.train()
