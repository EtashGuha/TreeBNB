import torch.sparse
import faulthandler
from TreeLSTM import TreeLSTMCell, TreeLSTM, LinLib, ShallowLib
from dagger import TreeDagger, RankDagger
import os
import matplotlib.pyplot as plt

device = torch.device('cuda:0')

# hyper parameters
x_size = 14
h_size = 14
dropout = 0.5
lr = 0.05
weight_decay = 1000000000000
epochs = 10

# create the model
lstmFeature = TreeLSTM(x_size,
                       h_size,
                       dropout,
                       device=torch.device("cuda:0"))
lstmFeature.to(torch.device("cuda:0"))

if os.path.exists("../lstmFeature.pt"):
    lstmFeature.load_state_dict(torch.load("../lstmFeature.pt"))

my_dagger = TreeDagger(lstmFeature, "../data/instances/setcover/train_200r_400c_0.1d_0mc_10se", torch.device("cuda:0"), num_train = 1000, num_epoch=4, save_path="../lstmFeature.pt")
my_dagger.train()
