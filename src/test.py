import torch.sparse
import faulthandler
from TreeLSTM import TreeLSTMCell, TreeLSTM, LinLib, ShallowLib
from dagger import TreeDagger, RankDagger
import os
import matplotlib.pyplot as plt

device = torch.device('cpu')

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
                       device=device)
lstmFeature.to(device)
lstmFeature.cell.to(device)
# if os.path.exists("../lstmFeature.pt"):
#     lstmFeature.load_state_dict(torch.load("../lstmFeature.pt"))

my_dagger = TreeDagger(lstmFeature, "../data/instances/setcover/train_200r_400c_0.1d_0mc_10se", device, num_train = 1000, num_epoch=4, save_path="../lstmFeature.pt")
tree_vals, def_vals = my_dagger.test("../data/instances/setcover/test_200r_400c_0.1d_0mc_10se")

