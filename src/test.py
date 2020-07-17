import torch.sparse
import faulthandler
from TreeLSTM import TreeLSTMCell, TreeLSTM, LinLib, ShallowLib
from dagger import TreeDagger, RankDagger
import pickle
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
if os.path.exists("../lstmFeature.pt"):
    lstmFeature.load_state_dict(torch.load("../lstmFeature.pt"))

my_dagger = TreeDagger(lstmFeature, "../data/instances/setcover/train_200r_400c_0.1d_0mc_10se", device, num_train = 1000, num_epoch=4, save_path="../lstmFeature.pt")
tree_vals, def_vals = my_dagger.test("../data/instances/setcover/test_200r_400c_0.1d_0mc_10se")
with open('answer.pkl', 'wb') as f:
    pickle.dump([tree_vals, def_vals], f)
print(tree_vals)
print(def_vals)

# How many nodes to get last primal bound change
# Why is the feature shallow very unstable?
# How to use to create a structure  by thhe relationship of variables
# aggrregate upper bound and lower bound of each variable: use variable bounds as a feature of the nodes
# Once two nodes have the same variable bounds, create an edge between the nodes?
# use vector indicating variable bounds
# aggreaget variable bounds along tree structure
# try with more datapoints to reduce overfitting of shallow
# try to fix the SCIP's branching policy