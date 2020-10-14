import torch.sparse
import faulthandler
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from utilities import init_scip_params
device = torch.device('cpu')

# hyper parameters
x_size = 14
h_size = 14
dropout = 0.5
lr = 0.05
weight_decay = 1000000000000
epochs = 10

lstmFeature = TreeLSTMBranch(x_size,
                       h_size,
                       dropout,
                       device=device)
lstmFeature.load_state_dict(torch.load("branch_lstm.pt"))

my_dagger = branchDagger(lstmFeature, "../realsingle", device, num_train = 1000, num_epoch=4, save_path="../lstmFeature.pt")
tree_vals, def_vals = my_dagger.test("../realsingle")

with open('answer.pkl', 'wb') as f:
    pickle.dump([tree_vals, def_vals], f)
print(tree_vals)
print(def_vals)
