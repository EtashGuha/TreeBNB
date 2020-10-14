import torch.sparse
import faulthandler
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

my_dagger = branchDagger(lstmFeature, "../realsingle", device, num_train = 1000, num_epoch=1, num_repeat=1, save_path="../lstmFeature.pt")
my_dagger.train()