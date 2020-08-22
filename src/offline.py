import torch
# import utilities as ut
# import utilities_gnn as ut_gnn
import os
import argparse
import time
import pickle
import torch.sparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
import torch.sparse
import itertools
from treelib import Tree
import networkx as nx
import dgl
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
import sys
import faulthandler
from datetime import datetime
from TreeLSTM import TreeLSTMCell, TreeLSTM
from NodeSel import MyNodesel, LinNodesel
from nodeutil import getListOptimalID, checkIsOptimal
from pyscipopt import Model, Heur, quicksum, multidict, SCIP_RESULT, SCIP_HEURTIMING, SCIP_PARAMSETTING, Sepa, \
    Branchrule, Nodesel
import glob
from TreeLSTM import TreeLSTMBranch
from utilities import init_scip_params, init_scip_params_haoran, personalize_scip
from brancher import TreeBranch

import os

class tree_offline():
    def __init__(self, model, epoch=5, batch_size=5):

    def train(self, data_path):
        s_loader = DataLoader(samples, batch_size=self.batch_size, shuffle=True, collate_fn=collate)
        print('Number of datapoints: %d' % (len(samples)))
        for epoch in range(self.num_epoch):
            running_loss = 0.0
            number_right = 0
            total_weight = 0
            print("loading")
            for (bg, labels, weights) in s_loader:
                self.optimizer.zero_grad()

                unbatched, outputs = self.compute(bg)
                total_loss = None
                for i in range(len(unbatched)):
                    output = outputs[i]
                    label = labels[i]
                    weight = weights[i]
                    total_weight += weight
                    _, indices = torch.max(output, 0)
                    if indices.item() == label.item():
                        number_right += 1 * weight
                    output = output.unsqueeze(0)
                    label = label.unsqueeze(0)
                    loss = self.loss(output, label.to(device=self.device))
                    if total_loss == None:
                        total_loss = loss
                    else:
                        total_loss = total_loss + loss
                    running_loss += loss.item() * weight
                    total_weight += weight
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
            total_loss += running_loss
            average_loss += total_loss
            total_num_cases += len(samples)
            total_num_right += number_right
            print('[%d] loss: %.3f accuracy: %.3f number right: %.3f' %
                  (epoch + 1, running_loss / total_weight, number_right / total_weight, number_right))
            running_loss = 0.0
        # except:
        #     continue
        if os.path.exists(self.save_path):
            os.remove(self.save_path)
        torch.save(self.policy.state_dict(), self.save_path)
