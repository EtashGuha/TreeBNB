import torch.sparse
import faulthandler
from TreeLSTM import TreeLSTMCell, TreeLSTM, LinLib
faulthandler.enable()
nodeLimit = 30000
from dagger import Dagger, LinDagger
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
                       dropout)
# if os.path.exists("/Users/etashguha/Documents/TreeBnB/lstmFeature.pt"):
#     lstmFeature.load_state_dict(torch.load("/Users/etashguha/Documents/TreeBnB/lstmFeature.pt"))

#my_dagger = Dagger(lstmFeature, "../data/instances/setcover/train_150r_300c_0.1d_0mc_10se", "cpu", num_train = 200, num_epoch=1)
my_dagger = Dagger(lstmFeature, "../data/instances/setcover/train_200r_400c_0.1d_0mc_10se", "cpu", num_train = 1000, num_epoch=100)
# my_dagger.listNNodes = [685, 444, 878, 687, 518, 3569, 529, 482, 599, 312, 575, 912, 499, 559, 674, 521, 2166, 693, 619, 515, 538, 507, 452, 543, 443, 471, 422, 507, 521, 786, 1754, 807, 758, 513, 449, 465, 640, 443, 451, 489, 492, 347, 590, 609, 414, 588, 1387, 598, 507, 477, 478, 457, 676, 510, 673, 1040, 542, 527, 495, 539, 577, 471, 380, 544, 553, 628, 445, 644, 490, 467, 479, 582, 442, 471, 653, 501, 573, 429, 926, 433, 444, 943, 428, 562, 530, 505, 424, 530, 499]
my_dagger.train()
# torch.save(lstmFeature.state_dict(), "/Users/etashguha/Documents/TreeBnB/lstmFeature.pt")

# linClassifier = LinLib(x_size)
#
# linDagger = LinDagger(linClassifier, "../singledata/instance_1972.lp", "cpu", num_train = 200, num_epoch=200)
# linDagger.train()
# print(my_dagger.listNNodes)
plt.plot(my_dagger.listNNodes)
plt.show()

