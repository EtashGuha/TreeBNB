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
if os.path.exists("/Users/etashguha/Documents/TreeBnB/lstmFeature.pt"):
    lstmFeature.load_state_dict(torch.load("/Users/etashguha/Documents/TreeBnB/lstmFeature.pt"))

my_dagger = Dagger(lstmFeature, "../data/instances/setcover/train_200r_400c_0.1d_0mc_10se", "cpu", num_train = 1000, num_epoch=100)

# torch.save(lstmFeature.state_dict(), "/Users/etashguha/Documents/TreeBnB/lstmFeature.pt")
print(my_dagger.test("../data/instances/setcover/test_200r_400c_0.1d_0mc_10se"))
# linClassifier = LinLib(x_size)
#
# linDagger = LinDagger(linClassifier, "../singledata/instance_1972.lp", "cpu", num_train = 200, num_epoch=200)
# linDagger.train()
# print(my_dagger.listNNodes)
plt.plot(my_dagger.listNNodes)
plt.show()

