import torch.sparse
import faulthandler
from TreeLSTM import TreeLSTMCell, TreeLSTM
faulthandler.enable()
nodeLimit = 30000
from dagger import Dagger

device = torch.device('cpu')
# hyper parameters
x_size = 7
h_size = 7
dropout = 0.5
lr = 0.05
weight_decay = 1000000000000
epochs = 10

# create the model
lstmFeature = TreeLSTM(x_size,
                       h_size,
                       dropout)
lstmFeature.load_state_dict(torch.load("/Users/etashguha/Documents/TreeBnB/lstmFeature.pt"))

my_dagger = Dagger(lstmFeature, "/Users/etashguha/Documents/research/singledata", "cpu", num_train = 200, num_epoch=1)
my_dagger.train()
torch.save(lstmFeature.state_dict(), "/Users/etashguha/Documents/TreeBnB/lstmFeature.pt")

