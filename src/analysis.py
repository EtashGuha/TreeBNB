import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean

tree = [626, 94, 516, 316, 23, 132, 96, 40, 686, 342, 418, 408, 37, 274, 397, 321, 505, 40, 605, 551]
default = [359, 16, 902, 200, 15, 214, 97, 14, 1185, 440, 511, 458, 10, 189, 307, 287, 504, 17, 1687, 543]
print(len(tree))
tree = np.array(tree)
default = np.array(default)
mask = np.logical_or(tree != 1, default != 1)
tree_fil = tree[mask]
def_fil = default[mask]
print(gmean(tree_fil))
print(gmean(def_fil))

