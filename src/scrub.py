import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean

tree = [310, 5, 306, 1, 173, 5, 307, 3, 30, 3, 321, 264, 248, 11, 330, 206, 71, 115, 279, 229, 6, 305, 287, 7, 243, 2, 182, 251, 31, 329, 305, 17, 278, 295, 85, 190, 285, 303, 14, 284, 241, 26, 18, 213, 280, 314, 18, 285, 238, 283]
default = [217, 5, 287, 1, 313, 5, 1305, 3, 32, 3, 692, 1626, 277, 7, 586, 203, 97, 52, 863, 142, 4, 551, 563, 14, 3949, 2, 35, 616, 11, 131, 340, 14, 1937, 830, 111, 131, 1545, 2768, 17, 734, 130, 58, 14, 376, 936, 2084, 41, 250, 233, 1302]
print(len(tree))
tree = np.array(tree)
default = np.array(default)
mask = np.logical_or(tree != 1, default != 1)
tree_fil = tree[mask]
def_fil = default[mask]
print(gmean(tree_fil))
print(gmean(def_fil))

