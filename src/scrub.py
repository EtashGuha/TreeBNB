#
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean

#
# treeThenLinear = np.array([177, 337, 2, 53, 15, 431, 7, 9, 402, 25, 39, 438, 198, 4, 1, 428, 94, 30, 2, 1])
# linLearnToRank = np.array([157, 293, 2, 46, 1059, 731, 11, 8, 326, 93, 40, 545, 886, 4, 1, 932, 10, 75, 2, 1])
# shalLearnToRank = np.array([113, 231, 3, 17639, 7, 21256, 3, 6, 14996, 22153, 43, 20532, 24951, 5, 1, 5119, 61, 18127, 3, 1])
# default = np.array([246, 265, 2, 269, 7, 364, 3, 7, 656, 186, 168, 634, 290, 164, 1, 193, 132, 203, 68, 1])
#
# treeThenShallow = np.array([2, 35, 1, 47, 227, 393, 3, 347, 17, 115, 64, 219, 489, 1, 47, 15, 79, 80, 2, 2301])
# defaultDifferentOrder = np.array([68, 168, 1, 202, 193, 656, 164, 634, 3, 246, 203, 290, 364, 1, 186, 7, 7, 132, 2, 265])
#
#


tree = [1, 7, 1, 669, 2, 1, 1, 1, 1, 9, 11, 359, 25, 1, 32, 4, 1, 1, 17, 34, 1, 1, 5, 1, 2, 2, 1, 3, 1, 1, 2, 22, 44, 11, 1, 2, 4, 49, 1, 13, 362, 1, 1, 1, 1, 2, 1, 8, 19, 3]
default = [1, 74, 1, 603, 16, 1, 1, 1, 1, 29, 74, 109, 405, 1, 57, 26, 1, 1, 26, 119, 1, 1, 14, 1, 81, 24, 1, 41, 1, 1, 5, 50, 68, 263, 1, 18, 521, 83, 1, 4, 146, 1, 1, 1, 1, 2, 1, 14, 302, 92]
tree = np.array(tree)
default = np.array(default)
mask = np.logical_or(tree != 1, default != 1)
tree_fil = tree[mask]
def_fil = default[mask]
# print((def_fil - tree_fil).mean())
print(gmean(tree_fil))
print(gmean(def_fil))
# print((default-tree).mean())


# print(gmean(treeThenLinear))
# print(gmean(treeThenShallow[:-1]))
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
#
# from treelib import Node, Tree
# tree = Tree()
# tree.create_node("A", "a")  # root node
# tree.create_node("B", "b", parent="a")
# tree.create_node("C", "c", parent="a")
# tree.create_node("D", "d", parent="b")
# tree.create_node("E", "e", parent="b")
# tree.create_node("F", "f", parent="c")
# tree.create_node("G", "g", parent="c")
#
#
# setRoot(tree, "c")
# tree.show()
#

