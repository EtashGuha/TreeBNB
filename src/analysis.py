import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gmean

tree = [154, 1, 226, 207, 216, 8, 249, 188, 5, 197, 3, 214, 13, 212, 34, 212, 230, 173, 3, 194, 104, 214, 238, 1, 249, 85, 235, 249, 4, 233, 6, 202, 196, 221, 223, 269, 215, 240, 242, 164, 150, 198, 37, 223, 1, 48, 12, 234, 222, 275]
default = [67, 1, 327, 1961, 1001, 11, 334, 2034, 5, 522, 2, 578, 10, 3508, 70, 1096, 75, 589, 3, 3728, 81, 705, 496, 1, 344, 152, 902, 1008, 3, 252, 5, 207, 263, 821, 138, 3388, 730, 294, 2796, 194, 64, 26, 24, 181, 1, 52, 10, 1658, 246, 2621]
print(len(tree))

plt.title("Default SCIP")
plt.xlabel("Number of Nodes")
plt.hist(default)
plt.show()

print(np.mean)
print(np.std(tree))
print(np.std(default))
mask = np.logical_or(tree != 1, default != 1)
tree_fil = tree[mask]
def_fil = default[mask]
print(gmean(tree_fil))
print(gmean(def_fil))

