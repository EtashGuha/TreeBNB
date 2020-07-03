
import numpy as np
import matplotlib.pyplot as plt

# li = np.array([1276, 710, 627, 604, 585, 682, 664, 595, 747, 830, 658, 710, 527, 540, 833, 477, 520, 580, 568, 544, 514, 605, 648, 807, 863, 755, 474, 845, 747, 502, 657, 624, 685, 647, 578, 656, 568, 632, 505, 576, 598, 603, 457, 530, 548, 559, 724, 758, 565, 632, 590, 572, 518, 526, 609, 712, 580, 530, 473, 457, 572])
# li = li[np.where(li < 1000)]
# print(len(li))
# print(np.mean(li[-10:]))
# print(np.std(li[-1:]))
# plt.plot(li)
# plt.show()
tree = np.array([177, 337, 2, 53, 15, 431, 7, 9, 402, 25, 39, 438, 198, 4, 1, 428, 94, 30, 2, 1])
default = np.array([246, 265, 2, 269, 7, 364, 3, 7, 656, 186, 168, 634, 290, 164, 1, 193, 132, 203, 68, 1])
differences = (default - tree)/default
differences = differences[differences != 0]
print(np.mean(default-tree))
print(np.mean(differences))