import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

file= os.path.join(os.path.join(os.path.dirname(__file__)), './data/online_b_4500.pickle')
with open(file, 'rb') as handle:
    s = pickle.load(handle)

x=range(len(s['train_acc']))
plt.plot(x, np.ones(len(s['train_acc']))-s['train_acc'], label="Train Error",color="darkorange")
plt.plot(x, np.ones(len(s['train_acc']))-s['true_train_acc'], label="Train Error wrt True Labels",color="royalblue")
plt.ylabel("Classification error")
plt.xlabel("Number of samples")
plt.legend()
file= os.path.join(os.path.join(os.path.dirname(__file__)), './data/fig_b_4500')
plt.savefig(file+".png")
plt.cla()

