import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

file= os.path.join(os.path.join(os.path.dirname(__file__)), './data_real/regular_small.pickle')
with open(file, 'rb') as handle:
    s = pickle.load(handle)

'''
x=range(50,len(s['train_acc']))
plt.plot(x, np.ones(len(s['train_acc'])-50)-s['train_acc'][50:], label="Train Error",color="darkorange")
plt.plot(x, np.ones(len(s['train_acc'])-50)-s['true_train_acc'][50:], label="Train Error wrt True Labels",color="royalblue")
plt.plot(x, np.ones(len(s['val_acc'])-50)-s['val_acc'][50:], label="Test Error",color="lightgreen")
'''
x=range(len(s['train_acc']))
plt.plot(x, np.ones(len(s['train_acc']))-s['train_acc'], label="Train Error",color="darkorange")
plt.plot(x, np.ones(len(s['train_acc']))-s['true_train_acc'], label="Train Error wrt True Labels",color="royalblue")
plt.plot(x, np.ones(len(s['val_acc']))-s['val_acc'], label="Test Error",color="g")

plt.ylabel("Classification error")
plt.xlabel("Number of Samples")
plt.legend()
file= os.path.join(os.path.join(os.path.dirname(__file__)), './data_real/regular')
plt.savefig(file+".png")
plt.cla()

