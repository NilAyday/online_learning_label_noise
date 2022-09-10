import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
from typing import Union
import random
from src import util,perturbed_dataloader, training_online
from src.util import MyDataset, MyCorruptedDataset
import torch.nn as nn
import os
import pickle
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt



dim=20
num_samples=3200
batch_size=32
#batch_size=100
alpha=0.3

samples_train=[]
samples_test=[]
labels_train=[]
labels_test=[]


center0=util.sphere(center=np.zeros(dim),radius=1,mode="SURFACE",size=dim)

for i in range(int(num_samples/2)):
    samples_train.append(np.array(util.sphere(center=center0,radius=0.5,mode="INTERIOR",size=dim)))
    labels_train.append(int(0))
    if i<num_samples/4:
        samples_test.append(np.array(util.sphere(center=center0,radius=0.5,mode="INTERIOR",size=dim)))
        labels_test.append(int(0))

center1=center0
while(np.linalg.norm(center0-center1)<1):
    center1=util.sphere(center=np.zeros(dim),radius=1,mode="SURFACE",size=dim)

for i in range(int(num_samples/2),num_samples):
    samples_train.append(np.array(util.sphere(center=center1,radius=0.5,mode="INTERIOR",size=dim)))
    labels_train.append(int(1))
    if i<int(num_samples/2)+num_samples/4:
        samples_test.append(np.array(util.sphere(center=center1,radius=0.5,mode="INTERIOR",size=dim)))
        labels_test.append(int(1))

samples_train=np.array(samples_train)
labels_train=np.array(labels_train)
samples_test=np.array(samples_test)
labels_test=np.array(labels_test)
indices = [i for i in range(len(samples_train))]
random.shuffle(indices)
samples_train=samples_train[indices]
labels_train=labels_train[indices]
indices_test = [i for i in range(len(samples_test))]
random.shuffle(indices_test)
samples_test=samples_test[indices_test]
labels_test=labels_test[indices_test]


tensor_x = torch.Tensor(samples_train) # transform to torch tensor
tensor_y = torch.Tensor(labels_train)
ds_train = MyCorruptedDataset(tensor_x,tensor_y) # create your datset
tensor_x = torch.Tensor(samples_test) # transform to torch tensor
tensor_y = torch.Tensor(labels_test)
ds_test = MyDataset(tensor_x,tensor_y)

dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=1, shuffle=True)
dataset_test = torch.utils.data.Subset(ds_test, indices_test[:])
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False)

dl_single=[dataloader_train,dataloader_test]

file= os.path.join(os.path.join(os.path.dirname(__file__)), './data/dataloader_single.pickle')
with open(file, 'wb') as handle:
    pickle.dump(dl_single, handle, protocol=pickle.HIGHEST_PROTOCOL)

dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
dataset_test = torch.utils.data.Subset(ds_test, indices_test[:])
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False)

dl_batch=[dataloader_train,dataloader_test]

file= os.path.join(os.path.join(os.path.dirname(__file__)), './data/dataloader_batch.pickle')
with open(file, 'wb') as handle:
    pickle.dump(dl_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)

# For visualization of the synthetic dataset, use with dim=3
'''
fig = plt.figure(figsize=(4,4))

ax = fig.add_subplot(111, projection='3d')

for i in range(np.shape(samples_test)[0]):
    if labels_test[i]==0:
        ax.scatter(samples_test[i][0],samples_test[i][1],samples_test[i][2],color='red',alpha=0.25) 
    else:
        ax.scatter(samples_test[i][0],samples_test[i][1],samples_test[i][2],color='blue',alpha=0.25)


    
ax.scatter(center0[0], center0[1],  center0[2], color='red',alpha=1)
ax.scatter(center1[0], center1[1],  center1[2], color='blue',alpha=1)
plt.show()
'''
