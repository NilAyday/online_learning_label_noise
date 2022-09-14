import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
from typing import Union
import random
from src import util,perturbed_dataloader, training_online, training,datasets
from src.util import Net
import torch.nn as nn
import os
import pickle
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 5, stride = 1, padding = 0)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 5, stride = 2, padding = 0)
        self.linear1 = nn.Linear(12800, 512)
        self.linear2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

num_epoch=1
lr=0.001

alpha = 0.3
num_data = 80000
batch_size = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model=  LeNet()
optimizer = torch.optim.SGD(model.parameters(), lr = lr)
loss = torch.nn.CrossEntropyLoss()

ds_train = datasets.load_MNIST(True)
ds_test = datasets.load_MNIST(False)
indices_test = [i for i in range(len(ds_test))]
random.shuffle(indices_test)

dataset_train = perturbed_dataloader.PerturbedDataset(ds_train, alpha, size = num_data,enforce_false = False)
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataset_test = torch.utils.data.Subset(ds_test, indices_test[:10000])
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=300, shuffle=False)



history=training_online.train(model, optimizer, loss, dataloader_train, dataloader_test, num_epoch, device=device)

file= os.path.join(os.path.join(os.path.dirname(__file__)), './data/online_lenet.pickle')
with open(file, 'wb') as handle:
    pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
