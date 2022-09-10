import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
from typing import Union
import random
from src import util,perturbed_dataloader, training_online
from src.util import Net
import torch.nn as nn
import os
import pickle
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time

num_iter=400000
lr=0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net()
optimizer = torch.optim.SGD(model.parameters(), lr = lr)
loss = torch.nn.CrossEntropyLoss()

file= os.path.join(os.path.join(os.path.dirname(__file__)), './data/dataloader_batch.pickle')
with open(file, 'rb') as handle:
    [dataloader_train,dataloader_test] = pickle.load(handle)



t0 = time.time()
history=training_online.train(model, optimizer, loss, dataloader_train, dataloader_test, num_iter, device=device)
print("Training time:", time.time()-t0)
print(history['true_train_acc'][-1])

file= os.path.join(os.path.join(os.path.dirname(__file__)), './data/online_c_{}k.pickle'.format(num_iter/1000))
with open(file, 'wb') as handle:
    pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

