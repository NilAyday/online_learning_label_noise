import torch
import torchvision
from src import datasets, perturbed_dataloader, training, util, subloader
import random
import os
import pickle
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import matplotlib.pyplot as plt
import numpy as np

# Takes the classes 0 airplane, 1 automobile and 8 ship
ds_train=subloader.SubLoader(include_list=[0,1],num_data=300,root="./datasets", train=True, download=True, transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
ds_test=subloader.SubLoader(include_list=[0,1],num_data=60 ,root="./datasets", train=False, download=True, transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))


indices_test = [i for i in range(len(ds_test))]
random.shuffle(indices_test)

batch_size=1

num_data=len(ds_train.targets)
print(num_data)
print(np.count_nonzero(ds_train.targets))
print(np.count_nonzero(ds_train.targets==1))
#dataset_train = torch.utils.data.Subset(ds_train, indices_train[:])
dataset_train = perturbed_dataloader.PerturbedDataset(ds_train, 0.3, size = num_data,num_classes = 2,enforce_false = False)
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

#dataset_test = torch.utils.data.Subset(ds_test, indices_test[:200])
dataloader_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False)

dl_batch=[dataloader_train,dataloader_test]

file= os.path.join(os.path.join(os.path.dirname(__file__)), './data/dataloader_single_CIFAR10.pickle')
with open(file, 'wb') as handle:
    pickle.dump(dl_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)

# visualization
'''
training_data=dataset_test
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.T.squeeze(), cmap="gray")
plt.show()
'''
