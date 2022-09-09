import numpy as np
import numpy.matlib
from typing import Union
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import random

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def sphere(center: Union[np.ndarray, list], radius: float, mode: str,size) -> np.ndarray:
    """ Samples a point from the surface or from the interior of solid sphere.

    https://math.stackexchange.com/a/87238
    https://math.stackexchange.com/a/1585996

    Example 1: Sample a point from the surface of the solid sphere of a defined radius and center location.

    .. code-block:: python

        Sphere.sample(
            center=Vector([0, 0, 0]),
            radius=2,
            mode="SURFACE"
        )

    :param center: Location of the center of the sphere.
    :param radius: The radius of the sphere.
    :param mode: Mode of sampling. Determines the geometrical structure used for sampling. Available: SURFACE (sampling
                 from the 2-sphere), INTERIOR (sampling from the 3-ball).
    """
    center = np.array(center)

    # Sample
    direction = np.random.normal(loc=0.0, scale=1.0, size=size)

    if np.count_nonzero(direction) == 0:  # Check no division by zero
        direction[0] = 1e-5

    # For normalization
    norm = np.sqrt(direction.dot(direction))

    # If sampling from the surface set magnitude to radius of the sphere
    if mode == "SURFACE":
        magnitude = radius
    # If sampling from the interior set it to scaled radius
    elif mode == "INTERIOR":
        magnitude = radius * np.cbrt(np.random.uniform())
    else:
        raise Exception("Unknown sampling mode: " + mode)

    # Normalize
    sampled_point = list(map(lambda x: magnitude*x/norm, direction))

    # Add center
    location = np.array(sampled_point) + center

    return location

def get_Jacobian_svd(model, dl_train):
    device = torch.device('cuda')
    num_classes=10
    grad_batch = []
    i=0
    for batch in dl_train:
        
        i+=1
        cur_gradient = []
        x = batch[0].to(device)
        y = batch[1].to(device)
        for cur_lbl in range(1):
            model.zero_grad()
            cur_output = model(x)
            cur_one_hot = [0] * int(num_classes)
            cur_one_hot[cur_lbl] = 1
            cur_one_hot=np.matlib.repmat([cur_one_hot], np.shape(cur_output)[0], 1)
            cur_one_hot = torch.FloatTensor(cur_one_hot).cuda()

          
            cur_output.backward(cur_one_hot)
            for para in model.parameters():
                cur_gradient.append(para.grad.data.cpu().numpy().flatten())
        
        grad_batch.append(np.concatenate(cur_gradient)) 


    uv, sv, vtv = np.linalg.svd(grad_batch, full_matrices=False) 
    
    return sv


class MyDataset(Dataset):
    def __init__(self, data, targets):
        self._dataset_kind = None
        self.multiprocessing_context = None
        
        self.data = data
        #self.targets = torch.LongTensor(targets)
        #self.targets = targets.clone().detach().requires_grad_(True)
        self.targets = torch.tensor(targets, dtype=torch.long)
        
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        return x, y
    
    def __len__(self):
        return len(self.data)

class MyCorruptedDataset(Dataset):
    def __init__(self, data, targets, alpha=0.3, num_classes=2):
        self._dataset_kind = None
        self.multiprocessing_context = None
        
        self.data = data
        #self.targets = torch.LongTensor(targets)
        self.targets = torch.tensor(targets, dtype=torch.long)
        #self.targets = targets.clone().detach().requires_grad_(True)
        size=len(data)
        num_noise = int(size * alpha)
        indices = [i for i in range(len(self.data))]
        random.shuffle(indices)
        self.correct_indices = indices[num_noise:size]
        self.perturbed_indices = indices[:num_noise]
        self.perturbed_labels = [random.randint(0,num_classes-1) for _ in self.perturbed_indices]
        self.y=self.targets.clone().detach()
        self.y[self.perturbed_indices]=torch.tensor(self.perturbed_labels, dtype=torch.long)

        
    def __getitem__(self, index):
        x = self.data[index]
        y_true = self.targets[index]
        y=self.y[index]
        
        

        return x, y,y_true
    
    def __len__(self):
        return len(self.data)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 5, stride = 1, padding = 0)
        #self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 5, stride = 2, padding = 0)
        self.linear1 = nn.Linear(20, 1000)
        self.linear2 = nn.Linear(1000, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        #x = self.conv1(x)
        #x = self.relu(x)
        #x = self.conv2(x)
        #x = self.relu(x)

        #x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
