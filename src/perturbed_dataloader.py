import torch
import torchvision
import random

class PerturbedDataset(torch.utils.data.DataLoader):
    def __init__(self, dataset, num_noise, size = -1, num_classes = 10, enforce_false = True):
        size = len(dataset) if size == -1 else size
        if num_noise <= 1:
            num_noise = int(size * num_noise)

        self.dataset = dataset
        indices = [i for i in range(len(dataset))]
        random.shuffle(indices)
        self.correct_indices = indices[num_noise:size]
        self.perturbed_indices = indices[:num_noise]
        self.enforce_false = enforce_false

        if enforce_false:
            self.perturbed_labels = []
            s = {i for i in range(num_classes)}
            for i in self.perturbed_indices:
                label = self.dataset[i][1]
                self.perturbed_labels.append(random.choice(list(s-{label})))
        else:
            self.perturbed_labels = [random.randint(0,num_classes-1) for _ in self.perturbed_indices]

    def __getitem__(self, i):
        if i < len(self.perturbed_indices):
            ret = self.dataset[self.perturbed_indices[i]]
            if self.enforce_false:
                assert ret[1] != self.perturbed_labels[i]
            return ret[0], self.perturbed_labels[i], ret[1]
        else:
            i -= len(self.perturbed_indices)
            ret = self.dataset[self.correct_indices[i]]
            return ret[0], ret[1], ret[1]

    def __len__(self):
        return len(self.correct_indices) + len(self.perturbed_indices)
