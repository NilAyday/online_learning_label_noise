import torch
import torchvision


def load_MNIST(train):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    return torchvision.datasets.MNIST("./datasets", download=True, transform=transform, train=train)

def load_CIFAR10(train):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    return torchvision.datasets.CIFAR10("./datasets", download=True, transform=transform, train=train)