import torch
from torch.utils.data import DataLoader, sampler
import torchvision.transforms as T
import torchvision.datasets as dset
import numpy as np

NUM_TRAIN = 50000
NUM_VAL = 5000
batch_size = 128


class RangeSampler(sampler.Sampler):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __iter__(self):
        return iter(range(self.start, self.end))

    def __len__(self):
        return self.end - self.start


def get_data():
    mnist_train = dset.MNIST('./datasets/MNIST_data', train=True, download=True,
                             transform=T.ToTensor())
    mnist_loader = DataLoader(mnist_train, batch_size=batch_size,
                              sampler=RangeSampler(0, NUM_TRAIN))
    cifar_train = dset.CIFAR10('./datasets/CIFAR10_data', train=True,
                               download=True, transform=T.ToTensor())
    cifar_loader = DataLoader(cifar_train, batch_size=batch_size)

    return mnist_loader, cifar_loader


def get_cifar_category_loader(category: int):
    cifar_train = dset.CIFAR10('./datasets/CIFAR10_data', train=True,
                               download=True, transform=T.ToTensor())
    category_indices = torch.tensor(np.where(np.array(cifar_train.train_labels) == category)).squeeze()
    category_images = torch.utils.data.dataset.Subset(cifar_train, category_indices)
    category_loader = DataLoader(category_images, batch_size=batch_size)
    return category_loader
