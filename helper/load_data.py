from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset

import torch

class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def load_data():
    trf = Compose([
        ToTensor(),
        Normalize((0.5,), (0.5,)),
    ])
    testset = MNIST(root='./data', train=False, download=True, transform=trf)

    images, labels = torch.load('/app/data/data.pt') #dir inside container
    trainset = CustomDataset(images, labels)

    trainloader = DataLoader(trainset, batch_size=32, shuffle=False)
    testloader = DataLoader(testset, batch_size=32)

    return trainloader, testloader