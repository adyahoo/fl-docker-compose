from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

def load_data():
    trf = Compose([
        ToTensor(),
        Normalize((0.5,), (0.5,)),
    ])
    trainset = MNIST(root='./data', train=True, download=True, transform=trf)
    testset = MNIST(root='./data', train=False, download=True, transform=trf)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=False)
    testloader = DataLoader(testset, batch_size=32)

    return trainloader, testloader