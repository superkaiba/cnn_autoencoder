import torchvision
import torchvision.transforms as transforms
import torch

def get_loaders():
    # Load data
    transform = transforms.Compose(
        [transforms.ToTensor(), ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    breakpoint()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                                shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                                shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader