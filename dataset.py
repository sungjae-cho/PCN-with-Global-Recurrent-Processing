import torch
import torchvision

def load_dataset(dataset, batch_size):
    if dataset == 'MNIST':
        n_input_channels = 1
        n_classes = 10
        # Reference: https://neuroai.neuromatch.io/tutorials/W1D2_ComparingTasks/student/W1D2_Tutorial1.html
        mnist_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),  # Convert images to tensor
            torchvision.transforms.Normalize((0.1307,), (0.3081,))  # Normalize the images with mean and standard deviation
        ])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transforms)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)  # Enable persistent_workers=True if more than 1 worker to save CPU
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transforms)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)   # Enable persistent_workers=True if more than 1 worker to save CPU

    elif 'CIFAR10': # CIFAR-10
        n_input_channels = 3
        n_classes = 10
        # Reference: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)  # Enable persistent_workers=True if more than 1 worker to save CPU
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)   # Enable persistent_workers=True if more than 1 worker to save CPU

    elif 'CIFAR100': # CIFAR-10
        n_input_channels = 3
        n_classes = 100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize_test, shuffle=False, num_workers=2)

    return (trainset, testset), (trainloader, testloader), n_input_channels, n_classes