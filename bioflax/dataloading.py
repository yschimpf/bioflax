import torch
from torchvision import datasets, transforms

def create_dataset( seed, batch_size, dataset):
    if(dataset == "mnist"):
        return create_mnist_dataset(seed, batch_size)
   

def create_mnist_dataset(seed, batch_size):
    _name_ = "mnist"
    d_input = 1
    d_output = 10
    L = 784
    # Define transformations to be applied to the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(d_input, L).t()),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load the training and test datasets
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)

    # Create generator for seeding random number draws
    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None
    
    # Create data loaders for the training and test datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator = rng)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, generator = rng)

    train_size = len(train_dataset)

    return train_loader, test_loader, d_output, L, d_input, train_size