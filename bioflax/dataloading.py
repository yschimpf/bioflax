import torch
from torchvision import datasets, transforms
from flax import linen as nn
import jax
import jax.numpy as jnp
from tqdm import tqdm
from .model import (
    BatchTeacher
)


def create_dataset(seed, batch_size, dataset, val_split):
    if(dataset == "mnist"):
        return create_mnist_dataset(seed, batch_size, val_split)
    elif(dataset == "teacher"):
        return create_teacher_dataset(seed, batch_size, val_split)

def create_teacher_dataset(seed, batch_size, val_split):
    _name_ = "teacher"
    d_input = 1
    d_output = 10
    L = 20

    model = BatchTeacher()
    rng = jax.random.PRNGKey(seed)
    model_rng, data_rng = jax.random.split(rng, num=2)
    params = model.init(model_rng, jnp.ones((batch_size, d_input, L)))['params']

    for i in tqdm(range(50)):
        data_rng, key = jax.random.split(data_rng)
        x = jax.random.normal(key, shape=(batch_size, d_input, L))
        y = model.apply({'params': params}, x)
        if(i == 0):
            inputs = x
            outputs = y
        else:
            inputs = jnp.concatenate((inputs, x), axis=0)
            outputs = jnp.concatenate((outputs, y), axis=0)
    
    print(inputs.shape)
    print(outputs.shape)

    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    datalaoder = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(inputs, outputs), batch_size=batch_size, shuffle=True, Generator = rng)
    return datalaoder, None, datalaoder, d_output, L, d_input, len(inputs)

   

def create_mnist_dataset(seed, batch_size, val_split):
    _name_ = "mnist"
    d_input = 1
    d_output = 10
    L = 784
    # Define transformations to be applied to the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(L, d_input).T),
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

    if(val_split != 0.):
        test_dataset, val_dataset = split_train_val(test_dataset, val_split, seed)
    else:
        val_dataset = None
    
    # Create data loaders for the training and test datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator = rng)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, generator = rng)
    if(val_dataset != None):
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, generator = rng)
    else:
        val_loader = None

    train_size = len(train_dataset)

    return train_loader, val_loader, test_loader, d_output, L, d_input, train_size

def split_train_val(train_dataset, val_split, seed):
        """
        Randomly split self.dataset_train into a new (self.dataset_train, self.dataset_val) pair.
        """
        train_len = int(len(train_dataset) * (1.0 - val_split))
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset,
            (train_len, len(train_dataset) - train_len),
            generator=torch.Generator().manual_seed(seed),
        )
        return train_dataset, val_dataset