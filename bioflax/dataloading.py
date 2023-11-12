import torch
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
from flax import linen as nn
from .model import (
    BatchTeacher
)


def create_dataset(seed, batch_size, dataset, val_split, input_dim, output_dim, L, train_set_size, test_set_size, teacher_act):
    """
    Function that creates asked dataset. Returns pytorch dataloaders.
    ...
    Parameters
    __________
    seed : int
        seed for randomness
    batch_size : int
        size of batches
    dataset : str
        identifier for dataset
    val_split : float
        fraction of the training set used for validation
    input_dim : int
        dimensionality of inputs
    output_dim : int
        dimensionality of outputs
    L : int
        length of inputs
    train_set_size : int
        size of the training set in terms of batches
    test_set_size : int
        size of the test set in terms of batches
    teacher_act : str
        activation function of teacher network
    """
    if (dataset == "mnist"):
        return create_mnist_dataset(seed, batch_size, val_split)
    elif (dataset == "teacher" or dataset == "sinreg"):
        return create_random_dataset(seed, batch_size, val_split, input_dim, output_dim, L, train_set_size, test_set_size, dataset, teacher_act)
    else:
        raise ValueError("Unknown dataset")


def create_mnist_dataset(seed, batch_size, val_split):
    """
    Returns pytorch train-, test-, and validation-dataloaders for mnist dataset.
    ...
    Parameters
    __________
    seed : int
        seed for randomness
    batch_size : int
        size of batches
    val_split : float
        fraction of the training set used for validation
    """
    _name_ = "mnist"
    d_input = 1
    d_output = 10
    L = 784

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(L, d_input).T),
    ])

    train_dataset = datasets.MNIST(
        'data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(
        'data', train=False, download=True, transform=transform)

    train_loader, val_loader, test_loader = create_loaders(
        train_dataset, test_dataset, seed, val_split, batch_size)

    return train_loader, val_loader, test_loader, d_output, L, d_input,


def create_random_dataset(seed, batch_size, val_split, input_dim, output_dim, L, train_set_size, test_set_size, dataset, teacher_act):
    """
    Returns pytorch train-, test-, and validation-dataloaders for either a Teacher Network dataset or a sinus dataset.
    ...
    Parameters
    __________
    seed : int
        seed for randomness
    batch_size : int
        size of batches
    val_split : float
        fraction of the training set used for validation
    input_dim : int
        dimensionality of inputs
    output_dim : int
        dimensionality of outputs
    L : int
        length of inputs
    train_set_size : int
        size of the training set in terms of batches
    test_set_size : int
        size of the test set in terms of batches
    dataset : str
        identifier for dataset
    teacher_act : str
        activation function of teacher network
    """

    d_input = input_dim
    d_output = output_dim
    L = L

    rng = jax.random.PRNGKey(seed)
    model_rng,  key = jax.random.split(rng, num=2)
    test_rng, train_rng = jax.random.split(key)

    if (dataset == "teacher"):
        model = BatchTeacher(features=d_output, activation=teacher_act)
        params = model.init(model_rng, jnp.ones(
            (batch_size, d_input, L)))['params']
    elif (dataset == "sinreg"):
        model = None
        params = None

    train_dataset = generate_sample_set(
        model, params, train_rng, batch_size, d_input, L, train_set_size)
    test_dataset = generate_sample_set(
        model, params, test_rng, batch_size, d_input, L, test_set_size)

    train_loader, val_loader, test_loader = create_loaders(
        train_dataset, test_dataset, seed, val_split, batch_size)

    return train_loader, val_loader, test_loader, d_output, L, d_input,


def create_loaders(train_dataset, test_dataset, seed, val_split, batch_size):
    """
    Returns pytorch train-, test-, and validation-dataloaders for given pytorch datasets.
    ...
    Parameters
    __________
    train_dataset : torch.utils.data.Dataset
        pytorch dataset for training
    test_dataset : torch.utils.data.Dataset
        pytorch dataset for testing
    seed : int
        seed for randomness
    val_split : float
        fraction of the training set used for validation
    batch_size : int
        size of batches
    """

    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    if (val_split != 0.):
        train_dataset, val_dataset = split_train_val(
            train_dataset, val_split, seed)
    else:
        val_dataset = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=rng)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, generator=rng)

    if (val_dataset != None):
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True, generator=rng)
    else:
        val_loader = None
    return train_loader, val_loader, test_loader


def generate_sample_set(model, params, data_rng, batch_size, d_input, L, size):
    """
    Returns pytorch dataset for given model and parameters and for sinus function if no model is given.
    ...
    Parameters
    __________
    model : Any
        model to be trained
    params : Any
        parameters of model
    data_rng : jnp.PRNGKey
        key for randomness
    batch_size : int
        size of batches
    d_input : int
        dimensionality of inputs
    L : int
        length of inputs
    size : int
        size of dataset in terms of batches
    """

    for i in tqdm(range(size)):
        data_rng, key = jax.random.split(data_rng)
        if model == None:
            x = nn.initializers.uniform(
                scale=2 * jnp.pi)(key, shape=(batch_size, d_input, L)) - jnp.pi
            y = jnp.sin(x)
        else:
            x = nn.initializers.uniform(scale=2)(
                key, shape=(batch_size, d_input, L)) - 1.
            y = model.apply({'params': params}, x)
        if (i == 0):
            inputs = x
            outputs = y
        else:
            inputs = jnp.concatenate((inputs, x), axis=0)
            outputs = jnp.concatenate((outputs, y), axis=0)
    inputs_py = np.array(inputs)
    outputs_py = np.array(outputs)

    inputs_tensor = torch.from_numpy(inputs_py)
    outputs_tensor = torch.from_numpy(outputs_py)

    return torch.utils.data.TensorDataset(inputs_tensor, outputs_tensor)


def split_train_val(train_dataset, val_split, seed):
    """
    Randomly split self.dataset_train into a new (self.dataset_train, self.dataset_val) pair.
    ...
    Parameters
    __________
    train_dataset : torch.utils.data.Dataset
        pytorch dataset for training
    val_split : float
        fraction of the training set used for validation
    seed : int
        seed for randomness
    """
    train_len = int(len(train_dataset) * (1.0 - val_split))
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset,
        (train_len, len(train_dataset) - train_len),
        generator=torch.Generator().manual_seed(seed),
    )
    return train_dataset, val_dataset
