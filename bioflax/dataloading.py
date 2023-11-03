import torch
from torchvision import datasets, transforms
from flax import linen as nn
import jax
import jax.numpy as jnp
from tqdm import tqdm
import numpy as np
from .model import (
    BatchTeacher
)


def create_dataset(seed, batch_size, dataset, val_split, input_dim, output_dim, L, train_set_size, test_set_size):
    if(dataset == "mnist"):
        return create_mnist_dataset(seed, batch_size, val_split)
    elif(dataset == "teacher"):
        return create_teacher_dataset(seed, batch_size, val_split, input_dim, output_dim, L, train_set_size, test_set_size)
    elif(dataset == "sinprop"):
        return create_sinprop_dataset(seed, batch_size, val_split, input_dim, output_dim, L, train_set_size, test_set_size)
    else:
        raise ValueError("Unknown dataset")
    
def create_sinprop_dataset(seed, batch_size, val_split, input_dim, output_dim, L, train_set_size, test_set_size):
    _name_ = "teacher"
    d_input = input_dim
    d_output = output_dim
    L = L

    rng = jax.random.PRNGKey(seed)
    test_rng, train_rng = jax.random.split(rng, num=2)

    train_dataset = generate_sample_set(None, None, train_rng, batch_size, d_input, L, train_set_size)
    test_dataset = generate_sample_set(None, None, test_rng, batch_size, d_input, L, test_set_size)

    train_loader, val_loader, test_loader, train_size = create_loaders(train_dataset, test_dataset, seed, val_split, batch_size)

    return train_loader, val_loader, test_loader, train_size, d_output, L, d_input,



def create_teacher_dataset(seed, batch_size, val_split, input_dim, output_dim, L, train_set_size, test_set_size):
    _name_ = "teacher"
    d_input = input_dim
    d_output = output_dim
    L = L

    model = BatchTeacher()
    rng = jax.random.PRNGKey(seed)
    model_rng,  key = jax.random.split(rng, num=2)
    test_rng, train_rng = jax.random.split(key)
    
    params = model.init(model_rng, jnp.ones((batch_size, d_input, L)))['params']

    train_dataset = generate_sample_set(model, params, train_rng, batch_size, d_input, L, train_set_size)
    test_dataset = generate_sample_set(model, params, test_rng, batch_size, d_input, L, test_set_size)

    train_loader, val_loader, test_loader, train_size = create_loaders(train_dataset, test_dataset, seed, val_split, batch_size)

    return train_loader, val_loader, test_loader, train_size, d_output, L, d_input,
    

def generate_sample_set(model, params, data_rng, batch_size, d_input, L, size):
    for i in tqdm(range(size)):
        data_rng, key = jax.random.split(data_rng)
        if model == None:
            x = nn.initializers.uniform(scale = 2 * jnp.pi)(key, shape=(batch_size, d_input, L)) - jnp.pi
            y = jnp.sin(x)
        else:
            x = nn.initializers.uniform(scale = 2)(key, shape=(batch_size, d_input, L)) - 1.
            y = model.apply({'params': params}, x)
        if(i == 0):
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

def create_loaders(train_dataset, test_dataset, seed, val_split, batch_size):
    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    if(val_split != 0.):
        train_dataset, val_dataset = split_train_val(train_dataset, val_split, seed)
    else:
        val_dataset = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator = rng)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, generator = rng)

    if(val_dataset != None):
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, generator = rng)
    else:
        val_loader = None
    return train_loader, val_loader, test_loader, len(train_dataset) 

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

    train_loader, val_loader, test_loader, train_size = create_loaders(train_dataset, test_dataset, seed, val_split, batch_size)

    return train_loader, val_loader, test_loader, train_size, d_output, L, d_input,
   

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