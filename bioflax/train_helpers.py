import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import torch
import numpy as np
import flax.linen as nn
import flax
import jax.tree_util as jax_tree
from typing import Any
from jax.nn import one_hot
from tqdm import tqdm
from flax.training import train_state
from functools import partial

def create_train_state(model, rng, lr, momentum, in_dim, batch_size, seq_len):
    """
    Initializes the training state using optax
    ...
    Parameters
    __________
    model : Any
        model to be trained
    rng : jnp.PRNGKey
        key for randomness
    lr : float
        learning rate for optimizer
    momentum : float
        momentum for optimizer
    in_dim : int
        input dimension of model
    batch_size : int
        batch size used when running model
    seq_len : int
        sequence length used when running model
    """
    dummy_input = jnp.ones((batch_size, in_dim, seq_len))
    params = model.init(rng, dummy_input)['params']
    tx = optax.sgd(learning_rate=lr, momentum=momentum)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def train_epoch(state, model, trainloader, seq_len, in_dim, loss_function, n):
    """
    Training function for an epoch that loops over batches.
    ...
    Parameters
    __________
    state : TrainState
        current train state of the model
    model : BioBeuralNetwork
        model that is trained
    seq_len : int
        length of a single input
    in_dim
        dimensionality of a single input
    loss_function: str
        identifier to select loss function
    """
    batch_losses = []
    bias_als_per_layer = []
    wandb_grad_als_per_layer = []
    wandb_grad_als_total = []
    rel_norms_grads = []
    weight_als_per_layer = []

    for i, batch in enumerate(tqdm(trainloader)):
        inputs, labels = prep_batch(batch, seq_len, in_dim)
        state, loss, grads = train_step(state, inputs, labels, loss_function)
        batch_losses.append(loss)
        if i < n: 
            def loss_comp(params):
                logits = model.apply({'params': params}, inputs)
                loss = get_loss(loss_function, logits, labels)
                return loss
            loss_, grads_ = jax.value_and_grad(loss_comp)(reorganize_dict({'params': state.params})["params"])
            
            bias_al_per_layer = compute_bias_grad_al_layerwise(grads_, grads)
            bias_als_per_layer.append(bias_al_per_layer)

            wandb_grad_al_per_layer = compute_wandb_grad_al_layerwise(grads_, grads)
            wandb_grad_als_per_layer.append(wandb_grad_al_per_layer)

            wandb_grad_al_total = compute_wandb_al_total(grads_, grads)
            wandb_grad_als_total.append(wandb_grad_al_total)

            weight_al_per_layer = compute_weight_alignment(state.params)
            weight_als_per_layer.append(weight_al_per_layer)

            rel_norm_grads = compute_rel_norm(grads_, grads)
            rel_norms_grads.append(rel_norm_grads)
    
    avg_bias_al_per_layer = entrywise_average(bias_als_per_layer)
    avg_wandb_grad_al_per_layer = entrywise_average(wandb_grad_als_per_layer)
    avg_wandb_grad_al_total = jnp.mean(jnp.array(wandb_grad_als_total))
    avg_weight_al_per_layer = entrywise_average(weight_als_per_layer)
    avg_rel_norm_grads = jnp.mean(jnp.array(rel_norms_grads))
    return state, jnp.mean(jnp.array(batch_losses)), avg_bias_al_per_layer, avg_wandb_grad_al_per_layer, avg_wandb_grad_al_total, avg_weight_al_per_layer, avg_rel_norm_grads

#new stuff
def remove_keys(pytree, key_list):
    """Removes leaves from the pytree whose path contains any key from key_list."""
    
    def filter_fn(path, value):
        if not any(p.key in key_list for p in path):
            return value
    filtered_pytree = jax_tree.tree_map_with_path(filter_fn, pytree)
    return filtered_pytree

def flatten_array(arr):
    # Use the reshape method to flatten the array
    flattened = arr.reshape(-1)
    return flattened

def flatten_matrices_in_tree(pytree):
    return jax_tree.tree_map(flatten_array, pytree)

@jax.jit
def compute_weight_alignment(paramsFA):
    kernels, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(paramsFA, ['bias', 'B'])))
    Bs, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(paramsFA, ['bias', 'kernel'])))
    dot_prods = jax_tree.tree_map(jnp.dot, kernels, Bs)
    norm_kern = jax_tree.tree_map(jnp.linalg.norm, kernels)
    norm_B = jax_tree.tree_map(jnp.linalg.norm, Bs)
    layerwise_alignments = jax.tree_map((lambda x, y, z: x/(y*z)), dot_prods, norm_kern, norm_B)
    return layerwise_alignments

@jax.jit
def compute_bias_grad_al_layerwise(grads_, grads):
    bias_, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(grads_, ['kernel', 'B'])))
    bias, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(grads, ['kernel', 'B'])))
    dot_prods = jax_tree.tree_map(jnp.dot, bias_, bias)
    norm_grad_ = jax_tree.tree_map(jnp.linalg.norm, bias_)
    norm_grad = jax_tree.tree_map(jnp.linalg.norm, bias)
    layerwise_alignments = jax.tree_map((lambda x, y, z: x/(y*z)), dot_prods, norm_grad_, norm_grad)
    return layerwise_alignments

@jax.jit
def compute_wandb_grad_al_layerwise(grads_, grads):
    bias_, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(grads_, ['kernel', 'B'])))
    bias, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(grads, ['kernel', 'B'])))
    kernel_, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(grads_, ['bias', 'B'])))
    kernel, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(grads, ['bias', 'B'])))
    layerwise_alignments = jax.tree_map((lambda a, b, c, d : (jnp.dot(a, b) + jnp.dot(c, d))/
                                         (jnp.sqrt(jnp.sum(jnp.multiply(a, a)) + jnp.sum(jnp.multiply(c, c)))
                                          * jnp.sqrt(jnp.sum(jnp.multiply(b, b)) + jnp.sum(jnp.multiply(d, d)))) ), bias_, bias, kernel_, kernel)
    return layerwise_alignments

@jax.jit
def compute_wandb_al_total(grads_, grads):
    bias_, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(grads_, ['kernel', 'B'])))
    bias, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(grads, ['kernel', 'B'])))
    kernel_, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(grads_, ['bias', 'B'])))
    kernel, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(grads, ['bias', 'B'])))
    layerwise_alignments = jax.tree_map((lambda a, b, c, d : (jnp.dot(a, b) + jnp.dot(c, d))), bias_, bias, kernel_, kernel)
    squared_norms_ = jax.tree_map((lambda a,b : jnp.sum(jnp.multiply(a, a)) + jnp.sum(jnp.multiply(b, b))), bias_, kernel_)
    squared_norms = jax.tree_map((lambda a,b : jnp.sum(jnp.multiply(a, a)) + jnp.sum(jnp.multiply(b, b))), bias, kernel)
    return jnp.sum(jnp.array(layerwise_alignments))/(jnp.sqrt(jnp.sum(jnp.array(squared_norms_))) * jnp.sqrt(jnp.sum(jnp.array(squared_norms))))

@jax.jit
def compute_rel_norm(grads_, grads):
    bias_, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(grads_, ['kernel', 'B'])))
    bias, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(grads, ['kernel', 'B'])))
    kernel_, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(grads_, ['bias', 'B'])))
    kernel, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(grads, ['bias', 'B'])))
    squared_norms_ = jax.tree_map((lambda a,b : jnp.sum(jnp.multiply(a, a)) + jnp.sum(jnp.multiply(b, b))), bias_, kernel_)
    squared_norms = jax.tree_map((lambda a,b : jnp.sum(jnp.multiply(a, a)) + jnp.sum(jnp.multiply(b, b))), bias, kernel)
    return jnp.sqrt(jnp.sum(jnp.array(squared_norms_)))/jnp.sqrt(jnp.sum(jnp.array(squared_norms)))

#new stuff

@jax.jit
def compute_grad_al_layerwise(grads_, grads):
    flattened_grads_ = flatten_arrays_layerwise(extract_gradients(grads_, ['bias', 'kernel'], True))
    flattened_grads = flatten_arrays_layerwise(extract_gradients(grads, ['bias', 'kernel'], True))
    return compute_layerwise_alignments(flattened_grads_, flattened_grads)

@jax.jit
def compute_bias_grad_layerwise(grads_, grads):
    flattened_grads_ = flatten_arrays_layerwise(extract_gradients(grads_, ['bias'], False))
    flattened_grads = flatten_arrays_layerwise(extract_gradients(grads, ['bias'], False))
    return compute_layerwise_alignments(flattened_grads_, flattened_grads)

@jax.jit
def compute_wandb_alignment(grads_, grads):
    extract_grads = extract_gradients(grads, ['bias', 'kernel'], False)
    extract_grads_ = extract_gradients(grads_, ['bias', 'kernel'], False)
    return compute_alignment(flatten_arrays(extract_grads_),
                              flatten_arrays(extract_grads))

@jax.jit
def compute_weight_alignment_layerwise(state):
    kernels, bs = extract_kernel_and_B_arrays(state.params, False)
    return compute_layerwise_alignments(flatten_arrays_layerwise(kernels), flatten_arrays_layerwise(bs))
            

def entrywise_average(array_of_arrays):
    average_array = np.zeros_like(array_of_arrays[0])
    for arr in array_of_arrays:
        average_array += arr
    
    average_array = average_array/len(array_of_arrays)
    
    return average_array

def merge_consecutive_pairs(array_list):
    # Ensure there is an even number of elements to merge in pairs
    if len(array_list) % 2 != 0:
        raise ValueError("The number of arrays should be even to merge in pairs.")

    # Merge each consecutive pair
    merged_list = []
    for i in range(0, len(array_list), 2):
        a = array_list[i]
        b = array_list[i+1].T
        
        # Ensure both arrays have the same number of dimensions
        if a.ndim < b.ndim:
            a = a[..., jnp.newaxis]
        elif a.ndim > b.ndim:
            b = b[..., jnp.newaxis]
        
        merged_list.append(jnp.concatenate((a, b), axis=1))
    
    return merged_list

@partial(jax.jit, static_argnums=(3))
def train_step(state, inputs, labels, loss_function):
    """
    Performs a single training step given a batch of data
    ...
    Parameters
    __________
    state : TrainState
        current train state of the model
    inputs : float32
        inputs for the batch
    labels : int32
        labels for the batch
    loss_function: str
        identifier to select loss function
    """
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, inputs)
        loss = get_loss(loss_function, logits, labels)
        return loss
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, grads

def extract_arrays(d, key, collected_arrays):
    if isinstance(d, dict):
        for k, v in d.items():
            if k in key:
                collected_arrays.append(v)
            else:
                extract_arrays(v, key, collected_arrays)
    return collected_arrays

def extract_gradients(param_dict, keys, merge):
    kernels = extract_arrays(param_dict, keys, [])
    kernels = merge_consecutive_pairs(kernels) if merge else kernels
    return kernels

def extract_kernel_and_B_arrays(param_dict, merge):
    kernels = extract_arrays(param_dict, ['kernel'], [])
    Bs = extract_arrays(param_dict, ['B'], [])
    #if kernels:
    #    kernels = kernels[1:]
    #if Bs:
    #    Bs = Bs[1:]
    kernels = merge_consecutive_pairs(kernels) if merge else kernels
    Bs = merge_consecutive_pairs(Bs) if merge else Bs
    return kernels, Bs

def flatten_array(arr):
    # Use the reshape method to flatten the array
    flattened = arr.reshape(-1)
    return flattened

def concatenate_arrays(array_list):
    # Use a list comprehension to concatenate the arrays
    concatenated_list = [item for array in array_list for item in array]
    return concatenated_list

def flatten_arrays_layerwise(arrays):
    list = [flatten_array(x) for x in arrays]
    return list

def flatten_arrays(arrays):
    list = [flatten_array(x) for x in arrays]
    return concatenate_arrays(list)

def compute_layerwise_alignments(as_, bs_):
    alignments = [compute_alignment(a, b) for a, b in zip(as_, bs_)]
    return alignments

def reorganize_dict(input_dict):
    new_dict = {'params': {}}

    for layer_key, layer_val in input_dict['params'].items():
        # Extract layer number
        layer_num = layer_key.split('_')[-1]
        
        # Iterate over items in each layer
        for param_key, param_val in layer_val.items():
            # Skip 'B' entries
            if param_key == 'B':
                continue
            
            # Construct new key as 'Dense_x'
            new_key = param_key.split('_')[0] + '_' + layer_num

            # Add or update the new dictionary
            if new_key not in new_dict['params']:
                new_dict['params'][new_key] = {}
            new_dict['params'][new_key] = param_val

    return new_dict

def compute_alignment(a,b):
    a = jnp.array(a)
    b = jnp.array(b)
    return jnp.dot(a,b)/(jnp.linalg.norm(a)*jnp.linalg.norm(b))

def get_loss(loss_function, logits, labels):
    """
    Returns the loss for network outputs and labels
    ...
    Parameters
    __________
    loss_function : str
        identifier that slects the correct loss function
    logits : float32
        outputs of the network for the batch
    labels : int32
        labels for the batch
    """
    if(loss_function == "CE"):
        return optax.softmax_cross_entropy_with_integer_labels(logits=jnp.squeeze(logits), labels=labels).mean()
    elif(loss_function == "MSE"):
        return optax.l2_loss(jnp.squeeze(logits), jnp.squeeze(labels)).mean()


def prep_batch(batch, seq_len, in_dim):
    """
    Prepares a batch of data for training.
    ...
    Parameters
    __________
    batch : tuple
        batch of data
    seq_len : int
        sequence length used when running model
    in_dim : int
        input dimension of model
    """
    inputs, labels = batch
    inputs = jnp.array(inputs.numpy()).astype(float)
    labels = jnp.array(labels.numpy()) 
    return inputs, labels





@partial(jnp.vectorize, signature="(c),()->()")
def compute_accuracy(logits, label):
    """
    Computes the accuracy of the network outputs for a batch foe a classification task
    ...
    Parameters
    __________
    logits : float32
        outputs of the network for the batch
    label : int32
        labels for the batch
    """
    return jnp.mean(jnp.argmax(logits) == label)


@partial(jax.jit, static_argnums=(3))
def eval_step(inputs, labels, state, loss_function):
    """
    Performs a single evaluation step given a batch of data
    ...
    Parameters
    __________
    inputs : float32
        inputs for the batch
    labels : int32
        labels for the batch
    state : TrainState
        current train state of the model
    loss_function: str
        identifier to select loss function
    """

    logits = state.apply_fn({"params": state.params}, inputs)
    losses = get_loss(loss_function, logits, labels)
    accs = None
    if loss_function == "CE":
        accs = compute_accuracy((jnp.squeeze(logits)), labels)
    return jnp.mean(losses), accs, logits


def validate(state, testloader, seq_len, in_dim, loss_function):
    """
    Validation function that loops over batches
    ...
    Parameters
    __________
    state : TrainState
        current train state of the model
    testloader : torch.utils.data.DataLoader
        pytorch dataloader for the test set
    seq_len : int
        sequence length used when running model
    in_dim : int
        input dimension of model
    loss_function: str
        identifier to select loss function
    """

    losses, accuracies = jnp.array([]), jnp.array([])

    for batch in tqdm(testloader):
        inputs, labels = prep_batch(batch, seq_len, in_dim)
        loss, acc, logits = eval_step(inputs, labels, state, loss_function)
        losses = jnp.append(losses, loss)
        if loss_function == "CE":
            accuracies = jnp.append(accuracies, acc)
        loss_mean = jnp.mean(losses)
    
    acc_mean = 10000000.
    if loss_function == "CE":
        acc_mean = jnp.mean(accuracies)
    return loss_mean, acc_mean



def pred_step(state, batch, seq_len, in_dim, task):
    """
    Prediction function to obtain outputs for a batch of data
    ...
    Parameters
    __________
    state : TrainState
        current train state of the model
    batch : tuple
        batch of data
    seq_len : int
        sequence length used when running model
    in_dim : int
        input dimension of model
    task : str 
        identifier for task
    """

    inputs, labels = prep_batch(batch, seq_len, in_dim)
    logits = state.apply_fn({'params': state.params}, inputs)
    if(task == "classification"):
        return jnp.squeeze(logits).argmax(axis=1)
    elif(task == "regression"):
        return logits
    else :
        print("Task not supported")
        return None
    

def plot_sample(testloader, state, seq_len, in_dim, task):
    """
    Plots a sample of the test set
    ...
    Parameters
    __________
    testloader : torch.utils.data.DataLoader
        pytorch dataloader for the test set
    state : TrainState
        current train state of the model
    seq_len : int
        sequence length used when running model
    in_dim : int
        input dimension of model
    task : str
        identifier for task
    """

    if(task == "classification"):
        return plot_mnist_sample(testloader, state, seq_len, in_dim, task)
    elif (task == "regression"):
        return plot_regression_sample(testloader, state, seq_len, in_dim, task)
    else:
        print("Task not supported")
        return None


def plot_mnist_sample(testloader, state, seq_len, in_dim, task):
    """
    Plots a sample of the test set for the MNIST dataset
    ...
    Parameters
    __________
    testloader : torch.utils.data.DataLoader
        pytorch dataloader for the test set
    state : TrainState
        current train state of the model
    seq_len : int
        sequence length used when running model
    in_dim : int
        input dimension of model
    task : str
        identifier for task
    """

    test_batch = next(iter(testloader))
    pred = pred_step(state, test_batch, seq_len, in_dim, task)

    fig, axs = plt.subplots(5, 5, figsize=(12, 12))
    inputs, labels = test_batch
    inputs = torch.reshape(inputs, (inputs.shape[0], 28, 28, 1))
    print(inputs.shape)
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(inputs[i, ..., 0], cmap='gray')
        ax.set_title(f"label={pred[i]}")
        ax.axis('off')
    plt.show()


def plot_regression_sample(testloader, state, seq_len, in_dim, task):
    """
    Plots a sample of the test set for the regression task
    ...
    Parameters
    __________
    testloader : torch.utils.data.DataLoader
        pytorch dataloader for the test set
    state : TrainState
        current train state of the model
    seq_len : int
        sequence length used when running model
    in_dim : int
        input dimension of model
    task : str
        identifier for task
    """
    
    inputs_array = np.array([])
    labels_array = np.array([])
    pred_array = np.array([])
    if in_dim != 1 | seq_len != 1:
            print("Plotting only possible for 1D inputs")
            return
    for i in range(5):
        test_batch = next(iter(testloader))
        pred = pred_step(state, test_batch, seq_len, in_dim, task)
        inputs, labels = test_batch
        labels = labels
        inputs_array = np.append(inputs_array, inputs)
        labels_array = np.append(labels_array, labels)
        pred_array = np.append(pred_array, pred)
    plt.scatter(inputs_array.flatten(), labels_array.flatten(), label="True")
    plt.scatter(inputs_array.flatten(), pred_array.flatten(), label="Pred")
    plt.show()


