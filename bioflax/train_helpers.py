from functools import partial
import jax
import jax.numpy as jnp
from jax.nn import one_hot
from tqdm import tqdm
from flax.training import train_state
import optax
import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import Any

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
    #print("params in train state: ", params["Dense_0"]["kernel"].shape)
    tx = optax.sgd(learning_rate=lr, momentum=momentum) #make more generic later
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def train_epoch(state, model, trainloader, seq_len, in_dim, loss_function):
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

    for batch in tqdm(trainloader):
        inputs, labels = prep_batch(batch, seq_len, in_dim)
        state, loss = train_step(state, inputs, labels, loss_function)
        batch_losses.append(loss)  # log loss value

    # Return average loss over batches
    return state, jnp.mean(jnp.array(batch_losses))

def get_loss(loss_function, logits, labels):
    """
    Returns the loss for network outputs and labels
    ...
    Parameters
    __________
    loss_function : str
        identifier that slects the correct loss function
    logits : -- have to look up --
        outputs of the network for the batch
    labels : -- have to look up --
        labels for the batch
    """
    ###print("Logits shape in get_loss: ", logits.shape)
    ##print("Logits in get_loss: ", logits)
    #print("Labels shape in get loss: ", labels.shape)
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
    #print("Shape inputs in prep_batch:", inputs.shape)
    #print("Shape labels in prep_batch:", labels.shape)
    #print("Shape in prep_batch:", inputs.shape)
    inputs = jnp.array(inputs.numpy()).astype(float)  # convert to jax
    labels = jnp.array(labels.numpy())  # convert to jax

    # Make all batches have same sequence length
    #num_pad = seq_len - inputs.shape[1]
    #if num_pad > 0:
    #    inputs = jnp.pad(inputs, ((0, 0), (0, num_pad)), "constant", constant_values=(0,))

    # Inputs size is [n_batch, seq_len] or [n_batch, seq_len, in_dim].
    # If there are not three dimensions and trailing dimension is not equal to in_dim then
    # transform into one-hot.  This should be a fairly reliable fix.
    #if (inputs.ndim < 3) and (inputs.shape[-1] != in_dim):
    #    inputs = one_hot(inputs, in_dim)
    return inputs, labels

@partial(jax.jit, static_argnums=(3))
def train_step(state, inputs, labels, loss_function):
    """
    Performs a single training step given a batch of data
    ...
    Parameters
    __________
    state : TrainState
        current train state of the model
    inputs : -- have to look up --
        inputs for the batch
    labels : -- have to look up --
        labels for the batch
    loss_function: str
        identifier to select loss function
    """
    def loss_fn(params):
        #print(params)
        #print(inputs.shape)
        #with jax.checking_leaks():
        logits = state.apply_fn({'params': params}, inputs) #model.apply genau gleicher fehler der fix geht also nicht: hier vllt. model.apply aber dann muss ich glaub im model die attribute als parameter machen
        loss = get_loss(loss_function, logits, labels)
        return loss
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@partial(jnp.vectorize, signature="(c),()->()")
def compute_accuracy(logits, label):
    return jnp.mean(jnp.argmax(logits) == label)

@partial(jax.jit, static_argnums=(3))
def eval_step(inputs, labels, state, loss_function):
    logits = state.apply_fn({"params": state.params}, inputs)
    losses = get_loss(loss_function, logits, labels)
    accs = None
    if loss_function == "CE":
        accs = compute_accuracy((jnp.squeeze(logits)), labels)
    return jnp.mean(losses), accs, logits


def validate(state, testloader, seq_len, in_dim, loss_function):
    """Validation function that loops over batches"""
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
  inputs, labels = prep_batch(batch, seq_len, in_dim)
  logits = state.apply_fn({'params': state.params}, inputs)
  if(task == "classification"):
    return jnp.squeeze(logits).argmax(axis=1)
  elif(task == "regression"):
    print("Logits shape in pred_step: ", logits)
    return logits
  else :
    print("Task not supported")
    return None

def plot_mnist_sample(testloader, state, seq_len, in_dim, task):
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
    #labels sind blau
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