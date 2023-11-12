![Layer Unit Tests](https://github.com/yschimpf/bioflax/actions/workflows/run_tests.yml/badge.svg?event=push)

# bioflax

This repository provides an unofficial JAX implementation of biologically plausible deep learning algorithms. In particular, Feedback Alignment, Kolen-Pollack, and Direct Feedback Alignment are implemented and a framework for running experiments with them is given.

Content:

- [bioflax](#bioflax)
  - [Introduction](#introduction)
  - [Implementation overview](#implementation-overview)
    - [Theory Overview](#theory-overview)
    - [Code](#code)
  - [Requirements \& Installation](#requirements--installation)
  - [Data Download](#data-download)
  - [Repository structure](#repository-structure)
  - [Use layers](#use-layers)
  - [Run experiments](#run-experiments)
  - [References](#references)

## Introduction

Backpropagation [1], combined with stochastic gradient, is a powerful and widely used algorithm for learning via artificial neural networks. Yet, as pointed out quickly after its introduction, for several reasons it's not plausible for this algorithm to run in the brain in a similar fashion [2,3]. One of the most prominent issues is the so-called weight transport problem. A typical deep-learning architecture works in two phases: The forward pass and the feedback path.

The forward pass generates outputs $y_{l+1}$ of layer $l+1$ given inputs $y_l$ according he following update rule, where $\phi$ is a (mostly) non-linear activation function:
$$y_{l+1} = \phi(W_{l+1}y_l+b_{l+1})\ \ \ (1)$$
The backward pass feeds the error $\delta_{l+1}$ at layer $l+1$ back to layer $l$ to generate $\delta_l$ according to the backpropagation equation.
$$\delta_l = \phi'(y_l)W_{l+1}^T\delta_{l+1} \ \ \ (2)$$
Using those $\delta$ updates for $W$ are computed on the fly:
$$\frac{\partial L}{\partial W_l} = \delta_l y_{l-1}^T \ \ \ (3)$$
For some learning rate $\eta$ this finally yields:
$$W_l^{t+1} = W_l^t - \eta\frac{\partial L}{\partial W_l} \ \ \ (4)$$
A typical network chains together multiple such layers and schematically looks like this.

<div align="center">
  <img src="/docs/figures/BP.png" alt="BP Neural Network" width="150"/>
  <br>
  <em>Figure 1: Neural Network using BP. Taken from [5].</em>
</div>

The weight transport problem stems from the fact, that the matrices $W_l$ appear on both the forward and the backward pass, and the required symmetry is unlikely to exist in the brain. That's because in the brain the synapses in the forward and feedback paths are physically distinct and there is no known way in which they could coordinate themselves to reflect the symmetry.

## Implementation overview

The follwoing algorithms solving the weight transport problem are implemented:

- Feedback Alignment (FA) [4]
- Kolen Pollack (KP) [6]
- Direct Feedback Alignment (DFA) [5]

### Theory Overview

How do these alogithms circumvent the weight transport problem? The key concept in each case is to decouple the forward pass from the backward pass by using fixed random weights for the backward pass.

Feedback alignment and Kolen-Pollack both use the following structure:

<div align="center">
  <img src="/docs/figures/FA.png" alt="FA Neural Network" width="150"/>
  <br>
  <em>Figure 1: Neural Network using FA/KP. Taken from [5].</em>
</div>

For Feedback Alignment all computations remain the same as for Backpropagation except for (2) which now uses the fixed backward weights $B$.
$$\delta_l = \phi'(y_l)B_{l}\delta_{l+1} \ \ \ (2.1)$$
The staggering result is that while the $W$ and $B$ are initially uncorrelated, the network learns how to learn using these fixed feedback weights and achieves good optimization performance.

For Kolen-Pollack this same modification is applied as well. Apart from that, weight updates with specific structure are imposed on $B$. In doing so Kolen-Pollack makes the $W$ and $B$ converge which ultimately speeds up alignment of the gradients. In particular the weight updates (3) and (4) now change to
$$\frac{\partial L}{\partial W_l} = \delta_l y_{l-1}^T \ \ \ (3.1)$$
$$\frac{\partial L}{\partial B_l} =  y_{l-1}\delta_l^T \ \ \ (3.2)$$
$$W_l^{t+1}  = W_l^t - \eta_W^t\frac{\partial L}{\partial W_l^t} - \lambda W_{l}^t  \ \ \ (4.1)$$
$$B_l^{t+1} = B_l^t - \eta_W^t \frac{\partial L}{\partial B_l^t} - \lambda B_{l}^t \ \ \ (4.2)$$
Here $\eta_W$ is the learning rate and $\lambda$ the weight decay. One can easily verify that
$$W^t - {B^t}^T = (1-\lambda)^t[W^0-{B^0}^T] \ \ \ (5)$$
where $B(t)$ and $W(t)$ refers to the respective matrix at step t. Hence, as long as $0 < \lambda < 1$ $W$ and $B$ matrices will converge, and with them the respective gradients.

Direct Feedback Alignment uses yet another network architeture:

<div align="center">
  <img src="/docs/figures/DFA.png" alt="DFA Neural Network" width="150"/>
  <br>
  <em>Figure 1: Neural Network using DFA. Taken from [5].</em>
</div>
Now errors are not backpropagated through all layers but only through one fixed random matrix $B$ per layer directly to that layer. All computations stay the same as for Feedback Alignment and only the meaning of $B$ in (2.1) changes.

### Code

The code is in the [model.py](/bioflax/model.py) file. The functionality is implemented via a custom Flax linen module for each mode:

- RandomDenseLayerFA
- RandomDenseLayerKP
- RandomDenseLayerDFAOutput
- RandomDenseLayerDFAHidden

More specifically, a [custom_vjp](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/_autosummary/flax.linen.custom_vjp.html) is defined for each algorithm that on the forward pass uses a standard dense layer and on the backward pass uses the custom update rules as defined above. One technicality regarding the DFA layers is that since is nature is nothing like backpropagation anymore tricks need to be applied which require the activation function to be specified as part of the layer and the layer to contain the final output dimensions. Another side product is the neccesity to split into output and hidden layer for DFA. The tricks and the entire implementation serve the purpose of flawlessly intgrating with the Flax framework. Modules can be used exactly like standard Flax modules in deep learning architectures.

## Requirements & Installation

To run the code on your own machine, run `pip install -r requirements.txt`.

For the GPU installation of JAX, which is a little more involved please refer to the [JAX installation instructions](https://github.com/google/jax#installation).

For dataloading PyTorch is needed. It has to be installed separately and more importantly, the CPU version must be installed because of inference issues with JAX. Please refer to [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

## Data Download

In this first release, the code has built-in support to run experiments on the MNIST dataset, a teacher-student dataset, and a sin-regression dataset. None of these require explicit data download. MNIST will be downloaded automatically on execution and the teacher-student dataset as well as the sin-propagation dataset are created on the fly.

## Repository structure

Directories and files that ship with the GitHub repo

```
.github/workflows/
    run_tests.yml            Workflow to run unit tests that ensure the functionality of the layer implementations.
bioflax/
    dataloading.py           Dataloading functions.
    metric_computation.py    Functions for metric computation
    model.py                 Layer implementations for FA, KP, and DFA.
    test_layers.py           Unit tests for layer implementations.
    train.py                 Training loop code.
    train_helpers.py         Functions for optimization, training, and evaluation steps.
requirements.txt             Requirements for running the code.
run_train.py                 Training loop entry point.

```

Directories that may be created on-the-fly:

```
data/MNIST/raw               Raw MNIST data as downloaded.
wandb/                       Local WandB log file
```

## Use layers

Separately from the rest of the code, the Flax custom modules - the biological layer implementations respectively - can be used to define custom modules (Dense networks) that run with the respective deep learning algorithm. For example, a two-layer Dense network with sigmoid activation in the hidden layer for the available algorithms that perfectly integrates with the Flax framework can be created as follows:

```python
import jax
import flax.linen as nn
from model import (
    RandomDenseLinearFA,
    RandomDenseLinearKP,
    RandomDenseLinearDFAOutput,
    RandomDenseLinearDFAHidden
)

class NetworkFA(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = RandomDenseLinearFA(15)(x)
                x = nn.sigmoid(x)
                x = RandomDenseLinearFA(10)(x)
                return x

class NetworkKP(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = RandomDenseLinearKP(15)(x)
                x = nn.sigmoid(x)
                x = RandomDenseLinearKP(10)(x)
                return x

# Note the differences for DFA. In particular, activations must be handed to the hidden layers and mustn't be on the
# computational path elsewhere. Secondly, the hidden layers need the final output dimension as an additional input
 class NetworkDFA(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = RandomDenseLinearDFAHidden(15, 10, nn.sigmoid)(x)
                x = RandomDenseLinearDFAOutput(10)(x)
                return x

```

If you need help on how to use modules for actual learning in Flax please refer to the [Flax doxumentation](https://flax.readthedocs.io/en/latest/).

## Run experiments

To run an experiment simply execute

```
python run_train.py
```

which will result in a run with the default configuration. For information about the arguments and their default settings excute on of the following commands

```
python run_train.py --help
python run_train.py --h
```

## References

[1] David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams. Learning representations by back-propagating errors. Nature, 323(6088), 1986.

[2] Stephen Grossberg. Competitive learning: From interactive activation to adaptive resonance.
Cognitive science, 11(1):23–63, 1987.

[3] Francis Crick. The recent excitement about neural networks. Nature, 337:129–132, 1989.

[4] Timothy P. Lillicrap, Daniel Cownden, Douglas B. Tweed, and Colin J. Akerman. Random
synaptic feedback weights support error backpropagation for deep learning. Nature Communications,
7(1), 2016.

[5] Arild Nøkland. Direct feedback alignment provides learning in deep neural networks. In
Advances in Neural Information Processing Systems, 2016.

[6] J.F. Kolen and J.B. Pollack. Backpropagation without weight transport. In Proceedings of
1994 IEEE International Conference on Neural Networks, 1994.
