![Python Unit Tests](https://github.com/yschimpf/bioflax/actions/workflows/run_tests.yml/badge.svg?event=push)
# bioflax

This repository provides an unofficial JAX implementation of biologically plausible deep learning algorithms. In particular, Feedback Alignment, Kolen-Pollack, and Direct Feedback Alignment are implemented and a framework for running experiments with them is given.

Content:
1. [Introduction](#introduction)
2. [Feedback Alignment [4]](#feedback-alignment-4)
   - [Theory](#theory)
   - [Code](#code)
4. [Kolen Pollack [6]](#kolen-pollack-6)
   - [Theory](#theory-1)
   - [Code](#code-1)
5. [Direct Feedback Alignment [5]](#direct-feedback-alignment-5)
   - [Theory](#theory-2)
   - [Code](#code-2)
6. [Data Download](#data-download)
7. [Repository structure](#repository-structure)
8. [Use layers](#use-layers)
9. [Run experiments](#run-experiments)
10. [References](#references)
## Introduction

Backpropagation [1], combined with stochastic gradient, is a powerful and widely used algorithm for learning via artificial neural networks. Yet, as pointed out quickly after its introduction, for several reasons it's not plausible for this algorithm to run in the brain in a similar fashion [2,3]. One of the most prominent issues is the so-called weight transport problem. A typical deep-learning architecture works in two phases. Firstly, in the forward pass inputs are fed forward through the network from the input to the output layer along the forward path to produce an output. This output is compared to a target using a chosen loss function $L$ and the resulting error is passed back through the network in the backward pass from the output to the input layer along the feedback path. On the fly, the backward pass generates error signals for each layer to compute the parameter gradients that constitute the updates.

The forward pass generates outputs $y_{l+1}$ of layer $l+1$ given inputs $y_l$ according he follwoing update rule, where $\phi$ is a (mostly) non-linear activation function:
$$y_{l+1} = \phi(W_{l+1}y_l+b_{l+1})$$
The backward pass feeds the error $\delta_{l+1}$ at layer $l+1$ back to layer $l$ to generate $\delta_l$ according to the backpropagation equation.
$$\delta_l = \phi'(y_l)W_{l+1}^T\delta_{l+1}$$
Using those $\delta$ updates for $W$ are computed on the fly:
$$\frac{\partial L}{\partial W_l} = \delta_l y_{l-1}^T$$
For some learning rate $\eta$  this finally yields:
$$W_l^{t+1} = W_l^t - \eta\frac{\partial L}{\partial W_l}$$
A typical network chains together multiple such layers and schematically looks like this.

<div align="center">
  <img src="/docs/figures/BP.png" alt="BP Neural Network" width="150"/>
  <br>
  <em>Figure 1: Neural Network using BP. Taken from [5].</em>
</div>

The weight transport problem stems from the fact, that the matrices $W_l$ appear on both the forward and the backward pass, and the required symmetry is unlikely to exist in the brain. That's because in the brain the synapses in the forward and feedback paths are physically distinct and there is no known way in which they could coordinate themselves to reflect the symmetry.

This repository provides JAX implementations of alternative algorithms that decouple forward and backward passes. The algorithms are implemented by extending [Flax](https://github.com/google/flax) modules and defining a custom backward pass. By doing so they can be integrated with other Flax modules and respective functionality flawlessly.

The implemented algorithms are
- Feedback Alignment (FA) [4]
- Kolen Pollack (KP) [6]
- Direct Feedback Alignment (DFA) [5]

## Feedback Alignment [4]
### Theory
1. The Feedback alignment algorithm decouples feedforward and feedback path by using fixed random weights $B$ on the feedback path. The staggering result is that even though the forward and feedback weights are independent (in the beginning), in expectation uncorrelated (in the beginning), and hence the gradient computation is heavily biased, the network essentially learns how to learn using the fixed random weights $B$. This happens because information about the backward weights $B$ flows into the forward weights W via the update resulting in increasing alignment with the true gradients. Concretely, the $\delta$ computation of BP is modified yielding: 

- The forward pass generates outputs $y_{l+1}$ of layer $l+1$ given inputs $y_l$ according the following update rule:
$$y_{l+1} = \phi(W_{l+1}y_l+b_{l+1})$$
- The backward pass feeds the error $\delta_{l+1}$ at layer $l+1$ back to layer $l$ to generate $\delta_l$ using B instead of $W^T$
$$\delta_l = \phi'(y_l)B_{l}\delta_{l+1}$$
- Using those $\delta$ updates for $W$ are computed on the fly:
$$\frac{\partial L}{\partial W_l} = \delta_l y_{l-1}^T$$
- For some learning rate $\eta$  this finally yields:
$$W_l^{t+1} = W_l^t - \eta\frac{\partial L}{\partial W_l}$$

<div align="center">
  <img src="/docs/figures/FA.png" alt="FA Neural Network" width="150"/>
  <br>
  <em>Figure 1: Neural Network using FA. Taken from [5].</em>
</div>

### Code
The code is in the [model.py](/bioflax/model.py) file. The functionality is implemented via a custom Flax linen module RandomDenseLinearFA. More specifically, a [custom_vjp](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/_autosummary/flax.linen.custom_vjp.html) function is defined that uses a standard Dense layer on the forward path but passes the error through the layer by multiplying with $B$ instead of $W$ on the backward path. Note that the way Flax works the nonlinearity is independent of the layer and hence will be handled by auto differentiation when composing the entire backward path. Overall, this extension perfectly integrates into the Flax framework.

## Kolen Pollack [6]
### Theory
KP uses the exact same architecture as FA. While it is fascinating that with FA the network learns how to make use of the fixed feedback weights $B$, the obvious issue from an optimization point of view is that the gradients are not exact anymore. Kolen Pollack doesn't leave the feedback weights $B$ untouched as does FA but uses a simple update rule which ensures that forward matrices $W$ and feedback matrices $B$ converge. By that gradient alignment is sped up compared to FA and reaches higher levels as well.

The computations for forward and feedback paths remain the same as for FA:

- The forward pass generates outputs $y_{l+1}$ of layer $l+1$ given inputs $y_l$ according the following update rule:
$$y_{l+1} = \phi(W_{l+1}y_l+b_{l+1})$$
- The backward pass feeds the error $\delta_{l+1}$ at layer $l+1$ back to layer $l$ to generate $\delta_l$ using B instead of $W^T$
$$\delta_l = \phi'(y_l)B_{l}\delta_{l+1}$$

<div align="center">
  <img src="/docs/figures/FA.png" alt="KP Neural Network" width="150"/>
  <br>
  <em>Figure 1: Neural Network using KP. Taken from [5].</em>
</div>

While the update rules covered so far and the graphical representation is identical to FA, the difference comes in the form of update rules for the feedback weights $B$ which are now updated as follows: 

- Forward weights $W$ are updated according to:
$$\Delta W  = - \eta_W \delta_{l+1}y_l^T - \lambda W_{l+1}$$
- Backward weights $B$ are updated according to: 
$$\Delta B = - \eta_W y_l \delta_{l+1}^T - \lambda B_{l+1}$$
Note that $B$ has the shape of $W^T$ hence the transposing in the update rule. $\eta_W$ is the learning rate and $\lambda$ the weight decay. One can easily verify that
$$W(t) - B(t) = (1-\lambda)^t[W(0)-B(0)]$$
where $B(t)$ and $W(t)$ refers to the respective matrix at step t. Hence, as long as $0 < \lambda < 1$ $W$ and $B$ matrices will converge, and with them the respective gradients. 

### Code
The code is contained in the [model.py](/bioflax/model.py) file. The functionality is implemented via a custom Flax linen module RandomDenseLinearKP. More specifically, a [custom_vjp](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/_autosummary/flax.linen.custom_vjp.html) function is defined that uses a standard Dense layer on the forward path but passes the error through the layer by multiplying with $B$ instead of $W$ on the backward path. Moreover, the backward pass does not return a zero gradient for $B$ anymore, as it does in RandomDenseLinearKP but sets the gradient of $B$ to that of $W$ (transposed). Using a respective weight decay in the optimizer will yield the KP updates of $B$.

## Direct Feedback Alignment [5]
### Theory
The direct feedback alignment algorithm decouples the feedforward and feedback path by using fixed random weights $B$ on the feedback path. The difference compared to FA is that the error is no longer backpropagated through all layers until it reaches a specific layer. Instead, it is directly propagated to the layer $l$ from the output layer via a single matrix $B_l$. This matrix obviously no longer has the dimension $W^T$ but its shape is defined by the dimension of the hidden layer and the output layer together. The update rule stays the same as for FA but the matrices $B$ have a different meaning as explained: 

- The forward pass generates outputs $y_{l+1}$ of layer $l+1$ given inputs $y_l$ according the following update rule:
$$y_{l+1} = \phi(W_{l+1}y_l+b_{l+1})$$
- The backward pass feeds the error $\delta_{l+1}$ at layer $l+1$ back to layer $l$ to generate $\delta_l$ using B instead of $W^T$
$$\delta_l = \phi'(y_l)B_{l}\delta_{l+1}$$

<div align="center">
  <img src="/docs/figures/DFA.png" alt="DFA Neural Network" width="150"/>
  <br>
  <em>Figure 1: Neural Network using DFA. Taken from [5].</em>
</div>

The forward weights are then updated according to:
$$\Delta W  = - \eta_W \delta_{l+1}y_l^T - \lambda W_{l+1}$$

### Code
The code is contained in the [model.py](/bioflax/model.py) file. The functionality is implemented via two custom Flax linen modules RandomDenseLinearDFAOutput and RandomDenseLinearDFAHidden. More specifically, a [custom_vjp](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/_autosummary/flax.linen.custom_vjp.html) is defined in each to compute the correct updates. The modules are less generic compared to standard Flax linen modules for the reason that to still integrate with the framework which is built for backpropagation the auto differentiation must be tricked in a more intriguing way. In its nature DFA doesn't propagate anything through layers but directly to the respective layers. In the code, this is achieved by propagating the error at the output layer directly to the next layer in the custom_vjp function. In doing so, every layer can apply its own matrix $B$ to that error. The remaining problem is that now the derivative of the activation should also no longer be applied on the path but layerwise. To circumvent this issue the activations are integrated into the layer. That way, they can be considered for the local computations but neglected for the global pass of the error. This is a little different compared to the separation of layer and activation as usual in Flax. Yet, once the layers are defined with this knowledge all other features of the framework can flawlessly be used once again. In the output layer, no matrix $B$ must be applied to the error. This is done via defining a separate module for the output layer, which also doesn't use an activation as is usual in neural networks for the output layer, but could be achieved by setting $B$ to the identity matrix of the correct dimension as well.

## Requirements & Installation
To run the code on your own machine, run `pip install -r requirements.txt`. For the GPU installation of JAX, which is a little more involved please refer to the [JAX installation instructions](https://github.com/google/jax#installation). For dataloading PyTorch is needed as well. It needs to be installed separately and more importantly, the CPU version needs to be installed because of inference issues with JAX. Please refer to [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

## Data Download
In this first release, the code has built-in support to run experiments on the MNIST dataset, a teacher-student dataset, and a sin-propagation dataset. None of these require explicit data download. MNIST will be downloaded automatically on execution and the teacher-student dataset as well as the sin-propagation dataset are created on the fly. 

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
Detached from the rest of the code the Flax modules - the biological layer implementations respectively - can be used to define custom modules (Dense networks) that run with the respective deep learning algorithm. For example, a two-layer Dense network for the available algorithms that perfectly integrates with the Flax framework can be created as follows: 
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

# Note the differences for DFA. In particular, activations must be handed to the hidden layers and mustn't be
# on the computational path elsewhere. Secondly, the Hidden layers need the final output dimension as input
# apart from the layers features in activation.
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
which will result in a run with the default configuration. The settable arguments (and their respective default values) are:
```
--batch_size           Batch size during training. Type: int, Default: 32
--val_split            Percentage of training samples for validation. Type: float, Default: 0.1
--epochs               Number of training epochs. Type: int, Default: 5
--mode                 Training mode. Choices: ['bp', 'fa', 'kp', 'dfa']. Type: str, Default: "fa"
--activations          Activation functions for each layer. Type: str (list), Default: ["relu", "relu"]
--hidden_layers        Neurons in each hidden layer. Type: int (list), Default: [500, 500]
--jax_seed             Seed for JAX RNG. Type: int, Default: 0
--dataset              Dataset for training. Choices: ['mnist', 'sinprop', 'teacher']. Type: str, Default: "mnist"
--in_dim               Input vector dimension. Type: int, Default: 1
--seq_len              Length of input sequence. Type: int, Default: 1
--output_features      Number of output features. Type: int, Default: 10
--train_set_size       Size of the training set. Type: int, Default: 50
--test_set_size        Size of the test set. Type: int, Default: 10
--lr                   Learning rate. Type: float, Default: 0.02
--momentum             Momentum value. Type: float, Default: 0
--plot                 Enable/disable plotting. Type: boolean, Default: True
--compute_alignments   Enable/disable alignment computations. Type: boolean, Default: True
--wandb_project        Weights & Biases project name. Type: str, Default: "test_project"
--use_wandb            Enable/disable Weights & Biases for logging. Type: boolean, Default: True
--n                    Batches for alignment averaging. Type: int, Default: 5
--wandb_entity         Weights & Biases entity name. Type: str, Default: "bioflax"
   
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
