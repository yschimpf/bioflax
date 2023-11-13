![Layer Unit Tests](https://github.com/yschimpf/bioflax/actions/workflows/run_tests.yml/badge.svg?event=push)

# bioflax

Content:

- [bioflax](#bioflax)
  - [Introduction](#introduction)
  - [Requirements \& Installation](#requirements--installation)
  - [Data Download](#data-download)
  - [Repository structure](#repository-structure)
  - [Use layers](#use-layers)
  - [Run experiments](#run-experiments)
  - [References](#references)

## Introduction

bioflax provides an unofficial JAX implementation of biologically plausible deep learning algorithms. In particular, Feedback Alignment, Kolen-Pollack, and Direct Feedback alignment are implemented and a framework for running experiments with them is given. The code implemnts custom [Flax](https://flax.readthedocs.io/en/latest/quick_start.html) modules, which flawlessly integrate with the Flax framework.

The respective algorihms network structures are depicted in the following scheme. For a more detailed overview please refer to the [docs](/docs/README.md).

| Backpropagation                                                       | Feedback Alignment &<br> Kolen-Pollack                               | Direct Feedback Alignment                                               |
| --------------------------------------------------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| <img src="/docs/figures/BP.png" alt="BP Neural Network" width="105"/> | <img src="/docs/figures/FA.png" alt="FA Neural Network" width="95"/> | <img src="/docs/figures/DFA.png" alt="DFA Neural Network" width="100"/> |

<em> Network architectures for different algorithms. Taken from [5]. </em>

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
