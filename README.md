# bioflax

## Introduction

Backpropagation [1], combined with stochastic gradient, is a powerful and widely used algorithm for learning via artificial neural networks. Yet, for several reasons it's not plausible for this algorithm to run in the brain in a similar fashion as pointed out quickly after it's introduction [2,3]. One of the most prominent issues is the so called weight transport problem. A typical deep-learning architecture works in two phases. Firstly, in the forward pass inputs are fed forward through the layers of the network from input to output layer along the forwrd path to produce an output. This output is compared to the target and the resulting error is passed back through the network in the backward pass from output to input layer along the feedback path. On the fly, the backward pass generates error signals for each layer used to compute the parameter gradients which constitute the updates.

The forward pass generates outputs $y_{l+1}$ of layer $l+1$ given inputs $y_l$ according he follwoing update rule:
$$y_{l+1} = \phi(W_{l+1}y_l+b_{l+1})$$
The backward pass feeds the error $\delta_{l+1}$ at layer $l+1$ back to layer $l$ to generate $\delta_l$ according to the backpropagation equation:
$$\delta_l = \phi'(y_l)W_{l+1}^T\delta_{l+1}$$

The weight transport problem stems from the fact, that the matrices $W_l$ appear on both the forward and the backwards pass, and the requried symmetry is unlikely to exist in the brain.

This repository provides JAX implementations of alternative algorithms which decouple forward and backward pass. The algorithms are implemented via extending Flax modules and defining a custom backward pass. Doing so they can be integrated with other Flax modules and respective functionality in a flawless manner.

The implemented algortihms are

- Feedback Alignment [4]
- Direct Feedback Alignment [5]
- Kollen Pollack [6]

## References

[1] David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams. Learning representationsby back-propagating errors. Nature, 323(6088), 1986.

[2] Stephen Grossberg. Competitive learning: From interactive activation to adaptive resonance.
Cognitive science, 11(1):23–63, 1987.

[3] Francis Crick. The recent excitement about neural networks. Nature, 337:129–132, 1989.
