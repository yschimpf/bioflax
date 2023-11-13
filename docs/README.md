# bioflax docs

Content:

- [bioflax docs](#bioflax-docs)
  - [Introduction](#introduction)
  - [Feedback Alignment \[4\]](#feedback-alignment-4)
    - [Theory](#theory)
    - [Code](#code)
  - [Kolen Pollack \[6\]](#kolen-pollack-6)
    - [Theory](#theory-1)
    - [Code](#code-1)
  - [Direct Feedback Alignment \[5\]](#direct-feedback-alignment-5)
    - [Theory](#theory-2)
    - [Code](#code-2)
  - [References](#references)

## Introduction

Backpropagation [1], combined with stochastic gradient, is a powerful and widely used algorithm for learning via artificial neural networks. Yet, as pointed out quickly after its introduction, for several reasons it's not plausible for this algorithm to run in the brain in a similar fashion [2,3]. One of the most prominent issues is the so-called weight transport problem. A typical deep-learning architecture works in two phases. Firstly, in the forward pass inputs are fed forward through the network from the input to the output layer along the forward path to produce an output. This output is compared to a target using a chosen loss function $L$ and the resulting error is passed back through the network in the backward pass from the output to the input layer along the feedback path. On the fly, the backward pass generates error signals for each layer to compute the parameter gradients that constitute the updates.

The forward pass generates outputs $h_{l+1}$ of layer $l+1$ given inputs $h_l$ according he follwoing update rule, where $\phi$ is a (usually) non-linear activation function:
$$a_{l+1} = W_{l+1}h_l+b_{l+1} \ \ \ (1.1)$$
$$h_{l+1} = \phi(a_{l+1}) \ \ \ (1.2)$$
The backward pass feeds the error $\delta_{l+1}$ at layer $l+1$ back to layer $l$ to generate $\delta_l$ according to the backpropagation equation:
$$\delta_l = \phi'(a_l) \odot W_{l+1}^T\delta_{l+1} \ \ \ (2)$$
Using those $\delta$, updates for $W$ are computed on the fly:
$$\frac{\partial L}{\partial W_l} = \delta_l h_{l-1}^T \ \ \ (3)$$
For some learning rate $\eta$, this finally yields:
$$W_l^{t+1} = W_l^t - \eta\frac{\partial L}{\partial W_l^t} \ \ \ (4)$$
A typical network chains together multiple such layers and schematically looks like this.

<div align="center">
  <img src="/docs/figures/BP.png" alt="BP Neural Network" width="150"/>
  <br>
  <em>Figure 1: Neural Network using BP. Taken from [5].</em>
</div>

The weight transport problem stems from the fact that the matrices $W_l$ appear on both the forward and the backward pass, and the required symmetry is unlikely to exist in the brain. That's because in the brain the synapses in the forward and feedback paths are physically distinct and axons can only send information in one direction. Given these constraints, there is no known way in which this symmetry could be reflected.

This repository provides JAX implementations of biologically more plausible algorithms that decouple forward and backward passes. The algorithms are implemented by extending [Flax](https://github.com/google/flax) modules and defining a custom backward pass. By doing so they can be integrated with other Flax modules and respective functionality flawlessly.

The implemented algorithms are

- Feedback Alignment (FA) [4]
- Kolen Pollack (KP) [6]
- Direct Feedback Alignment (DFA) [5]

## Feedback Alignment [4]

### Theory

The Feedback alignment algorithm decouples feedforward and feedback path by using fixed random weights $B$ on the feedback path. The staggering result is that, even though the forward and feedback weights are in expectation uncorrelated (in the beginning), the network essentially learns how to learn using the fixed random weights $B$. That's somewhat unintuitive because the gradients computed are not aligned with the true gradients (in the beginning). The explanation is that information about the backward weights $B$ flows into the forward weights W via the updates. This results in increasing alignment of the computed gradients with the true gradients. Concretely, the $\delta$ computation of BP, i.e. equation (2), is modified as follows:
$$\delta_l = \phi'(a_l) \odot B_{l}\delta_{l+1} \ \ \ (2.1)$$

The visual representation of the network hence becomes:

<div align="center">
  <img src="/docs/figures/FA.png" alt="FA Neural Network" width="150"/>
  <br>
  <em>Figure 1: Neural Network using FA. Taken from [5].</em>
</div>

### Code

The code is in the [model.py](/bioflax/model.py) file. The functionality is implemented via a custom Flax linen module RandomDenseLinearFA. More specifically, a [custom_vjp](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/_autosummary/flax.linen.custom_vjp.html) function is defined that uses a standard Dense layer on the forward path but passes the error through the layer by multiplying with $B$ instead of $W^T$ on the backward path. Note that the way Flax works the nonlinearity is independent of the layer and hence will be handled by auto differentiation when composing the entire backward path. Overall, this extension perfectly integrates into the Flax framework.

## Kolen Pollack [6]

### Theory

KP uses the exact same architecture as FA. While it is fascinating that with FA the network learns how to make use of the fixed feedback weights $B$, the obvious issue from an optimization point of view is that the gradients are not exact anymore. To alleviate this, Kolen Pollack uses a simple update rule which ensures that forward matrices $W$ and feedback matrices $B$ converge. By that gradient alignment is sped up compared to FA and ultimately reaches higher levels.

The computations for forward and feedback paths remain the same as for FA. That is on the forward path (1.1) and (1.2) according to BP (and FA) are used and the backward path follows (2.1) from FA.
The only difference compared to FA is that now the $B$ are updated as well and (3) from BP changes to:
$$W_l^{t+1}  = W_l^t - \eta_W^t\delta_l h_{l-1}^T - \lambda W_{l}^t \ \ \ (3.1)$$
$$B_l^{t+1} = B_l^t - \eta_W^t h_{l-1}\delta_l^T - \lambda B_{l}^t \ \ \ (3.2)$$
Here $\eta_W$ is the learning rate and $\lambda$ is the weight decay. One can easily verify that
$$W^t - {B^t}^T = (1-\lambda)^t[W^0-{B^0}^T] \ \ \ (5)$$
where $B^t$ and $W^t$ refer to the respective matrix at step t. Hence, as long as $0 < \lambda < 1, $W$ and $B$ matrices will converge, and with them the respective gradients.

The visual representation of the network stays the same as for FA.

<div align="center">
  <img src="/docs/figures/FA.png" alt="KP Neural Network" width="150"/>
  <br>
  <em>Figure 1: Neural Network using KP. Taken from [5].</em>
</div>

### Code

The code is in the [model.py](/bioflax/model.py) file. The functionality is implemented via a custom Flax linen module RandomDenseLinearKP. More specifically, a [custom_vjp](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/_autosummary/flax.linen.custom_vjp.html) function is defined that uses a standard Dense layer on the forward path but passes the error through the layer by multiplying with $B$ instead of $W$ on the backward path. Moreover, the backward pass does not return a zero gradient for $B$ anymore, as it does in RandomDenseLinearKP but sets the gradient of $B$ to that of $W$ (transposed). Using a respective weight decay in the optimizer will yield the KP updates of $B$.

## Direct Feedback Alignment [5]

### Theory

The direct feedback alignment algorithm decouples the feedforward and feedback path by using fixed random weights $B$ on the feedback path. The difference compared to FA is that the error is no longer backpropagated through all layers until it reaches a specific layer. Instead, it is directly propagated to the layer $l$ from the output layer via a single matrix $B_l$. This matrix obviously no longer has the dimension $W^T$ but its shape is defined by the dimension of the hidden layer and the output layer together. All update rules except for (2) stay the same as for BP and (2) becomes:
$$\delta_l = \phi'(a_l) \odot B_{l}\delta_{L} \ \ \ (2.2)$$
Note that the only difference compared to (2.1) for FA is that now all layers directly receive $\delta_L$, the error of the output layer, without it passing through all layers beforehand.

<div align="center">
  <img src="/docs/figures/DFA.png" alt="DFA Neural Network" width="150"/>
  <br>
  <em>Figure 1: Neural Network using DFA. Taken from [5].</em>
</div>

### Code

The code is contained in the [model.py](/bioflax/model.py) file. The functionality is implemented via two custom Flax linen modules RandomDenseLinearDFAOutput and RandomDenseLinearDFAHidden. More specifically, a [custom_vjp](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/_autosummary/flax.linen.custom_vjp.html) is defined in each to compute the correct updates. The modules are less generic compared to standard Flax linen modules because Flax is intended for backpropagation-like behavior. However, in its nature, DFA behaves differently as it doesn't propagate anything through layers but directly to the respective layers. To still make the layers integrate with the framework, the auto differentiation must be tricked in a more intriguing way. In the code, this is achieved by propagating the untouched output layer error directly to the next layer in the custom_vjp function. In doing so, every layer can apply its own matrix $B$ to that error. The remaining problem is that now the derivative of the activation should also no longer be applied on the path but layerwise. To circumvent this issue the activations are integrated into the layer. That way, they can be considered for the local computations but neglected for the global pass of the error. This is a little different compared to the separation of layer and activation as usual in Flax. Yet, once the layers are defined with this knowledge, all other features of the framework can flawlessly be used once again. In the output layer, no matrix $B$ must be applied to the error. This is done via defining a separate module for the output layer but could be achieved by setting $B$ to the identity matrix of the correct dimension as well.

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
