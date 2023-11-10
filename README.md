# bioflax

## Introduction

Backpropagation [1], combined with stochastic gradient, is a powerful and widely used algorithm for learning via artificial neural networks. Yet, for several reasons, it's not plausible for this algorithm to run in the brain in a similar fashion as pointed out quickly after its introduction [2,3]. One of the most prominent issues is the so-called weight transport problem. A typical deep-learning architecture works in two phases. Firstly, in the forward pass inputs are fed forward through the layers of the network from the input to the output layer along the forward path to produce an output. This output is compared to the target and the resulting error is passed back through the network in the backward pass from the output to the input layer along the feedback path. On the fly, the backward pass generates error signals for each layer to compute the parameter gradients that constitute the updates.

The forward pass generates outputs $y_{l+1}$ of layer $l+1$ given inputs $y_l$ according he follwoing update rule:
$$y_{l+1} = \phi(W_{l+1}y_l+b_{l+1})$$
The backward pass feeds the error $\delta_{l+1}$ at layer $l+1$ back to layer $l$ to generate $\delta_l$ according to the backpropagation equation:
$$\delta_l = \phi'(y_l)W_{l+1}^T\delta_{l+1}$$

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

## Feedback alignment [4]
### Theory
The Feedback alignment algorithm decouples feedforward and feedback path by using fixed random weights $B$ on the feedback path. The staggering result is that even though the forward and feedback weights are independent (in the beginning) and hence the gradient computation is by no means exact, the network essentially learns how to learn using the fixed random weights $B$. This happens because information about the backward weights $B$ flows into the forward weights W via the update. Concretely, the update rules of BP are modified as follows: 

The forward pass generates outputs $y_{l+1}$ of layer $l+1$ given inputs $y_l$ according he follwoing update rule:
$$y_{l+1} = \phi(W_{l+1}y_l+b_{l+1})$$
The backward pass feeds the error $\delta_{l+1}$ at layer $l+1$ back to layer $l$ to generate $\delta_l$ using B instead of $W^T$
$$\delta_l = \phi'(y_l)B_{l}\delta_{l+1}$$

<div align="center">
  <img src="/docs/figures/FA.png" alt="FA Neural Network" width="150"/>
  <br>
  <em>Figure 1: Neural Network using FA. Taken from [5].</em>
</div>

### Code
The code is contained in the [model.py](/bioflax/model.py) file. The functionality is implemented via a custom Flax linen module RandomDenseLinearFA. More specifically, a [custom_vjp](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/_autosummary/flax.linen.custom_vjp.html) function is defined that uses a standard Dense layer on the forward path but passes the error through the layer by multiplying with $B$ instead of $W$ on the backward path. Note that the way Flax works the nonlinearity is independent of the layer and hence will be handled by auto differentiation when composing the entire backward path. Overall, this extension perfectly integrates into the Flax framework.

## Kolen Pollack [6]
### Theory
KP uses the exact same architecture as FA. While it is fascinating that with FA the network learns how to make use of the fixed feedback weights $B$, the obvious issue from an optimization point of view is that the gradients are not exact anymore. Kolen Pollack doesn't leave the feedback weights $B$ untouched as does FA but uses a simple update rule which ensures that forward matrices $W$ and feedback matrices $B$ converge. By that gradient alignment is sped up compared to FA and reaches higher levels as well.

The computations for forward and feedback paths remain the same as for FA. The feedback weights $B$ are now updated according to: 

Forward weights $W$ are updated according to:
$$\Delta W  = - \eta_W \delta_{l+1}y_l^T - \lambda W_{l+1}$$
Backward weights $B$ are updated according to: 
$$\Delta B = - \eta_W y_l \delta_{l+1}^T - \lambda B_{l+1}$$
Note that $B$ has the shape of $W^T$ hence the transposing in the update rule. $\eta_W$ is the learning rate and $\lambda$ the weight decay. One can easily verify that
$$W(t) - B(t) = (1-\lambda)^t[W(0)-B(0)]$$
where $B(t)$ and $W(t)$ refers to the respective matrix at step t. Hence, as long as $0 < \lambda < 1$ $W$ and $B$ matrices will converge, and with them the respective gradients. 

### Code
The code is contained in the [model.py](/bioflax/model.py) file. The functionality is implemented via a custom Flax linen module RandomDenseLinearKP. More specifically, a [custom_vjp](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/_autosummary/flax.linen.custom_vjp.html) function is defined that uses a standard Dense layer on the forward path but passes the error through the layer by multiplying with $B$ instead of $W$ on the backward path. Moreover, the backward pass does not return a zero gradient for $B$ anymore, as it does in RandomDenseLinearKP but sets the gradient of $B$ to that of $W$ (transposed). Using a respective weight decay in the optimizer will yield the KP updates of $B$.

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
