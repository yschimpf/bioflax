import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core.frozen_dict import freeze, unfreeze
from typing import Any

class RandomDenseLinearFA(nn.Module):
    """
    Creates a linear layer which uses feedback alignment for the backward pass.
    ...
    Attributes
    __________
    features : int
        number of output features
    """

    features : int

    @nn.compact
    def __call__(self, x):
        B = self.param("B", nn.initializers.lecun_normal(), (jnp.shape(x)[-1], self.features))

        def f(module, x, B):
            return module(x)

        def fwd(module, x, B):
            primals_out, vjp_fun = nn.vjp(f, module, x, B)
            return primals_out, (vjp_fun, B)

        def bwd(vjp_fn_B, delta):
            vjp_fn, B = vjp_fn_B
            delta_params, _, _ = vjp_fn(delta)
            delta_x = (B @ delta.transpose()).transpose()
            return (delta_params, delta_x, jnp.zeros_like(B))

        forward_module = nn.Dense(self.features)
        custom_f = nn.custom_vjp(fn=f, forward_fn=fwd, backward_fn=bwd)
        return custom_f(forward_module, x, B)



class RandomDenseLinearKP(nn.Module):
    """
    Creates a linear layer which uses feedback alignment for the backward pass and uses Kollen-Pollack 
    updates to align forward and backward matrices.
    ...
    Attributes
    __________
    features : int
        number of output features
    """

    features : int

    @nn.compact
    def __call__(self, x):
        B = self.param("B", nn.initializers.lecun_normal(), (jnp.shape(x)[-1], self.features))

        def f(module, x, B):
            return module(x)

        def fwd(module, x, B):
            primals_out, vjp_fun = nn.vjp(f, module, x, B)
            return primals_out, (vjp_fun, B)

        def bwd(vjp_fn_B, delta):
            vjp_fn, B = vjp_fn_B
            delta_params, _, _ = vjp_fn(delta)
            delta_x = (B @ delta.transpose()).transpose()
            return (delta_params, delta_x, delta_params["params"]["kernel"])

        forward_module = nn.Dense(self.features)
        custom_f = nn.custom_vjp(fn=f, forward_fn=fwd, backward_fn=bwd)
        return custom_f(forward_module, x, B)
    


class RandomDenseLinearDFAOutput(nn.Module):
    """
    Creates an output linear layer which uses direct feedback alignment for the backward pass.
    ...
    Attributes
    __________
    features : int
        number of output features
    """

    features : int

    @nn.compact
    def __call__(self, x):

        def f(module, x):
            return module(x)
        
        def fwd(module, x):
            return nn.vjp(f, module, x)

        def bwd(vjp_fn, delta):
            #print("Shape of delta in DFA output: ", delta.shape)
            delta_params, _ = vjp_fn(delta)
            return (delta_params, delta)
        
        forward_module = nn.Dense(self.features)
        custom_f = nn.custom_vjp(fn = f, forward_fn = fwd, backward_fn = bwd)
        return custom_f(forward_module, x)


class RandomDenseLinearDFAHidden(nn.Module):
    """
    Creates a hidden linear layer which uses direct feedback alignment for the backward pass.
    ...
    Attributes
    __________
    features : int
        number of output features
    final_output_dim : int
        number of output features of the output layer
    activation : Any = nn.relu
        activation function applied to the weighted inputs of the layer
    """
    features : int
    final_output_dim : int
    activation : Any = nn.relu

    @nn.compact
    def __call__(self, x):
        B = self.param("B", nn.initializers.lecun_normal(), (self.features,self.final_output_dim))

        def f(module, x, B):
            return self.activation(module(x))
        
        def fwd(module, x, B):
            primals_out, vjp_fun = nn.vjp(f, module, x, B)
            return primals_out, (vjp_fun, B)

        #problem: even if I pass the delta directly further the activation functions derviative will still be applied and passed (without applying it however I don't get that information here)
        def bwd(vjp_fn_B, delta):
            vjp_fn, B = vjp_fn_B
            delta_params, _, _ = vjp_fn((B @ delta.transpose()).transpose())
            return (delta_params, delta, jnp.zeros_like(B))
        
        forward_module = nn.Dense(self.features)
        custom_f = nn.custom_vjp(fn = f, forward_fn = fwd, backward_fn = bwd)
        return custom_f(forward_module, x, B)
    


class BioNeuralNetwork(nn.Module):
    """
    Creates a neural network with bio-inspired layers according to selected mode.
    ...
    Attributes
    __________
    hidden_layers : [int]
        list of hidden layer dimensions
    activations : [str]
    features : int 
        number of output features
        list of activation functions for hidden layers
    mode : str
        mode of the network, either "bp" for backpropagation, "fa" for feedback alignment, 
        "dfa" for direct feedback alignment, or "kp" for kollen-pollack
    """

    hidden_layers : [int]
    activations : [str]
    features : int = 4
    mode : str = "bp"

    @nn.jit
    @nn.compact
    def __call__(self, x):
        for features,activation in zip(self.hidden_layers,self.activations):
            if self.mode == "bp":
                x = nn.Dense(features)(x)
                x = getattr(nn, activation)(x)
            elif self.mode == "fa":
                x = RandomDenseLinearFA(features)(x)
                x = getattr(nn, activation)(x)
            elif self.mode == "dfa":
                x = RandomDenseLinearDFAHidden(features, self.features, getattr(nn, activation))(x)
            elif self.mode == "kp":
                x = RandomDenseLinearKP(features)(x)
                x = getattr(nn, activation)(x)
        if self.mode == "bp":
            x = nn.Dense(self.features)(x)
        elif self.mode == "fa":
            x = RandomDenseLinearFA(self.features)(x)
        elif self.mode == "dfa":
            x = RandomDenseLinearDFAOutput(self.features)(x)
        elif self.mode == "kp":
            x = RandomDenseLinearKP(self.features)(x)
        return x
    
class TeacherNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(32)(x)
        x = nn.sigmoid(x)
        x = nn.Dense(1)(x)
        return x
    
BatchTeacher= nn.vmap(
    TeacherNetwork,
    in_axes=0,
    out_axes=0,
    variable_axes={'params': None},
    split_rngs={'params': False},
    axis_name='batch',
    )

BatchBioNeuralNetwork = nn.vmap(
    BioNeuralNetwork,
    in_axes=0,
    out_axes=0,
    variable_axes={'params': None},
    split_rngs={'params': False},
    axis_name='batch',
    )