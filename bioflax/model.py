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

        def f(module, x):
            return module(x)
        
        def fwd(module, x):
            return nn.vjp(f, module, x)
        
        B = self.param("B", nn.initializers.lecun_normal(), (jnp.shape(x)[-1],self.features))

        def bwd(vjp_fn, delta):
            delta_params, _ = vjp_fn(delta)
            delta_x = B @ delta
            return (delta_params, delta_x)
        
        forward_module = nn.Dense(self.features)
        custom_f = nn.custom_vjp(fn = f, forward_fn = fwd, backward_fn = bwd)
        return custom_f(forward_module, x)



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
        
        forward_module = RandomDenseLinearFA(self.features)

        def f(module, x):
            return module(x)
        
        def fwd(module, x):
            return nn.vjp(f, module, x)

        def bwd(vjp_fn, delta):
            delta_params, _ = vjp_fn(delta)
            delta = forward_module.variables["params"]["B"] @ delta
            delta_params = unfreeze(delta_params)
            delta_params["params"]["B"] = delta_params["params"]["Dense_0"]["kernel"]
            delta_params["params"] = freeze(delta_params["params"])
            return (delta_params, delta)
        
        custom_f = nn.custom_vjp(fn = f, forward_fn = fwd, backward_fn = bwd)
        return custom_f(forward_module, x)


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
        
        def f(module, x):
            return self.activation(module(x))
        
        def fwd(module, x):
            return nn.vjp(f, module, x)
        
        B = self.param("B", nn.initializers.lecun_normal(), (self.features,self.final_output_dim))

        def bwd(vjp_fn, delta):
            delta_params, _ = vjp_fn(B @ delta)
            return (delta_params, delta)
        
        forward_module = nn.Dense(self.features)
        custom_f = nn.custom_vjp(fn = f, forward_fn = fwd, backward_fn = bwd)
        return custom_f(forward_module, x)


class BioNeuralNetwork(nn.Module):
    (idee loop over layers)