from typing import Any
import jax
import jax.numpy as jnp
from flax import linen as nn

#idea is to have activation function as part of the layer
class RandomDenseLinearDFA(nn.Module):
    features : int
    activation : Any = nn.relu

    @nn.compact
    def __call__(self, x):
        
        def f(module, x):
            return self.activation(module(x))
        
        def fwd(module, x):
            return nn.vjp(f, module, x)
        
        B = self.param("B", nn.initializers.lecun_normal(), (jnp.shape(x)[-1],self.features))

        #problem: even if I pass the delta directly further the activation functions derviative will still be applied and passed (without applying it however I don't get that information here)
        def bwd(vjp_fn, delta):
            delta_params, _ = vjp_fn(delta)
            delta_x = delta
            return (delta_params, delta_x)
        
        forward_module = nn.Dense(self.features)
        custom_f = nn.custom_vjp(fn = f, forward_fn = fwd, backward_fn = bwd)
        return custom_f(forward_module, x)