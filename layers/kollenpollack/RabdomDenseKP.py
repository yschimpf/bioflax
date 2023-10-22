import jax
import jax.numpy as jnp
from flax import linen as nn

class RandomDenseLinear(nn.Module):
    features : int
    lamda : jnp.float32 = 1. #needs to be scaled by stepsize i.e. stepsize = 0.1 requires lambda = 1 to get an effective lambda of 0.1


    #def __init__(self, features : int, rng_key = None):
    #    super().__init__()
    #    self.features = features
    #    self.rng_key = rng_key

    #def setup(self):
    #    self.W = self.param('W', nn.initializers.lecun_normal(), (self.features,))
    #    self.B = self.param('B', nn.initializers.lecun_normal(), (self.features,))
    #    self.b = self.param('b', nn.initializers.zeros, (self.features,))

    
    @nn.compact
    def __call__(self, x):

        def f(module, x):
            return module(x)
        
        def fwd(module, x):
            return nn.vjp(f, module, x)
        
        B = self.param("B", nn.initializers.lecun_normal(), (jnp.shape(x)[-1],self.features))

        def bwd(vjp_fn, delta):
            delta_params, _ = vjp_fn(delta)
            delta_params["params"]["B"] = delta_params["params"]["Dense_0"]["kernel"] + self.lamda * self.params["params"]["B"]
            delta_params["params"]["Dense_0"]["kernel"] = delta_params["params"]["Dense_0"]["kernel"] + self.lamda * self.params["params"]["Dense_0"]["kernel"]
            delta_params["params"]["Dense_0"]["bias"] = delta_params["params"]["Dense_0"]["bias"] + self.lamda * self.params["params"]["Dense_0"]["bias"]
            delta_x = B @ delta
            return (delta_params, delta_x)
        
        forward_module = nn.Dense(self.features)
        custom_f = nn.custom_vjp(fn = f, forward_fn = fwd, backward_fn = bwd)
        return custom_f(forward_module, x)