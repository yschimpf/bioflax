import jax
import jax.numpy as jnp
from flax import linen as nn

class RandomDenseLinear(nn.Module):
    features : int


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
            delta_x = B @ delta
            return (delta_params, delta_x)
        
        forward_module = nn.Dense(self.features)
        custom_f = nn.custom_vjp(fn = f, forward_fn = fwd, backward_fn = bwd)
        return custom_f(forward_module, x)

#testing stuff
model = RandomDenseLinear(features = 3)
params = model.init(jax.random.PRNGKey(0), jnp.ones((2,)))
print(jnp.ones((2,)))
print(params) 

x = jnp.array([1., 2.])
y = model.apply(params, x)
assert jnp.allclose(params["params"]["Dense_0"]["kernel"].T @ x + params["params"]["Dense_0"]["bias"], y)
target = y-1

def loss(params, x):
    return 0.5 * jnp.sum((model.apply(params, x) - target)**2)

delta_params, delta_x = jax.grad(loss, argnums=(0,1))(params, x)
print("DELTA PARAMS")
print(delta_params)
print("DELTA X")
print(delta_x)
print(params["params"]["B"] @ jnp.array([1., 1., 1.]))

