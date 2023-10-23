  
    #testing stuff
from fa import RandomDenseLinear
import jax
import jax.numpy as jnp


model = RandomDenseLinear(features = 3, lam=1)
params = model.init(jax.random.PRNGKey(0), jnp.ones((2,)))
print(jnp.ones((2,)))
print(params)

x = jnp.array([1., 2.])
y = model.apply(params, x)
#assert jnp.allclose(params["params"]["Dense_0"]["kernel"].T @ x + params["params"]["Dense_0"]["bias"], y)
target = y-1

def loss(params, x):
    return 0.5 * jnp.sum((model.apply(params, x) - target)**2)

delta_params, delta_x = jax.grad(loss, argnums=(0,1))(params, x)
print("DELTA PARAMS")
print(delta_params)
print("DELTA X")
print(delta_x)
print(params["params"]["B"] @ jnp.array([1., 1., 1.]))