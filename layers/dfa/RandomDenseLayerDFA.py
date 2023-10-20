from typing import Any
import optax
import jax
import jax.numpy as jnp
from flax import linen as nn

#idea is to have activation function as part of the layer which is a nice workaround for using automated differentiation but makes the interface less generic

class RandomDenseLinearDFAOutput(nn.Module):
    features : int
    activation : Any = nn.relu

    @nn.compact
    def __call__(self, x):

        def f(module, x):
            return self.activation(module(x))
        
        def fwd(module, x):
            return nn.vjp(f, module, x)
        
        B = self.param("B", nn.initializers.lecun_normal(), (jnp.shape(x)[-1],self.features))

        def bwd(vjp_fn, delta):
            delta_params, _ = vjp_fn(delta)
            delta_x = delta
            return (delta_params, delta_x)
        
        forward_module = nn.Dense(self.features)
        custom_f = nn.custom_vjp(fn = f, forward_fn = fwd, backward_fn = bwd)
        return custom_f(forward_module, x)

class RandomDenseLinearDFAHidden(nn.Module):
    features : int
    final_output_dim : int
    next_layer_dim : int
    activation : Any = nn.relu

    @nn.compact
    def __call__(self, x):
        
        def f(module, x):
            return self.activation(module(x))
        
        def fwd(module, x):
            return nn.vjp(f, module, x)
        
        B = self.param("B", nn.initializers.lecun_normal(), (self.features,self.final_output_dim))

        #problem: even if I pass the delta directly further the activation functions derviative will still be applied and passed (without applying it however I don't get that information here)
        def bwd(vjp_fn, delta):
            delta_params, _ = vjp_fn(B @ delta)
            delta_x = delta
            return (delta_params, delta_x)
        
        forward_module = nn.Dense(self.features)
        custom_f = nn.custom_vjp(fn = f, forward_fn = fwd, backward_fn = bwd)
        return custom_f(forward_module, x)
    
"""  
model = RandomDenseLinearDFAOutput(3)
params = model.init(jax.random.PRNGKey(0), jnp.ones((2,)))
x = jnp.array([1., 2.])
y = model.apply(params, x)
print(params)
print("W.T “x + b")
print(params["params"]["Dense_0"]["kernel"].T @ x)
assert jnp.allclose(nn.relu(params["params"]["Dense_0"]["kernel"].T @ x + params["params"]["Dense_0"]["bias"]), y)
target = y-1
def loss(params, x):
    return 0.5 * jnp.sum((model.apply(params, x)-target)**2)
delta_params, delta_x = jax.grad(loss, argnums=(0,1))(params, x)
print("DELTA PARAMS")
print(delta_params)
print("DELTA X")
print(delta_x)
print(params["params"]["B"]@jnp.array([1,1,1]))


"""
class Network(nn.Module):

    @nn.compact
    def __call__(self, x):
        x = RandomDenseLinearDFAHidden(3,2,2)(x)
        x = RandomDenseLinearDFAOutput(2)(x)
        return x

model = Network()
params = model.init(jax.random.PRNGKey(0), jnp.ones((2,)))

# Define the optimizer
optimizer = optax.sgd(learning_rate=0.1)

# Initialize the optimizer state
optimizer_state = optimizer.init(params)


x = jnp.array([1., 2.])
y = model.apply(params, x)
print("a_1")
a_1 = params["params"]["RandomDenseLinearDFAHidden_0"]["Dense_0"]["kernel"].T @ x + params["params"]["RandomDenseLinearDFAHidden_0"]["Dense_0"]["bias"]
print(a_1)

print("h_1")
h_1 = nn.relu(a_1)
print(h_1)

print("a_2")
a_2 = params["params"]["RandomDenseLinearDFAOutput_0"]["Dense_0"]["kernel"].T @ h_1 + params["params"]["RandomDenseLinearDFAOutput_0"]["Dense_0"]["bias"]
print(a_2)

print("h_2")
h_2 = nn.relu(a_2)
print(h_2)

assert jnp.allclose(h_2, y)

target = y-1
def loss(params, x):
    return 0.5 * jnp.sum((model.apply(params, x)-target)**2)
delta_params, delta_x = jax.grad(loss, argnums=(0,1))(params, x)
print("DELTA PARAMS")
print(delta_params)
print("DELTA X")
print(delta_x)

print("dW_2")
print(jnp.outer(h_1, jnp.ones((2,))))

#define a function that computer^s derivative of relu
def relu_derivative(x):
    return jnp.where(x > 0, 1, 0)

print("dW_1")
print(jnp.outer((params["params"]["RandomDenseLinearDFAHidden_0"]["B"] @ jnp.ones((2,)) * relu_derivative(a_1)) , x))




# Compute the gradients of the loss function with respect to the parameters and the input
delta_params, delta_x = jax.grad(loss, argnums=(0,1))(params, x)

# Apply the gradient update to the parameters using the optimizer
updates, new_optimizer_state = optimizer.update(delta_params, optimizer_state)
new_params = optax.apply_updates(params, updates)

# Update the parameters of the model
print(params)
print(new_params)

# Update the optimizer state
optimizer_state = new_optimizer_state