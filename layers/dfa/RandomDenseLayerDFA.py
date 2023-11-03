from typing import Any
import optax
import jax
import jax.numpy as jnp
from flax import linen as nn

#idea is to have activation function as part of the layer which is a nice workaround for using automated differentiation but makes the interface less generic
class RandomDenseLinearDFAOutput(nn.Module):
    features : int
    self_activation : Any = nn.relu

    @nn.compact
    def __call__(self, x):

        def f(module, x):
            return self.self_activation(module(x))
        
        def fwd(module, x):
            return nn.vjp(f, module, x)

        def bwd(vjp_fn, delta):
            delta_params, _ = vjp_fn(delta)
            return (delta_params, delta)
        
        forward_module = nn.Dense(self.features)
        custom_f = nn.custom_vjp(fn = f, forward_fn = fwd, backward_fn = bwd)
        return custom_f(forward_module, x)
    
    """
    old version:
    ____________
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
    """

class RandomDenseLinearDFAHiddenModified(nn.Module):
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
            delta_params, _, _ = vjp_fn(B @ delta)
            return (delta_params, delta, jnp.zeros_like(B))
        
        forward_module = nn.Dense(self.features)
        custom_f = nn.custom_vjp(fn = f, forward_fn = fwd, backward_fn = bwd)
        return custom_f(forward_module, x, B)
    
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
        """
        old version:
        ____________
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
        """
    
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


z = True

if(z): #do everything for sigmoid
    class Network(nn.Module):

        @nn.compact
        def __call__(self, x):
            x = RandomDenseLinearDFAHiddenModified(3,2,2, nn.sigmoid)(x)
            x = RandomDenseLinearDFAOutput(2, nn.sigmoid)(x)
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
    a_1 = params["params"]["RandomDenseLinearDFAHiddenModified_0"]["Dense_0"]["kernel"].T @ x + params["params"]["RandomDenseLinearDFAHiddenModified_0"]["Dense_0"]["bias"]
    print(a_1)

    print("h_1")
    h_1 = nn.sigmoid(a_1)
    print(h_1)

    print("a_2")
    a_2 = params["params"]["RandomDenseLinearDFAOutput_0"]["Dense_0"]["kernel"].T @ h_1 + params["params"]["RandomDenseLinearDFAOutput_0"]["Dense_0"]["bias"]
    print(a_2)

    print("h_2")
    h_2 = nn.sigmoid(a_2)
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

    #define a function that computer^s derivative of relu
    def relu_derivative(x):
        return jnp.where(x > 0, 1, 0)

    #define a function that computes derivative of sigmoid
    def sigmoid_derivative(x):
        return jnp.exp(-x) / (1 + jnp.exp(-x))**2

    print("dW_2")
    print(jnp.outer(h_1,jnp.ones((2,))*sigmoid_derivative(a_2)))

    print("dW_1")
    print(jnp.outer(x,(params["params"]["RandomDenseLinearDFAHiddenModified_0"]["B"] @ jnp.ones((2,)) * sigmoid_derivative(a_1))))




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

else: #do everything for relu
    class Network(nn.Module):

        @nn.compact
        def __call__(self, x):
            x = RandomDenseLinearDFAHiddenModified(3,2,2)(x)
            x = RandomDenseLinearDFAOutput(2)(x)
            return x

    model = Network()
    params = model.init(jax.random.PRNGKey(0), jnp.ones((2,)))
    print("Params of model: ", params)

    # Define the optimizer
    optimizer = optax.sgd(learning_rate=0.1)

    # Initialize the optimizer state
    optimizer_state = optimizer.init(params)
    x = jnp.array([1., 2.])
    y = model.apply(params, x)
    print("a_1")
    a_1 = params["params"]["RandomDenseLinearDFAHiddenModified_0"]["Dense_0"]["kernel"].T @ x + params["params"]["RandomDenseLinearDFAHiddenModified_0"]["Dense_0"]["bias"]
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

    #define a function that computer^s derivative of relu
    def relu_derivative(x):
        return jnp.where(x > 0, 1, 0)

    print("dW_2")
    print(jnp.outer(h_1,jnp.ones((2,))*relu_derivative(a_2)))

    print("dW_1")
    print(jnp.outer(x,(params["params"]["RandomDenseLinearDFAHiddenModified_0"]["B"] @ jnp.ones((2,)) * relu_derivative(a_1))))




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