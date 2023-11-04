import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core.frozen_dict import freeze, unfreeze
import optax

class RandomDenseLinear(nn.Module):
    features : int
    #def __init__(self, features : int, rng_key = None):
    #    super().__init__()
    #    self.features = features
    #    self.rng_key = rng_key

    @nn.compact
    def __call__(self, x):

        def f(module, x):
            return module(x)
        
        def fwd(module, x):
            return nn.vjp(f, module, x)
        
        B = self.param("B", nn.initializers.lecun_normal(), (jnp.shape(x)[-1],self.features))

        def bwd(vjp_fn, delta):
            delta_params, _ = vjp_fn(delta)
            delta_x = (B @ delta.transpose()).transpose()
            return (delta_params, delta_x)
        
        forward_module = nn.Dense(self.features)
        custom_f = nn.custom_vjp(fn = f, forward_fn = fwd, backward_fn = bwd)
        return custom_f(forward_module, x)
    
class RandomDenseLinearFA(nn.Module):
    """
    Creates a linear layer which uses feedback alignment for the backward pass.
    ...
    Attributes
    __________
    features : int
        number of output features
    """

    features: int

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
    features : int

    
    @nn.compact
    def __call__(self, x):
        
        #forward_module = nn.Dense(self.features)
        forward_module = RandomDenseLinear(self.features)

        def f(module, x):
            return module(x)
        
        def fwd(module, x):
            return nn.vjp(f, module, x)
        

        def bwd(vjp_fn, delta):
            delta_params, _ = vjp_fn(delta)
            #print("1")
            #print(forward_module.variables)
            delta = forward_module.variables["params"]["B"] @ delta
            #print("2")
            #print(delta)
            #print(delta_params)
            delta_params = unfreeze(delta_params)
            delta_params["params"]["B"] = delta_params["params"]["Dense_0"]["kernel"]
            delta_params["params"] = freeze(delta_params["params"])
            #print(delta_params)
            return (delta_params, delta)
        
        
        custom_f = nn.custom_vjp(fn = f, forward_fn = fwd, backward_fn = bwd)
        return custom_f(forward_module, x)
    
        """
        old version:
        ____________
        @nn.compact
        def __call__(self, x):

            def f(module, x):
                return module(x)
            
            def fwd(module, x):
                return nn.vjp(f, module, x)
            
            B = self.param("B", nn.initializers.lecun_normal(), (jnp.shape(x)[-1],self.features))

            def bwd(vjp_fn, delta):
                delta_params, _ = vjp_fn(delta)
                #print("Shape of B in FA: ", B.shape)
                print("Shape of delta in FA: ", delta.shape)
                print("Self.variables in FA ", self.variables)
                delta_x = B @ delta.transpose()
                return (delta_params, delta_x)
            
            forward_module = nn.Dense(self.features)
            custom_f = nn.custom_vjp(fn = f, forward_fn = fwd, backward_fn = bwd)
            return custom_f(forward_module, x)
        """
    
class RandomDenseLinearKPmodified(nn.Module):
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

"""
#testing stuff

model = RandomDenseLinearKP(features = 3)
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
#print(params["params"]["B"] @ jnp.array([1., 1., 1.]))
"""

KollenP = True
if(KollenP):
    class Network(nn.Module):

            @nn.compact
            def __call__(self, x):
                x = RandomDenseLinearKPmodified(3)(x)
                x = RandomDenseLinearKPmodified(2)(x)
                return x

    model = Network()
    params = model.init(jax.random.PRNGKey(0), jnp.ones((2,)))

    print("1")
    print(params)

    # Define the optimizer
    optimizer = optax.sgd(learning_rate=0.1)

    # Initialize the optimizer state
    optimizer_state = optimizer.init(params)
    x = jnp.array([1., 2.])
    y = model.apply(params, x)
    print("a_1")
    #if using original: a_1 = params["params"]["RandomDenseLinearKP_0"]["RandomDenseLinear_0"]["Dense_0"]["kernel"].T @ x + params["params"]["RandomDenseLinearKP_0"]["RandomDenseLinear_0"]["Dense_0"]["bias"]
    a_1 = params["params"]["RandomDenseLinearKPmodified_0"]["Dense_0"]["kernel"].T @ x + params["params"]["RandomDenseLinearKPmodified_0"]["Dense_0"]["bias"]
    print(a_1)

    print("a_2")
    #if using original: a_2 = params["params"]["RandomDenseLinearKP_1"]["RandomDenseLinear_0"]["Dense_0"]["kernel"].T @ a_1 + params["params"]["RandomDenseLinearKP_1"]["RandomDenseLinear_0"]["Dense_0"]["bias"]
    a_2 = params["params"]["RandomDenseLinearKPmodified_1"]["Dense_0"]["kernel"].T @ a_1 + params["params"]["RandomDenseLinearKPmodified_1"]["Dense_0"]["bias"]
    print(a_2)
    
    assert jnp.allclose(a_2, y)


    target = y-1
    #@jax.jit
    def loss(params, x):
        return 0.5 * jnp.sum((model.apply(params, x)-target)**2)
    delta_params, delta_x = jax.grad(loss, argnums=(0,1))(params, x)
    print("DELTA PARAMS")
    print(delta_params)
    print("DELTA X")
    print(delta_x)
    #if using original: assert jnp.allclose(delta_x, params["params"]["RandomDenseLinearKP_0"]["RandomDenseLinear_0"]["B"] @ params["params"]["RandomDenseLinearKP_1"]["RandomDenseLinear_0"]["B"] @ jnp.ones((2,)))
    assert jnp.allclose(delta_x, params["params"]["RandomDenseLinearKPmodified_0"]["B"] @ params["params"]["RandomDenseLinearKPmodified_1"]["B"] @ jnp.ones((2,)))
    print("dW_output")
    print(jnp.outer(a_1,jnp.ones((2,))))

    print("dW_1")
    #if using original: print(jnp.outer(x,(params["params"]["RandomDenseLinearKP_1"]["RandomDenseLinear_0"]["B"] @ jnp.ones((2,)))))
    print(jnp.outer(x,(params["params"]["RandomDenseLinearKPmodified_1"]["B"] @ jnp.ones((2,)))))



    # Compute the gradients of the loss function with respect to the parameters and the input
    delta_params, delta_x = jax.grad(loss, argnums=(0,1))(params, x)

    # Apply the gradient update to the parameters using the optimizer
    updates, new_optimizer_state = optimizer.update(delta_params, optimizer_state)
    new_params = optax.apply_updates(params, updates)

    # Update the parameters of the model
    print(params)
    print(new_params)

    #if using original: assert jnp.allclose(new_params["params"]["RandomDenseLinearKP_1"]["RandomDenseLinear_0"]["B"], params["params"]["RandomDenseLinearKP_1"]["RandomDenseLinear_0"]["B"]-0.1*delta_params["params"]["RandomDenseLinearKP_1"]["RandomDenseLinear_0"]["B"])
    #if using original: assert jnp.allclose(new_params["params"]["RandomDenseLinearKP_0"]["RandomDenseLinear_0"]["B"], params["params"]["RandomDenseLinearKP_0"]["RandomDenseLinear_0"]["B"]-0.1*delta_params["params"]["RandomDenseLinearKP_0"]["RandomDenseLinear_0"]["B"])
    assert jnp.allclose(new_params["params"]["RandomDenseLinearKPmodified_1"]["B"], params["params"]["RandomDenseLinearKPmodified_1"]["B"]-0.1*delta_params["params"]["RandomDenseLinearKPmodified_1"]["B"])
    assert jnp.allclose(new_params["params"]["RandomDenseLinearKPmodified_0"]["B"], params["params"]["RandomDenseLinearKPmodified_0"]["B"]-0.1*delta_params["params"]["RandomDenseLinearKPmodified_0"]["B"])
    # Update the optimizer state
    optimizer_state = new_optimizer_state
else:
# Initialize module
    model = RandomDenseLinear(features=3)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 2)))
    print("Params: ", params)

    # Call model
    x = jnp.ones((1, 2))
    y = model.apply(params, x)
    print("Output: ", y)

    target = y-1


    def loss(params, x):
        return 0.5 * jnp.sum((model.apply(params, x)-target)**2)

    # Compute gradients
    grads, delta = jax.grad(loss, argnums=(0,1))(params, x)
    print("Gradients: ", grads)
    print("Delta: ", delta)


    # check whether other solution computes the same
    model_ = RandomDenseLinearFA(features = 3)
    params_ = model_.init(jax.random.PRNGKey(0), jnp.ones((1, 2)))
    print("Params: ", params_)

    x_ = jnp.ones((1, 2))
    y_ = model_.apply(params_, x_)
    print("Output: ", y_)

    target_ = y_-1

    @jax.jit
    def loss_(params, x):
        return 0.5 * jnp.sum((model_.apply(params, x)-target_)**2)

    grads_, delta_= jax.grad(loss_, argnums=(0,1))(params_, x_)

    print("Gradients: ", grads_,)
    print("Delta: ", delta_,)

