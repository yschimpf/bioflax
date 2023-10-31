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
