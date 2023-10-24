class LinearSystem:
    def __init__(
        self,
        seed: int = 42,
        d_input: int = 1,
        d_hidden: int = 10,
        d_output: int = 1,
        T: int = 100,
        parametrization: str = "diagonal_complex",
        input_type: str = "gaussian",
        noise_std: float = 0.0,
        batch_size: int = 32,
        **kwargs
    ):
        key = jax.random.PRNGKey(seed)
        self.batch_size = batch_size
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_output = d_output
        self.noise_std = noise_std
        self.input_type = input_type
        self.T = T

        if parametrization == "diagonal_complex":
            # Diagonal complex parametrization of the state space model
            # Enables controlling min and max norm of the eigenvalues of A
            key_nu, key_theta, key_B, key_C, key_D = jax.random.split(key, 5)
            nu = jax.random.uniform(
                key_nu, shape=(d_hidden,), minval=kwargs["min_nu"], maxval=kwargs["max_nu"]
            )
            theta = jax.random.uniform(key_theta, shape=(d_hidden,), minval=0.0, maxval=2 * jnp.pi)
            self.A = jnp.eye(d_hidden) * jnp.diag(nu * jnp.exp(1j * theta))
            self.B = jax.random.normal(
                key_B, shape=(d_hidden, d_input), dtype=jnp.complex64
            ) / jnp.sqrt(2 * d_hidden)
            self.C = jax.random.normal(
                key_C, shape=(d_output, d_hidden), dtype=jnp.complex64
            ) / jnp.sqrt(2 * d_hidden)
            self.D = jax.random.normal(
                key_D, shape=(d_output, d_input), dtype=jnp.float32
            ) / jnp.sqrt(d_output)
        elif parametrization == "diagonal_real":
            # Same as diagonal complex, but restrict the eigenvalues to be symmetric
            raise NotImplementedError
        elif parametrization == "controllable":
            # Controllable canonical form of the state space model
            # https://en.wikipedia.org/wiki/State-space_representation#Canonical_realizations
            assert d_input == 1
            assert d_output == 1
            raise NotImplementedError
        elif parametrization == "dense":
            # Vanilla parametrization of the state space model
            # All parameters are sampled from a normal distribution
            key_A, key_B, key_C, key_D = jax.random.split(key, 4)
            self.A = jax.random.normal(key_A, shape=(d_hidden, d_hidden)) / jnp.sqrt(d_hidden)
            self.A = kwargs["max_nu"] * self.A / np.max(np.abs(np.linalg.eigvals(self.A)))
            self.B = jax.random.normal(key_B, shape=(d_hidden, d_input)) / jnp.sqrt(d_input)
            self.C = jax.random.normal(key_C, shape=(d_output, d_hidden)) / jnp.sqrt(d_hidden)
            self.D = jax.random.normal(key_D, shape=(d_output, d_input)) / jnp.sqrt(d_input)
        else:
            raise ValueError

        # If we don't use B, C, D, we set them to Id, Id, 0
        if not kwargs["use_B_C_D"]:
            assert d_input == d_hidden
            assert d_hidden == d_output
            self.B = jnp.eye(d_hidden)
            self.C = jnp.eye(d_output)
            self.D = jnp.zeros_like(self.D)

    @partial(jax.jit, static_argnums=(0,))
    def sample(self, key):
        # TODO add different types of inputs
        key_inputs, key_noise = jax.random.split(key, 2)
        inputs = jax.random.normal(key_inputs, shape=(self.batch_size, self.T, self.d_input))
        if self.input_type == "gaussian":
            inputs = inputs
        elif self.input_type == "constant":
            inputs = jnp.ones_like(inputs) * inputs[0]
        noise = self.noise_std * jax.random.normal(
            key_noise, shape=(self.batch_size, self.T, self.d_output)
        )

        def _step(h, x):
            new_h = self.A @ h + self.B @ x
            return new_h, new_h

        init_hiddens = jnp.zeros((self.batch_size, self.d_hidden), dtype=self.A.dtype)
        hiddens = jax.vmap(partial(jax.lax.scan, _step))(init_hiddens, inputs)[1]
        outputs = jax.vmap(jax.vmap(lambda h, x, n: (self.C @ h).real + self.D @ x + n))(
            hiddens, inputs, noise
        )
        return inputs, outputs


class RegressionDataLoader:
    def __init__(self, dataset, seed, batch_size, size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.rng = jax.random.PRNGKey(seed)
        self.size = size

    def __iter__(self):
        # Sample a new random seed everytime we initialize an iterator
        self.rng, _ = jax.random.split(self.rng)
        yield self.dataset.sample(self.rng)

    def __len__(self):
        return self.size
    

    @partial(jax.jit, static_argnums=(5, 6, 7))
def train_step(state, rng, inputs, labels, masks, model, norm, classification=True):
    """Performs a single training step given a batch of data"""

    def _loss(params):
        if norm in ["batch"]: #contains batch norm (can be renmoved for now)
            p = {"params": params, "batch_stats": state.batch_stats}
            logits, vars = model.apply(p, inputs, rngs={"dropout": rng}, mutable=["batch_stats"])
        else:
            p = {"params": params}
            vars = None
            logits = model.apply(p, inputs, rngs={"dropout": rng})
        return loss_fn(logits, labels, masks, classification=classification), vars

    (loss, vars), grads = jax.value_and_grad(_loss, has_aux=True)(state.params)

    if norm in ["batch"]:
        state = state.apply_gradients(grads=grads, batch_stats=vars["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)

    return state, loss


def train_epoch(state, rng, model, trainloader, seq_len, in_dim, norm, lr_params):
    “”"
    Training function for an epoch that loops over batches.
    “”"
    model = model(training=True)  # model in training mode
    batch_losses = []
    decay_function, ssm_lr, lr, step, end_step, lr_min = lr_params
    print(“length train loader”, len(trainloader))
    for batch in trainloader:
        inputs, labels, masks = prep_batch(batch, seq_len, in_dim) #not needed here only if we need to do masking
        rng, drop_rng = jax.random.split(rng)
        state, loss = train_step(
            state, drop_rng, inputs, labels, masks, model, norm, model.classification
        )
        batch_losses.append(loss)  # log loss value
        lr_params = (decay_function, ssm_lr, lr, step, end_step, lr_min) #simpler way to use scheduler => see optax (beginning SGD with quite small learning rates and large batch sizes)
        state, step = update_learning_rate_per_step(lr_params, state)
    # Return average loss over batches
    return state, jnp.mean(jnp.array(batch_losses)), step