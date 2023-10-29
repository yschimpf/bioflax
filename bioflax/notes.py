@jax.jit
def train_step(state, rng, inputs, labels, model, classification=False):

    def loss_fn(params):
        p = {"params": params}
        vars = None
        logits = model.apply(p, inputs)
        return logits

    (loss, vars), grads = jax.value_and_grad(_loss, has_aux=True)(state.params)

    if norm in ["batch"]:
        state = state.apply_gradients(grads=grads, batch_stats=vars["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)

    return state, loss


def train_epoch(state, rng, model, trainloader, seq_len, in_dim, norm, lr_params):
    model = model(training=True)  # model in training mode
    batch_losses = []
    decay_function, ssm_lr, lr, step, end_step, lr_min = lr_params
    print(“length train loader”, len(trainloader))
    for batch in trainloader:
        inputs, labels, masks = prep_batch(batch, seq_len, in_dim)
        rng, drop_rng = jax.random.split(rng)
        state, loss = train_step(
            state, drop_rng, inputs, labels, masks, model, norm, model.classification
        )
        batch_losses.append(loss)  # log loss value
        lr_params = (decay_function, ssm_lr, lr, step, end_step, lr_min)
        state, step = update_learning_rate_per_step(lr_params, state)
    # Return average loss over batches
    return state, jnp.mean(jnp.array(batch_losses)), step