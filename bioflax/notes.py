import jax
import jax.numpy as jnp


#old code for metric computation
@jax.jit
def compute_grad_al_layerwise(grads_, grads):
    flattened_grads_ = flatten_arrays_layerwise(extract_gradients(grads_, ['bias', 'kernel'], True))
    flattened_grads = flatten_arrays_layerwise(extract_gradients(grads, ['bias', 'kernel'], True))
    return compute_layerwise_alignments(flattened_grads_, flattened_grads)

@jax.jit
def compute_bias_grad_layerwise(grads_, grads):
    flattened_grads_ = flatten_arrays_layerwise(extract_gradients(grads_, ['bias'], False))
    flattened_grads = flatten_arrays_layerwise(extract_gradients(grads, ['bias'], False))
    return compute_layerwise_alignments(flattened_grads_, flattened_grads)

@jax.jit
def compute_wandb_alignment(grads_, grads):
    extract_grads = extract_gradients(grads, ['bias', 'kernel'], False)
    extract_grads_ = extract_gradients(grads_, ['bias', 'kernel'], False)
    return compute_alignment(flatten_arrays(extract_grads_),
                              flatten_arrays(extract_grads))

@jax.jit
def compute_weight_alignment_layerwise(state):
    kernels, bs = extract_kernel_and_B_arrays(state.params, False)
    return compute_layerwise_alignments(flatten_arrays_layerwise(kernels), flatten_arrays_layerwise(bs))

def merge_consecutive_pairs(array_list):
    # Ensure there is an even number of elements to merge in pairs
    if len(array_list) % 2 != 0:
        raise ValueError("The number of arrays should be even to merge in pairs.")

    # Merge each consecutive pair
    merged_list = []
    for i in range(0, len(array_list), 2):
        a = array_list[i]
        b = array_list[i+1].T
        
        # Ensure both arrays have the same number of dimensions
        if a.ndim < b.ndim:
            a = a[..., jnp.newaxis]
        elif a.ndim > b.ndim:
            b = b[..., jnp.newaxis]
        
        merged_list.append(jnp.concatenate((a, b), axis=1))
    
    return merged_list

def extract_arrays(d, key, collected_arrays):
    if isinstance(d, dict):
        for k, v in d.items():
            if k in key:
                collected_arrays.append(v)
            else:
                extract_arrays(v, key, collected_arrays)
    return collected_arrays

def extract_gradients(param_dict, keys, merge):
    kernels = extract_arrays(param_dict, keys, [])
    kernels = merge_consecutive_pairs(kernels) if merge else kernels
    return kernels

def extract_kernel_and_B_arrays(param_dict, merge):
    kernels = extract_arrays(param_dict, ['kernel'], [])
    Bs = extract_arrays(param_dict, ['B'], [])
    #if kernels:
    #    kernels = kernels[1:]
    #if Bs:
    #    Bs = Bs[1:]
    kernels = merge_consecutive_pairs(kernels) if merge else kernels
    Bs = merge_consecutive_pairs(Bs) if merge else Bs
    return kernels, Bs

def flatten_array(arr):
    # Use the reshape method to flatten the array
    flattened = arr.reshape(-1)
    return flattened

def concatenate_arrays(array_list):
    # Use a list comprehension to concatenate the arrays
    concatenated_list = [item for array in array_list for item in array]
    return concatenated_list

def flatten_arrays_layerwise(arrays):
    list = [flatten_array(x) for x in arrays]
    return list

def flatten_arrays(arrays):
    list = [flatten_array(x) for x in arrays]
    return concatenate_arrays(list)

def compute_layerwise_alignments(as_, bs_):
    alignments = [compute_alignment(a, b) for a, b in zip(as_, bs_)]
    return alignments

def compute_alignment(a,b):
    a = jnp.array(a)
    b = jnp.array(b)
    return jnp.dot(a,b)/(jnp.linalg.norm(a)*jnp.linalg.norm(b))

"""
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
"""

