import jax
import jax.numpy as jnp
import jax.tree_util as jax_tree

def remove_keys(pytree, key_list):
    """Removes leaves from the pytree whose path contains any key from key_list."""
    
    def filter_fn(path, value):
        if not any(p.key in key_list for p in path):
            return value
    filtered_pytree = jax_tree.tree_map_with_path(filter_fn, pytree)
    return filtered_pytree

def flatten_array(arr):
    # Use the reshape method to flatten the array
    flattened = arr.reshape(-1)
    return flattened

def flatten_matrices_in_tree(pytree):
    return jax_tree.tree_map(flatten_array, pytree)

@jax.jit
def compute_weight_alignment(paramsFA):
    kernels, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(paramsFA, ['bias', 'B'])))
    Bs, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(paramsFA, ['bias', 'kernel'])))
    dot_prods = jax_tree.tree_map(jnp.dot, kernels, Bs)
    norm_kern = jax_tree.tree_map(jnp.linalg.norm, kernels)
    norm_B = jax_tree.tree_map(jnp.linalg.norm, Bs)
    layerwise_alignments = jax.tree_map((lambda x, y, z: x/(y*z)), dot_prods, norm_kern, norm_B)
    return layerwise_alignments

@jax.jit
def compute_bias_grad_al_layerwise(grads_, grads):
    bias_, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(grads_, ['kernel', 'B'])))
    bias, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(grads, ['kernel', 'B'])))
    dot_prods = jax_tree.tree_map(jnp.dot, bias_, bias)
    norm_grad_ = jax_tree.tree_map(jnp.linalg.norm, bias_)
    norm_grad = jax_tree.tree_map(jnp.linalg.norm, bias)
    layerwise_alignments = jax.tree_map((lambda x, y, z: x/(y*z)), dot_prods, norm_grad_, norm_grad)
    return layerwise_alignments

@jax.jit
def compute_wandb_grad_al_layerwise(grads_, grads):
    bias_, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(grads_, ['kernel', 'B'])))
    bias, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(grads, ['kernel', 'B'])))
    kernel_, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(grads_, ['bias', 'B'])))
    kernel, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(grads, ['bias', 'B'])))
    layerwise_alignments = jax.tree_map((lambda a, b, c, d : (jnp.dot(a, b) + jnp.dot(c, d))/
                                         (jnp.sqrt(jnp.sum(jnp.multiply(a, a)) + jnp.sum(jnp.multiply(c, c)))
                                          * jnp.sqrt(jnp.sum(jnp.multiply(b, b)) + jnp.sum(jnp.multiply(d, d)))) ), bias_, bias, kernel_, kernel)
    return layerwise_alignments

@jax.jit
def compute_wandb_al_total(grads_, grads):
    bias_, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(grads_, ['kernel', 'B'])))
    bias, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(grads, ['kernel', 'B'])))
    kernel_, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(grads_, ['bias', 'B'])))
    kernel, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(grads, ['bias', 'B'])))
    layerwise_alignments = jax.tree_map((lambda a, b, c, d : (jnp.dot(a, b) + jnp.dot(c, d))), bias_, bias, kernel_, kernel)
    squared_norms_ = jax.tree_map((lambda a,b : jnp.sum(jnp.multiply(a, a)) + jnp.sum(jnp.multiply(b, b))), bias_, kernel_)
    squared_norms = jax.tree_map((lambda a,b : jnp.sum(jnp.multiply(a, a)) + jnp.sum(jnp.multiply(b, b))), bias, kernel)
    return jnp.sum(jnp.array(layerwise_alignments))/(jnp.sqrt(jnp.sum(jnp.array(squared_norms_))) * jnp.sqrt(jnp.sum(jnp.array(squared_norms))))

@jax.jit
def compute_rel_norm(grads_, grads):
    bias_, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(grads_, ['kernel', 'B'])))
    bias, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(grads, ['kernel', 'B'])))
    kernel_, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(grads_, ['bias', 'B'])))
    kernel, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(remove_keys(grads, ['bias', 'B'])))
    squared_norms_ = jax.tree_map((lambda a,b : jnp.sum(jnp.multiply(a, a)) + jnp.sum(jnp.multiply(b, b))), bias_, kernel_)
    squared_norms = jax.tree_map((lambda a,b : jnp.sum(jnp.multiply(a, a)) + jnp.sum(jnp.multiply(b, b))), bias, kernel)
    return jnp.sqrt(jnp.sum(jnp.array(squared_norms_)))/jnp.sqrt(jnp.sum(jnp.array(squared_norms)))

def entrywise_average(array_of_arrays):
    average_array = np.zeros_like(array_of_arrays[0])
    for arr in array_of_arrays:
        average_array += arr
    
    average_array = average_array/len(array_of_arrays)
    
    return average_array

def reorganize_dict(input_dict):
    new_dict = {'params': {}}

    for layer_key, layer_val in input_dict['params'].items():
        # Extract layer number
        layer_num = layer_key.split('_')[-1]
        
        # Iterate over items in each layer
        for param_key, param_val in layer_val.items():
            # Skip 'B' entries
            if param_key == 'B':
                continue
            
            # Construct new key as 'Dense_x'
            new_key = param_key.split('_')[0] + '_' + layer_num

            # Add or update the new dictionary
            if new_key not in new_dict['params']:
                new_dict['params'][new_key] = {}
            new_dict['params'][new_key] = param_val

    return new_dict