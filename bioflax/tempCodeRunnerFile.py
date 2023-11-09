remove_keys(grads, ['kernel', 'B'])))
    kernel_, _ = jax_tree.tree_flatten(flatten_matrices_in_