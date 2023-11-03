import unittest
import jax.numpy as jnp
import jax
import flax.linen as nn
from model import (
    RandomDenseLinearFA,
    RandomDenseLinearKP,
    RandomDenseLinearDFAOutput,
    RandomDenseLinearDFAHidden
)

class Test_RandomDenseLinearFA(unittest.TestCase):

    def test_init(self):
        """
        Check whether the number of output features is correct
        """
        test = RandomDenseLinearFA(4)
        self.assertEqual(test.features, 4)

    def test_single_layer_output(self):
        """
        Check whether for a single layer network the output is correct
        y = Wx + b
        """
        model = RandomDenseLinearFA(10)
        params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 5)))
        x = jnp.ones((1, 5))
        y = model.apply(params, x)
        self.assertEqual(y.shape, (1, 10))
        # Due to dimensions: y = b + xW
        self.assertTrue((y == params["params"]["Dense_0"]["bias"]+ x @ params["params"]["Dense_0"]["kernel"]).all())

    def test_multi_layer_output(self):
        """
        Check whether for a two layer network the output is correct
        y = W2(W1x + b1) + b2
        """
        class MultiLayerModel(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = RandomDenseLinearFA(15)(x)
                x = RandomDenseLinearFA(10)(x)
                return x
        
        model = MultiLayerModel()
        params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 5)))
        x = jnp.ones((1, 5))
        y = model.apply(params, x)
        self.assertEqual(y.shape, (1, 10))
        # Due to dimensions: y = b2 + (b1 + xW1)W2
        self.assertTrue((y == params["params"]["RandomDenseLinearFA_1"]["Dense_0"]["bias"] + 
                         (params["params"]["RandomDenseLinearFA_0"]["Dense_0"]["bias"]+ 
                          x @ params["params"]["RandomDenseLinearFA_0"]["Dense_0"]["kernel"])
                          @ params["params"]["RandomDenseLinearFA_1"]["Dense_0"]["kernel"]).all())
        
    def test_multi_layer_activation(self):
        """
        Check whether for a two layer network the output is correct
        y = W2 * f(W1x + b1) + b2, where f is the sigmoid activation function
        """
        class MultiLayerModelAct(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = RandomDenseLinearFA(15)(x)
                x = nn.sigmoid(x)
                x = RandomDenseLinearFA(10)(x)
                return x
        
        model = MultiLayerModelAct()
        params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 5)))
        x = jnp.ones((1, 5))
        y = model.apply(params, x)
        self.assertEqual(y.shape, (1, 10))
        # Due to dimensions: y = b2 + f(b1 + xW1)W2
        self.assertTrue((y == params["params"]["RandomDenseLinearFA_1"]["Dense_0"]["bias"] + 
                         nn.sigmoid((params["params"]["RandomDenseLinearFA_0"]["Dense_0"]["bias"]+ 
                          x @ params["params"]["RandomDenseLinearFA_0"]["Dense_0"]["kernel"]))
                          @ params["params"]["RandomDenseLinearFA_1"]["Dense_0"]["kernel"]).all())
        
    def test_derivative_single_layer(self):
        model = RandomDenseLinearFA(10)
        x = jnp.ones((1, 5))
        params = model.init(jax.random.PRNGKey(0), x)
        y = model.apply(params, jnp.ones((1, 5)))
        target = y-1

        def loss_fn(params, x):
            return 0.5 * jnp.sum((model.apply(params, x) - target) ** 2)
        
        delta_params, delta_x = jax.grad(loss_fn, argnums=(0,1))(params, x)
        """
        CONTINUE HERE
        """

if __name__ == '__main__':
    unittest.main()