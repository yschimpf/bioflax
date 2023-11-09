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
        y = xW + b
        """
        model = RandomDenseLinearFA(10)
        params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 5)))
        x = jnp.ones((1, 5))
        y = model.apply(params, x)
        self.assertEqual(y.shape, (1, 10))
        # y = b + xW
        self.assertTrue((y == params["params"]["Dense_0"]["bias"] +
                        x @ params["params"]["Dense_0"]["kernel"]).all())

    def test_multi_layer_output(self):
        """
        Check whether for a two layer network the output is correct
        y = (xW_1 + b_1)W_2 + b_2
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
        # y = b2 + (b1 + xW1)W2
        self.assertTrue((y == params["params"]["RandomDenseLinearFA_1"]["Dense_0"]["bias"] +
                         (params["params"]["RandomDenseLinearFA_0"]["Dense_0"]["bias"] +
                          x @ params["params"]["RandomDenseLinearFA_0"]["Dense_0"]["kernel"])
                         @ params["params"]["RandomDenseLinearFA_1"]["Dense_0"]["kernel"]).all())

    def test_multi_layer_activation_output(self):
        """
        Check whether for a two layer network the output is correct
        y = f(xW_1 + b1)W_2 + b2, where f is the sigmoid activation function
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
        # y = b2 + f(b1 + xW1)W2
        self.assertTrue((y == params["params"]["RandomDenseLinearFA_1"]["Dense_0"]["bias"] +
                         nn.sigmoid((params["params"]["RandomDenseLinearFA_0"]["Dense_0"]["bias"] +
                                     x @ params["params"]["RandomDenseLinearFA_0"]["Dense_0"]["kernel"]))
                         @ params["params"]["RandomDenseLinearFA_1"]["Dense_0"]["kernel"]).all())

    def test_derivative_single_layer(self):
        """
        Check whether the derivatives are correct for a single layer network:
            model(x) = xW + b; with feedback matrix B
        For equations that need to hold please refer to the inline comments.
        """
        model = RandomDenseLinearFA(10)
        x = jnp.ones((1, 5))
        params = model.init(jax.random.PRNGKey(0), x)
        y = model.apply(params, jnp.ones((1, 5)))
        target = y-1

        def loss_fn(params, x):
            return 0.5 * jnp.sum((model.apply(params, x) - target) ** 2)

        delta_params, delta_x = jax.grad(loss_fn, argnums=(0, 1))(params, x)

        # dL/dB = 0
        self.assertTrue(
            (delta_params["params"]["B"] == jnp.zeros((5, 10))).all())
        # dL/dx = B(y-target)
        self.assertTrue(jnp.allclose(
            delta_x, (params["params"]["B"]@(y-target).T).T))
        # dL/dW = x^T * (y-target)
        self.assertTrue(jnp.allclose(
            delta_params["params"]["Dense_0"]["kernel"], x.T@(y-target)))
        # dL/db = y-target
        self.assertTrue(jnp.allclose(
            delta_params["params"]["Dense_0"]["bias"], y-target))

    def test_derivative_multi_layer(self):
        """
        Check whether the derivatives are correct for a two layer network:
            model(x) = (xW_1 + b_1)W_2 + b_2; with feedback matrices B_1 and B_2
        """
        class MultiLayerModel(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = RandomDenseLinearFA(15)(x)
                x = RandomDenseLinearFA(10)(x)
                return x

        model = MultiLayerModel()
        x = jnp.ones((1, 5))
        params = model.init(jax.random.PRNGKey(0), x)
        y = model.apply(params, jnp.ones((1, 5)))
        target = y-1

        def loss_fn(params, x):
            return 0.5 * jnp.sum((model.apply(params, x) - target) ** 2)

        # v_1 = xW_1 + b_1
        v_1 = (x @ params["params"]["RandomDenseLinearFA_0"]["Dense_0"]["kernel"]
               + params["params"]["RandomDenseLinearFA_0"]["Dense_0"]["bias"])

        delta_params, delta_x = jax.grad(loss_fn, argnums=(0, 1))(params, x)

        # dL/dB_2 = 0
        self.assertTrue(
            (delta_params["params"]["RandomDenseLinearFA_1"]["B"] == jnp.zeros((15, 10))).all())
        # dL/dB_1 = 0
        self.assertTrue(
            (delta_params["params"]["RandomDenseLinearFA_0"]["B"] == jnp.zeros((5, 15))).all())
        # dL/dx = B_1 * B_2 * (y-target)^T
        self.assertTrue(jnp.allclose(delta_x, (params["params"]["RandomDenseLinearFA_0"]["B"] @
                                               (params["params"]["RandomDenseLinearFA_1"]["B"]@(y-target).T)).T))
        # dL/dW_2 = v_1^T * (y-target)
        self.assertTrue(jnp.allclose(
            delta_params["params"]["RandomDenseLinearFA_1"]["Dense_0"]["kernel"], v_1.T@(y-target)))
        # dL/db_2 = y-target
        self.assertTrue(jnp.allclose(
            delta_params["params"]["RandomDenseLinearFA_1"]["Dense_0"]["bias"], y-target))
        # dL/dW_1 = x^T * (B_2 * (y-target)^T)^T = x^T * dL/dv_1
        self.assertTrue(jnp.allclose(delta_params["params"]["RandomDenseLinearFA_0"]["Dense_0"]["kernel"],
                                     x.T@(params["params"]["RandomDenseLinearFA_1"]["B"]@(y-target).T).T))
        # dL/db_1 = (B_2 * (y-target)^T)^T
        self.assertTrue(jnp.allclose(delta_params["params"]["RandomDenseLinearFA_0"]["Dense_0"]["bias"],
                                     (params["params"]["RandomDenseLinearFA_1"]["B"]@(y-target).T).T))

    def test_derivative_multi_layer_activation(self):
        """
        Check whether the derivatives are correct for a two layer network with activation function in the hidden layer:
            model(x) = f(xW_1 + b1)W_2 + b2, where f is the sigmoid activation function and feedback matrices B_1 and B_2
        """
        class MultiLayerModelAct(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = RandomDenseLinearFA(15)(x)
                x = nn.sigmoid(x)
                x = RandomDenseLinearFA(10)(x)
                return x

        model = MultiLayerModelAct()
        x = jnp.ones((1, 5))
        params = model.init(jax.random.PRNGKey(0), x)
        y = model.apply(params, jnp.ones((1, 5)))
        target = y-1

        def loss_fn(params, x):
            return 0.5 * jnp.sum((model.apply(params, x) - target) ** 2)

        # v_1 = xW_1 + b_1
        z_1 = (x @ params["params"]["RandomDenseLinearFA_0"]["Dense_0"]["kernel"]
               + params["params"]["RandomDenseLinearFA_0"]["Dense_0"]["bias"])
        v_1 = nn.sigmoid(z_1)

        def der_sig(x):
            return (nn.sigmoid(x)*(1-nn.sigmoid(x))).T

        delta_params, delta_x = jax.grad(loss_fn, argnums=(0, 1))(params, x)

        # dL/dB_2 = 0
        self.assertTrue(
            (delta_params["params"]["RandomDenseLinearFA_1"]["B"] == jnp.zeros((15, 10))).all())
        # dL/dB_1 = 0
        self.assertTrue(
            (delta_params["params"]["RandomDenseLinearFA_0"]["B"] == jnp.zeros((5, 15))).all())
        # dL/dx = (B_1 * (f'(v_1)^T *_ B_2 * (y-target)^T))^T; where *_is the hadamard (elementwise) product
        self.assertTrue(jnp.allclose(delta_x, (params["params"]["RandomDenseLinearFA_0"]["B"] @
                                               jnp.multiply(der_sig(z_1),
                                                            (params["params"]["RandomDenseLinearFA_1"]["B"]@(y-target).T))).T))
        # dL/dW_2 = v_1^T * (y-target)
        self.assertTrue(jnp.allclose(
            delta_params["params"]["RandomDenseLinearFA_1"]["Dense_0"]["kernel"], v_1.T@(y-target)))
        # dL/db_2 = y-target
        self.assertTrue(jnp.allclose(
            delta_params["params"]["RandomDenseLinearFA_1"]["Dense_0"]["bias"], y-target))
        # dL/dW_1 = x^T * (f'(v_1)^T *_ B_2 * (y-target)^T))^T = x^T * dL/dz_1
        self.assertTrue(jnp.allclose(delta_params["params"]["RandomDenseLinearFA_0"]["Dense_0"]["kernel"],
                                     x.T@(jnp.multiply(der_sig(z_1),
                                                       (params["params"]["RandomDenseLinearFA_1"]["B"]@(y-target).T))).T))
        # dL/db_1 = (f'(v_1)^T *_ B_2 * (y-target)^T))^T
        self.assertTrue(jnp.allclose(delta_params["params"]["RandomDenseLinearFA_0"]["Dense_0"]["bias"],
                                     (jnp.multiply(der_sig(z_1),
                                                   (params["params"]["RandomDenseLinearFA_1"]["B"]@(y-target).T))).T))


class Test_RandomDenseLinearKP(unittest.TestCase):

    def test_init(self):
        """
        Check whether the number of output features is correct
        """
        test = RandomDenseLinearKP(4)
        self.assertEqual(test.features, 4)

    def test_single_layer_output(self):
        """
        Check whether for a single layer network the output is correct
        y = xW + b
        """
        model = RandomDenseLinearKP(10)
        params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 5)))
        x = jnp.ones((1, 5))
        y = model.apply(params, x)
        self.assertEqual(y.shape, (1, 10))
        # y = b + xW
        self.assertTrue((y == params["params"]["Dense_0"]["bias"] +
                        x @ params["params"]["Dense_0"]["kernel"]).all())

    def test_multi_layer_output(self):
        """
        Check whether for a two layer network the output is correct
        y = (xW_1 + b_1)W_2 + b_2
        """
        class MultiLayerModel(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = RandomDenseLinearKP(15)(x)
                x = RandomDenseLinearKP(10)(x)
                return x

        model = MultiLayerModel()
        params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 5)))
        x = jnp.ones((1, 5))
        y = model.apply(params, x)
        self.assertEqual(y.shape, (1, 10))
        # y = b2 + (b1 + xW1)W2
        self.assertTrue((y == params["params"]["RandomDenseLinearKP_1"]["Dense_0"]["bias"] +
                         (params["params"]["RandomDenseLinearKP_0"]["Dense_0"]["bias"] +
                          x @ params["params"]["RandomDenseLinearKP_0"]["Dense_0"]["kernel"])
                         @ params["params"]["RandomDenseLinearKP_1"]["Dense_0"]["kernel"]).all())

    def test_multi_layer_activation_output(self):
        """
        Check whether for a two layer network the output is correct
        y = f(xW_1 + b1)W_2 + b2, where f is the sigmoid activation function
        """
        class MultiLayerModelAct(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = RandomDenseLinearKP(15)(x)
                x = nn.sigmoid(x)
                x = RandomDenseLinearKP(10)(x)
                return x

        model = MultiLayerModelAct()
        params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 5)))
        x = jnp.ones((1, 5))
        y = model.apply(params, x)
        self.assertEqual(y.shape, (1, 10))
        # y = b2 + f(b1 + xW1)W2
        self.assertTrue((y == params["params"]["RandomDenseLinearKP_1"]["Dense_0"]["bias"] +
                         nn.sigmoid((params["params"]["RandomDenseLinearKP_0"]["Dense_0"]["bias"] +
                                     x @ params["params"]["RandomDenseLinearKP_0"]["Dense_0"]["kernel"]))
                         @ params["params"]["RandomDenseLinearKP_1"]["Dense_0"]["kernel"]).all())

    def test_derivative_single_layer(self):
        """
        Check whether the derivatives are correct for a single layer network:
            model(x) = xW + b; with feedback matrix B
        For equations that need to hold please refer to the inline comments.
        """
        model = RandomDenseLinearKP(10)
        x = jnp.ones((1, 5))
        params = model.init(jax.random.PRNGKey(0), x)
        y = model.apply(params, jnp.ones((1, 5)))
        target = y-1

        def loss_fn(params, x):
            return 0.5 * jnp.sum((model.apply(params, x) - target) ** 2)

        delta_params, delta_x = jax.grad(loss_fn, argnums=(0, 1))(params, x)

        # dL/dB = 0
        self.assertTrue(
            (delta_params["params"]["B"] == delta_params["params"]["Dense_0"]["kernel"]).all())
        # dL/dx = B(y-target)
        self.assertTrue(jnp.allclose(
            delta_x, (params["params"]["B"]@(y-target).T).T))
        # dL/dW = x^T * (y-target)
        self.assertTrue(jnp.allclose(
            delta_params["params"]["Dense_0"]["kernel"], x.T@(y-target)))
        # dL/db = y-target
        self.assertTrue(jnp.allclose(
            delta_params["params"]["Dense_0"]["bias"], y-target))

    def test_derivative_multi_layer(self):
        """
        Check whether the derivatives are correct for a two layer network:
            model(x) = (xW_1 + b_1)W_2 + b_2; with feedback matrices B_1 and B_2
        """
        class MultiLayerModel(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = RandomDenseLinearKP(15)(x)
                x = RandomDenseLinearKP(10)(x)
                return x

        model = MultiLayerModel()
        x = jnp.ones((1, 5))
        params = model.init(jax.random.PRNGKey(0), x)
        y = model.apply(params, jnp.ones((1, 5)))
        target = y-1

        def loss_fn(params, x):
            return 0.5 * jnp.sum((model.apply(params, x) - target) ** 2)

        # v_1 = xW_1 + b_1
        v_1 = (x @ params["params"]["RandomDenseLinearKP_0"]["Dense_0"]["kernel"]
               + params["params"]["RandomDenseLinearKP_0"]["Dense_0"]["bias"])

        delta_params, delta_x = jax.grad(loss_fn, argnums=(0, 1))(params, x)

        # dL/dB_2 = dL/dW_2
        self.assertTrue((delta_params["params"]["RandomDenseLinearKP_1"]["B"] ==
                         delta_params["params"]["RandomDenseLinearKP_1"]["Dense_0"]["kernel"]).all())
        # dL/dB_1 = dL/dW_1
        self.assertTrue((delta_params["params"]["RandomDenseLinearKP_0"]["B"] ==
                         delta_params["params"]["RandomDenseLinearKP_0"]["Dense_0"]["kernel"]).all())
        # dL/dx = B_1 * B_2 * (y-target)^T
        self.assertTrue(jnp.allclose(delta_x, (params["params"]["RandomDenseLinearKP_0"]["B"] @
                                               (params["params"]["RandomDenseLinearKP_1"]["B"]@(y-target).T)).T))
        # dL/dW_2 = v_1^T * (y-target)
        self.assertTrue(jnp.allclose(
            delta_params["params"]["RandomDenseLinearKP_1"]["Dense_0"]["kernel"], v_1.T@(y-target)))
        # dL/db_2 = y-target
        self.assertTrue(jnp.allclose(
            delta_params["params"]["RandomDenseLinearKP_1"]["Dense_0"]["bias"], y-target))
        # dL/dW_1 = x^T * (B_2 * (y-target)^T)^T = x^T * dL/dv_1
        self.assertTrue(jnp.allclose(delta_params["params"]["RandomDenseLinearKP_0"]["Dense_0"]["kernel"],
                                     x.T@(params["params"]["RandomDenseLinearKP_1"]["B"]@(y-target).T).T))
        # dL/db_1 = (B_2 * (y-target)^T)^T
        self.assertTrue(jnp.allclose(delta_params["params"]["RandomDenseLinearKP_0"]["Dense_0"]["bias"],
                                     (params["params"]["RandomDenseLinearKP_1"]["B"]@(y-target).T).T))

    def test_derivative_multi_layer_activation(self):
        """
        Check whether the derivatives are correct for a two layer network with activation function in the hidden layer:
            model(x) = f(xW_1 + b1)W_2 + b2, where f is the sigmoid activation function and feedback matrices B_1 and B_2
        """
        class MultiLayerModelAct(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = RandomDenseLinearKP(15)(x)
                x = nn.sigmoid(x)
                x = RandomDenseLinearKP(10)(x)
                return x

        model = MultiLayerModelAct()
        x = jnp.ones((1, 5))
        params = model.init(jax.random.PRNGKey(0), x)
        y = model.apply(params, jnp.ones((1, 5)))
        target = y-1

        def loss_fn(params, x):
            return 0.5 * jnp.sum((model.apply(params, x) - target) ** 2)

        # v_1 = xW_1 + b_1
        z_1 = (x @ params["params"]["RandomDenseLinearKP_0"]["Dense_0"]["kernel"]
               + params["params"]["RandomDenseLinearKP_0"]["Dense_0"]["bias"])
        v_1 = nn.sigmoid(z_1)

        def der_sig(x):
            return (nn.sigmoid(x)*(1-nn.sigmoid(x))).T

        delta_params, delta_x = jax.grad(loss_fn, argnums=(0, 1))(params, x)

        # dL/dB_2 = dL/dW_2
        self.assertTrue((delta_params["params"]["RandomDenseLinearKP_1"]["B"] ==
                         delta_params["params"]["RandomDenseLinearKP_1"]["Dense_0"]["kernel"]).all())
        # dL/dB_1 = dL/dW_1
        self.assertTrue((delta_params["params"]["RandomDenseLinearKP_0"]["B"] ==
                         delta_params["params"]["RandomDenseLinearKP_0"]["Dense_0"]["kernel"]).all())
        # dL/dx = (B_1 * (f'(v_1)^T *_ B_2 * (y-target)^T))^T; where *_is the hadamard (elementwise) product
        self.assertTrue(jnp.allclose(delta_x, (params["params"]["RandomDenseLinearKP_0"]["B"] @
                                               jnp.multiply(der_sig(z_1),
                                                            (params["params"]["RandomDenseLinearKP_1"]["B"]@(y-target).T))).T))
        # dL/dW_2 = v_1^T * (y-target)
        self.assertTrue(jnp.allclose(
            delta_params["params"]["RandomDenseLinearKP_1"]["Dense_0"]["kernel"], v_1.T@(y-target)))
        # dL/db_2 = y-target
        self.assertTrue(jnp.allclose(
            delta_params["params"]["RandomDenseLinearKP_1"]["Dense_0"]["bias"], y-target))
        # dL/dW_1 = x^T * (f'(v_1)^T *_ B_2 * (y-target)^T))^T = x^T * dL/dz_1
        self.assertTrue(jnp.allclose(delta_params["params"]["RandomDenseLinearKP_0"]["Dense_0"]["kernel"],
                                     x.T@(jnp.multiply(der_sig(z_1),
                                                       (params["params"]["RandomDenseLinearKP_1"]["B"]@(y-target).T))).T))
        # dL/db_1 = (f'(v_1)^T *_ B_2 * (y-target)^T))^T
        self.assertTrue(jnp.allclose(delta_params["params"]["RandomDenseLinearKP_0"]["Dense_0"]["bias"],
                                     (jnp.multiply(der_sig(z_1),
                                                   (params["params"]["RandomDenseLinearKP_1"]["B"]@(y-target).T))).T))


class Test_RandomDenseLinearDFA(unittest.TestCase):

    def test_output_init(self):
        """
        Check whether the number of output features is correct
        """
        model = RandomDenseLinearDFAOutput(4)
        self.assertEqual(model.features, 4)

    def test_hidden_init(self):
        """
        Check whether the number of output features is correct
        """
        model = RandomDenseLinearDFAHidden(4, 4, (lambda x: x))
        self.assertEqual(model.features, 4)

    def test_output_single_layer_output(self):
        """
        Check whether output layer generates correct output
        y = xW + b
        """
        model = RandomDenseLinearDFAOutput(10)
        x = jnp.ones((1, 5))
        params = model.init(jax.random.PRNGKey(0), x)
        y = model.apply(params, x)
        self.assertTrue(jnp.allclose(
            y, params["params"]["Dense_0"]["bias"] + x @ params["params"]["Dense_0"]["kernel"]))

    def test_hidden_single_layer_output(self):
        """
        Check whether hidden layer generates correct output
        y = xW + b
        """
        model = RandomDenseLinearDFAHidden(10, 5, (lambda x: x))
        x = jnp.ones((1, 5))
        params = model.init(jax.random.PRNGKey(0), x)
        y = model.apply(params, x)
        self.assertTrue(jnp.allclose(
            y, params["params"]["Dense_0"]["bias"] + x @ params["params"]["Dense_0"]["kernel"]))

    def test_multi_layer_output(self):
        """
        Check whether for a two layer network the output is correct
        y = (xW_1 + b_1)W_2 + b_2
        """
        class MultiLayerModel(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = RandomDenseLinearDFAHidden(15, 10, (lambda x: x))(x)
                x = RandomDenseLinearDFAOutput(10)(x)
                return x

        model = MultiLayerModel()
        params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 5)))
        x = jnp.ones((1, 5))
        y = model.apply(params, x)
        self.assertEqual(y.shape, (1, 10))
        # y = b2 + (b1 + xW1)W2
        self.assertTrue((y == params["params"]["RandomDenseLinearDFAOutput_0"]["Dense_0"]["bias"] +
                         (params["params"]["RandomDenseLinearDFAHidden_0"]["Dense_0"]["bias"] +
                          x @ params["params"]["RandomDenseLinearDFAHidden_0"]["Dense_0"]["kernel"])
                         @ params["params"]["RandomDenseLinearDFAOutput_0"]["Dense_0"]["kernel"]).all())

    def test_multi_layer_output_activation(self):
        """
        Check whether for a two layer network the output is correct
        y = f(xW_1 + b_1)W_2 + b_2
        """
        class MultiLayerModel(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = RandomDenseLinearDFAHidden(15, 10, nn.sigmoid)(x)
                x = RandomDenseLinearDFAOutput(10)(x)
                return x

        model = MultiLayerModel()
        params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 5)))
        x = jnp.ones((1, 5))
        y = model.apply(params, x)
        self.assertEqual(y.shape, (1, 10))
        # y = b2 + (b1 + xW1)W2
        self.assertTrue((y == params["params"]["RandomDenseLinearDFAOutput_0"]["Dense_0"]["bias"] +
                         nn.sigmoid(params["params"]["RandomDenseLinearDFAHidden_0"]["Dense_0"]["bias"] +
                                    x @ params["params"]["RandomDenseLinearDFAHidden_0"]["Dense_0"]["kernel"])
                         @ params["params"]["RandomDenseLinearDFAOutput_0"]["Dense_0"]["kernel"]).all())

    def test_output_derivative_single_layer(self):
        """
        Check whether the derivatives are correct for a single layer network:
            model(x) = xW + b; with feedback matrix B
        For equations that need to hold please refer to the inline comments.
        """
        model = RandomDenseLinearDFAOutput(10)
        x = jnp.ones((1, 5))
        params = model.init(jax.random.PRNGKey(0), x)
        y = model.apply(params, jnp.ones((1, 5)))
        target = y-1

        def loss_fn(params, x):
            return 0.5 * jnp.sum((model.apply(params, x) - target) ** 2)

        delta_params, delta_x = jax.grad(loss_fn, argnums=(0, 1))(params, x)

        # dL/dx = (y-target)
        self.assertTrue(jnp.allclose(delta_x, y-target))
        # dL/dW = x^T * (y-target)
        self.assertTrue(jnp.allclose(
            delta_params["params"]["Dense_0"]["kernel"], x.T@(y-target)))
        # dL/db = y-target
        self.assertTrue(jnp.allclose(
            delta_params["params"]["Dense_0"]["bias"], y-target))

    def test_hidden_derivative_single_layer(self):
        """
        Check whether the derivatives are correct for a single layer network:
            model(x) = xW + b; with feedback matrix B
        For equations that need to hold please refer to the inline comments.
        """
        model = RandomDenseLinearDFAHidden(10, 10, (lambda x: x))
        x = jnp.ones((1, 5))
        params = model.init(jax.random.PRNGKey(0), x)
        y = model.apply(params, jnp.ones((1, 5)))
        target = y-1

        def loss_fn(params, x):
            return 0.5 * jnp.sum((model.apply(params, x) - target) ** 2)

        delta_params, delta_x = jax.grad(loss_fn, argnums=(0, 1))(params, x)

        # dL/dx = (y-target)
        self.assertTrue(jnp.allclose(delta_x, y-target))
        # dL/dW = x^T * (B * (y-target)^T)^T; note that the Hidden layer is not constructed to serve as an outputlayer
        # which it essentially does here. The derivative makes no semnantic sense.
        self.assertTrue(jnp.allclose(delta_params["params"]["Dense_0"]["kernel"],
                                     x.T@(params["params"]["B"]@(y-target).T).T))
        # dL/db = (B * (y-target)^T)^T; note that the Hidden layer is not constructed to serve as an outputlayer
        # which it essentially does here. The derivative makes no semnantic sense.
        self.assertTrue(jnp.allclose(
            delta_params["params"]["Dense_0"]["bias"], (params["params"]["B"]@(y-target).T).T))

    def test_derivative_multi_layer(self):
        """
        Check whether the derivatives are correct for a two layer network:
            model(x) = (xW_1 + b_1)W_2 + b_2; with feedback matrices B_1 and B_2
        """
        class MultiLayerModel(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = RandomDenseLinearDFAHidden(15, 10, (lambda x: x))(x)
                x = RandomDenseLinearDFAOutput(10)(x)
                return x

        model = MultiLayerModel()
        x = jnp.ones((1, 5))
        params = model.init(jax.random.PRNGKey(0), x)
        y = model.apply(params, jnp.ones((1, 5)))
        target = y-1

        def loss_fn(params, x):
            return 0.5 * jnp.sum((model.apply(params, x) - target) ** 2)

        # v_1 = xW_1 + b_1
        v_1 = (x @ params["params"]["RandomDenseLinearDFAHidden_0"]["Dense_0"]["kernel"]
               + params["params"]["RandomDenseLinearDFAHidden_0"]["Dense_0"]["bias"])

        delta_params, delta_x = jax.grad(loss_fn, argnums=(0, 1))(params, x)

        # dL/dB_1 = 0
        self.assertTrue(
            (delta_params["params"]["RandomDenseLinearDFAHidden_0"]["B"] == jnp.zeros((15, 10))).all())
        # dL/dx = (y-target)^T
        self.assertTrue(jnp.allclose(delta_x, y-target))
        # dL/dW_2 = v_1^T * (y-target)
        self.assertTrue(jnp.allclose(
            delta_params["params"]["RandomDenseLinearDFAOutput_0"]["Dense_0"]["kernel"], v_1.T@(y-target)))
        # dL/db_2 = y-target
        self.assertTrue(jnp.allclose(
            delta_params["params"]["RandomDenseLinearDFAOutput_0"]["Dense_0"]["bias"], y-target))
        # dL/dW_1 = x^T * (B_2 * (y-target)^T)^T = x^T * dL/dv_1
        self.assertTrue(jnp.allclose(delta_params["params"]["RandomDenseLinearDFAHidden_0"]["Dense_0"]["kernel"],
                                     x.T@(params["params"]["RandomDenseLinearDFAHidden_0"]["B"]@(y-target).T).T))
        # dL/db_1 = (B_2 * (y-target)^T)^T
        self.assertTrue(jnp.allclose(delta_params["params"]["RandomDenseLinearDFAHidden_0"]["Dense_0"]["bias"],
                                     (params["params"]["RandomDenseLinearDFAHidden_0"]["B"]@(y-target).T).T))

    def test_derivative_multi_layer_activation(self):
        """
        Check whether the derivatives are correct for a two layer network:
            model(x) = (xW_1 + b_1)W_2 + b_2; with feedback matrices B_1 and B_2
        """
        class MultiLayerModel(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = RandomDenseLinearDFAHidden(20, 10, nn.sigmoid)(x)
                x = RandomDenseLinearDFAHidden(15, 10, nn.sigmoid)(x)
                x = RandomDenseLinearDFAOutput(10)(x)
                return x

        model = MultiLayerModel()
        x = jnp.ones((1, 5))
        params = model.init(jax.random.PRNGKey(0), x)
        y = model.apply(params, jnp.ones((1, 5)))
        target = y-1

        def loss_fn(params, x):
            return 0.5 * jnp.sum((model.apply(params, x) - target) ** 2)

        # v_1 = f(xW_1 + b_1) = f(z_1)
        z_1 = (x @ params["params"]["RandomDenseLinearDFAHidden_0"]["Dense_0"]["kernel"]
               + params["params"]["RandomDenseLinearDFAHidden_0"]["Dense_0"]["bias"])
        v_1 = nn.sigmoid(z_1)

        # v_2 = f(v_1W_2 + b_2) = f(z_2)
        z_2 = (v_1 @ params["params"]["RandomDenseLinearDFAHidden_1"]["Dense_0"]["kernel"]
               + params["params"]["RandomDenseLinearDFAHidden_1"]["Dense_0"]["bias"])
        v_2 = nn.sigmoid(z_2)

        def deriv_sig(x):
            return (nn.sigmoid(x)*(1-nn.sigmoid(x))).T

        delta_params, delta_x = jax.grad(loss_fn, argnums=(0, 1))(params, x)

        # dL/dB_2 = 0
        self.assertTrue(
            (delta_params["params"]["RandomDenseLinearDFAHidden_1"]["B"] == jnp.zeros((15, 10))).all())
        # dL/dB_1 = 0
        self.assertTrue(
            (delta_params["params"]["RandomDenseLinearDFAHidden_0"]["B"] == jnp.zeros((20, 10))).all())
        # dL/dx = (y-target)^T
        self.assertTrue(jnp.allclose(delta_x, y-target))
        # dL/dW_2 = v_1^T * (y-target)
        self.assertTrue(jnp.allclose(
            delta_params["params"]["RandomDenseLinearDFAOutput_0"]["Dense_0"]["kernel"], v_2.T@(y-target)))
        # dL/db_2 = y-target
        self.assertTrue(jnp.allclose(
            delta_params["params"]["RandomDenseLinearDFAOutput_0"]["Dense_0"]["bias"], y-target))
        # dL/dW_1 = x^T * (B_2 * (y-target)^T)^T = x^T * dL/dv_1
        self.assertTrue(jnp.allclose(delta_params["params"]["RandomDenseLinearDFAHidden_1"]["Dense_0"]["kernel"],
                                     v_1.T@(jnp.multiply(deriv_sig(z_2), (params["params"]["RandomDenseLinearDFAHidden_1"]["B"]@(y-target).T))).T))
        # dL/db_1 = (B_2 * (y-target)^T)^T
        self.assertTrue(jnp.allclose(delta_params["params"]["RandomDenseLinearDFAHidden_1"]["Dense_0"]["bias"],
                                     (jnp.multiply(deriv_sig(z_2), (params["params"]["RandomDenseLinearDFAHidden_1"]["B"]@(y-target).T))).T))
        # dL/dW_1 = x^T * (B_2 * (y-target)^T)^T = x^T * dL/dv_1
        self.assertTrue(jnp.allclose(delta_params["params"]["RandomDenseLinearDFAHidden_0"]["Dense_0"]["kernel"],
                                     x.T@(jnp.multiply(deriv_sig(z_1), (params["params"]["RandomDenseLinearDFAHidden_0"]["B"]@(y-target).T))).T))
        # dL/db_1 = (B_2 * (y-target)^T)^T
        self.assertTrue(jnp.allclose(delta_params["params"]["RandomDenseLinearDFAHidden_0"]["Dense_0"]["bias"],
                                     (jnp.multiply(deriv_sig(z_1), (params["params"]["RandomDenseLinearDFAHidden_0"]["B"]@(y-target).T))).T))


if __name__ == '__main__':
    unittest.main()
