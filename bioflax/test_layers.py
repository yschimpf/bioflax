import unittest
from .model import (
    RandomDenseLinearFA,
    RandomDenseLinearKP,
    RandomDenseLinearDFAOutput,
    RandomDenseLinearDFAHidden
)

class Test_RandomDenseLinearFA(unittest.TestCase):

    def test_init(self):
        test = RandomDenseLinearFA(4)
        self.assertEqual(test.features, 4)
        self.assertEqual(test.bias, True)
        self.assertEqual(test.kernel, True)