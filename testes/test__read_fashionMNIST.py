from unittest import TestCase

from input import _read_fashionMNIST


class Test_read_fashionMNIST(TestCase):

    def test__read_fashionMNIST_train(self):
        x, y = _read_fashionMNIST(is_training=True)
        self.assertEqual((60000, 28, 28), x.shape)
        self.assertEqual((60000,), y.shape)

    def test__read_fashionMNIST_test(self):
        x, y = _read_fashionMNIST(is_training=False)
        self.assertEqual((10000, 28, 28), x.shape)
        self.assertEqual((10000,), y.shape)
