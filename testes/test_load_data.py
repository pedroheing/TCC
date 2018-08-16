from unittest import TestCase

from input import load_data


class TestLoad_data(TestCase):

    def test_load_data_fashion_train(self):
        x, y = load_data("fashionMNIST", is_training=True)
        self.assertEqual((60000, 28, 28, 1), x.shape)
        self.assertEqual((60000,), y.shape)

    def test_load_data_fashion_test(self):
        x, y = load_data("fashionMNIST", is_training=False)
        self.assertEqual((10000, 28, 28, 1), x.shape)
        self.assertEqual((10000,), y.shape)

    def test_load_data_traffic_train(self):
        x, y = load_data("traffic_sign", is_training=True)
        self.assertEqual((4575, 28, 28, 1), x.shape)
        self.assertEqual(4575, len(y))

    def test_load_data_traffic_test(self):
        x, y = load_data("traffic_sign", is_training=False)
        self.assertEqual((2520, 28, 28, 1), x.shape)
        self.assertEqual(2520, len(y))
