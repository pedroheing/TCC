from unittest import TestCase

from input import get_batch_data


class TestGet_batch_data(TestCase):

    def test_get_batch_data_fashion(self):
        x, y = get_batch_data("fashionMNIST", 128, 8)
        print(x.shape)
        print(y.shape)

    def test_get_batch_data_traffic(self):
        x, y = get_batch_data("traffic_sign", 128, 8)
        print(x.shape)
        print(y.shape)
