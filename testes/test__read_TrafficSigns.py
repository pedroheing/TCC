from unittest import TestCase

from input import _read_trafficSigns


class Test_read_trafficSigns(TestCase):
    def test__read_trafficSigns_train(self):
        x, y = _read_trafficSigns(is_training=True)
        self.assertEqual(4575, len(x))
        self.assertEqual(4575, len(y))

    def test__read_trafficSigns_test(self):
        x, y = _read_trafficSigns(is_training=False)
        self.assertEqual(2520, len(x))
        self.assertEqual(2520, len(y))
    # self.assertEqual((10000, 28, 28), x.shape)
    # self.assertEqual((10000,), y.shape)
