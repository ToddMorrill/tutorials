import unittest

import numpy as np

from utils import load_yaml
from train import train


class TestReproducibility(unittest.TestCase):
    def test_initialization(self):
        config = load_yaml('./config.yaml')
        average_loss_tracker = train(config)
        filename = 'avg_loss_tracker'
        load_average_loss_tracker = np.load(filename + '.npy')

        self.assertEqual(average_loss_tracker.all(),
                         load_average_loss_tracker.all())


if __name__ == '__main__':
    unittest.main()