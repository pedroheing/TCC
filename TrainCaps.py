"""
This module is used to execute the training of the CapsNet model.
"""
import tensorflow as tf

from model.CapsModel import CapsNet
from train.TrainModel import TrainModel


def train():
    """
    Train the CapsNet model
    """
<<<<<<< HEAD
    result_path = 'results/'
=======
    result_path = 'results/trainingCaps'
>>>>>>> 77c759207b6ac061903ac4009d1e04092c07c4ff

    train_model = TrainModel(CapsNet)

    train_model.train(result_path, result_path)


def main(argv=None):
    """
    Initiate the training.
    """
    train()


if __name__ == "__main__":
    tf.app.run()
