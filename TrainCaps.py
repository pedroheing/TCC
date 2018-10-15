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
    result_path = 'results/trainingCaps'

    train_model = TrainModel(CapsNet)

    train_model.train(result_path)


def main(argv=None):
    """
    Initiate the training.
    """
    train()


if __name__ == "__main__":
    tf.app.run()
