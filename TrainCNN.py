"""
This module is used to execute the training of the CNN model.
"""

import tensorflow as tf

from model.CNNModel import ConvolutionalNeuralNetwork
from train.TrainModel import TrainModel


def train():
    """
    Train the CNN model
    """
    result_path = 'results/'

    train_model = TrainModel(ConvolutionalNeuralNetwork)

    train_model.train(result_path, result_path)


def main(argv=None):
    """
    Initiate the training.
    """
    train()


if __name__ == "__main__":
    tf.app.run()
