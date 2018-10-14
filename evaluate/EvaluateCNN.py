"""
This module is used to evaluate the CNN model.
"""
import tensorflow as tf

from evaluate.EvaluateModel import EvaluateModel
from model.CNNModel import ConvolutionalNeuralNetwork
from shared import Utils


def avaliar():
    """
    Evaluate the CNN model.
    """
    result_path = Utils.get_results_path_cnn(is_training=False)

    eval = EvaluateModel(ConvolutionalNeuralNetwork)

    eval.evaluate(result_path)


def main(argv=None):
    """
    Initiate the evaluation.
    """
    avaliar()


if __name__ == "__main__":
    tf.app.run()
