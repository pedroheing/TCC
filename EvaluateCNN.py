"""
This module is used to evaluate the CNN model.
"""
import tensorflow as tf

from evaluate.EvaluateModel import EvaluateModel
from model.CNNModel import ConvolutionalNeuralNetwork


def evaluate():
    """
    Evaluate the CNN model.
    """
    result_path = 'results/evaluationCNN'

    restore_path = 'results/trainingCNN'

    eval = EvaluateModel(ConvolutionalNeuralNetwork)

    eval.evaluate(result_path, restore_path)


def main(argv=None):
    """
    Initiate the evaluation.
    """
    evaluate()


if __name__ == "__main__":
    tf.app.run()
