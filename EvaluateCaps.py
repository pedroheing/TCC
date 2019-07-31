"""
This module is used to evaluate the CapsNet model.
"""
import tensorflow as tf

from evaluate.EvaluateModel import EvaluateModel
from model.CapsModel import CapsNet


def avaliar():
    """
    Evaluate the CapsNet model.
    """
    result_path = 'results/'

    restore_path = 'results/'

    eval = EvaluateModel(CapsNet)

    eval.evaluate(result_path, restore_path)


def main(argv=None):
    """
    Initiate the evaluation.
    """
    avaliar()


if __name__ == "__main__":
    tf.app.run()
