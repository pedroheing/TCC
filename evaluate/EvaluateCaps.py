"""
This module is used to evaluate the CapsNet model.
"""
import tensorflow as tf

from evaluate.EvaluateModel import EvaluateModel
from model.CapsModel import CapsNet
from shared import Utils


def avaliar():
    """
    Evaluate the CapsNet model.
    """
    result_path = Utils.get_results_path_caps(is_training=False)

    eval = EvaluateModel(CapsNet)

    eval.evaluate(result_path)


def main(argv=None):
    """
    Initiate the evaluation.
    """
    avaliar()


if __name__ == "__main__":
    tf.app.run()
