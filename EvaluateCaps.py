"""
This module is used to evaluate the CapsNet model.
"""
import tensorflow as tf

from evaluate.EvaluateModel import EvaluateModel
from model.CapsModel import CapsNet


def evaluate():
    """
    Evaluate the CapsNet model.
    """
<<<<<<< HEAD
    result_path = 'results/'

    restore_path = 'results/'
=======
    result_path = 'results/evaluationCaps'

    restore_path = 'results/trainingCaps'
>>>>>>> 77c759207b6ac061903ac4009d1e04092c07c4ff

    eval = EvaluateModel(CapsNet)

    eval.evaluate(result_path, restore_path)


def main(argv=None):
    """
    Initiate the evaluation.
    """
    evaluate()


if __name__ == "__main__":
    tf.app.run()
