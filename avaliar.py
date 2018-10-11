"""
This module is used to evaluate the CNN model.
"""
import tensorflow as tf

import utils
from config import CFG
from input import get_batch_data
from modeloCnn import ConvolutionalNeuralNetwork


def avaliar():
    """
    Evaluate the CNN model.
    """
    iterator = get_batch_data(CFG.dataset, CFG.batch_size, is_training=False)
    imagens, labels = iterator.get_next()

    num_canais, num_caracteristicas, num_classes, num_input = utils.get_model_hyperparameter(is_training=False)

    cnn = ConvolutionalNeuralNetwork(num_canais, num_caracteristicas, num_classes)

    accuracy = cnn.evaluate(imagens, labels)

    with tf.Session() as sess:
        total_batch = num_input // CFG.batch_size
        avg_acc = 0.
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        saver.restore(sess, CFG.results + "/model.ckpt")
        for batch in range(total_batch):
            acc = sess.run(accuracy)
            avg_acc += acc / total_batch
        print("accuracy: {:.5f}".format(avg_acc))

def main(argv=None):
    """
    Initiate the evaluation.
    """
    avaliar()


if __name__ == "__main__":
    tf.app.run()
