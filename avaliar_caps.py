"""
This module is used to evaluate the CapsNet model.
"""
import tensorflow as tf

import utils
from config import CFG
from input import get_batch_data
from modeloCapsulas import CapsNet


def avaliar():
    """
    Evaluate the CapsNet model.
    """
    iterator = get_batch_data(CFG.dataset, CFG.batch_size, False)
    imagem, label = iterator.get_next()

    num_canais, num_caracteristicas, num_classes, num_input = utils.get_model_hyperparameter(is_training=False)

    caps_net = CapsNet(num_caracteristicas, num_caracteristicas, num_canais, num_classes)

    accuracy = caps_net.evaluate(imagem, label)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        total_batch = num_input // CFG.batch_size
        avg_acc = 0.
        saver = tf.train.Saver()
        saver.restore(sess, CFG.results + "/model.ckpt")
        for _ in range(total_batch):
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
