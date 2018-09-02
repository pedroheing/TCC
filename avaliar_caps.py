import tensorflow as tf

import utils
from config import cfg
from input import get_batch_data
from modeloCapsulas import CapsNet


def avaliar():
    iterator = get_batch_data(cfg.dataset, cfg.batch_size, cfg.num_threads, is_training=True)
    imagem, label = iterator.get_next()

    num_canais, num_caracteristicas, num_classes, num_input = utils.get_hyperparametros_modelo(is_training=False)

    capsNet = CapsNet(num_caracteristicas, num_caracteristicas, num_canais, num_classes)

    capsNet.create_network(imagem, label)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_batch = num_input // cfg.batch_size
        avg_acc = 0.
        saver = tf.train.Saver()
        saver.restore(sess, cfg.results + "/model.ckpt")
        for batch in range(total_batch):
            acc = sess.run(capsNet.accuracy)
            avg_acc += acc / total_batch
        print("accuracy: {:.5f}".format(avg_acc))


def main(argv=None):
    avaliar()


if __name__ == "__main__":
    tf.app.run()
