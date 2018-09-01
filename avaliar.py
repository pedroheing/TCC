import tensorflow as tf

import utils
from config import cfg
from input import get_batch_data
from modeloCnn import ConvolutionalNeuralNetwork


def avaliar():
    imagens, labels = get_batch_data(cfg.dataset, cfg.batch_size, cfg.num_threads, is_training=False)

    num_canais, num_caracteristicas, num_classes, num_input = utils.get_hyperparametros_modelo(is_training=False)

    cnn = ConvolutionalNeuralNetwork(num_canais, num_caracteristicas, num_classes)

    logits = cnn.construir_arquitetura(imagens)

    accuracy = cnn.precisao(logits, labels)

    with tf.Session() as sess:
        total_batch = num_input // cfg.batch_size
        avg_acc = 0.
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver.restore(sess, cfg.results + "/model.ckpt")
        for batch in range(total_batch):
            acc = sess.run(accuracy)
            avg_acc += acc / total_batch
        print("accuracy: {:.5f}".format(avg_acc))
        coord.request_stop()
        coord.join(threads)


def main(argv=None):
    avaliar()


if __name__ == "__main__":
    tf.app.run()
