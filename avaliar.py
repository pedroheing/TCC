import tensorflow as tf

from config import cfg
from input import load_data
from modelo import ConvolutionalNeuralNetwork


def avaliar():
    with tf.name_scope("entrada/label"):
        x = tf.placeholder(tf.float32, [None, 28, 28, 1])
        y = tf.placeholder(tf.float32, [None, 10])

    imagens, labels = load_data(cfg.dataset, False)

    cnn = ConvolutionalNeuralNetwork()

    logits = cnn.construir_arquitetura(x)

    custo = cnn.custo(logits, y)

    accuracy = cnn.accuracy(logits, y)

    with tf.Session() as sess:
        total_batch = len(imagens) // cfg.batch_size
        avg_acc = 0.
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "Output/model.ckpt")
        for batch in range(total_batch):
            batch_x = imagens[batch * cfg.batch_size:min((batch + 1) * cfg.batch_size, len(imagens))]
            batch_y = labels[batch * cfg.batch_size:min((batch + 1) * cfg.batch_size, len(labels))]

            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            avg_acc += acc / total_batch
        print("accuracy: {:.5f}".format(avg_acc))


def main(argv=None):
    avaliar()


if __name__ == "__main__":
    tf.app.run()
