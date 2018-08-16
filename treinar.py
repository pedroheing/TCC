import tensorflow as tf

from config import cfg
from input import get_batch_data
from modelo import ConvolutionalNeuralNetwork


def train():
    imagens, labels = get_batch_data(cfg.dataset, cfg.batch_size, cfg.num_threads)

    cnn = ConvolutionalNeuralNetwork()

    logits = cnn.construir_arquitetura(imagens)

    custo = cnn.custo(logits, labels)

    treinamento = cnn.treinar(custo)

    accuracy = cnn.accuracy(logits, labels)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        summary_writer = tf.summary.FileWriter('Output/treinamento', sess.graph)
        total_batch = 55000 // cfg.batch_size
        print("epoch " + str(cfg.epoch))
        count = 0
        for i in range(cfg.epoch):
            avg_cost = 0.
            avg_acc = 0.
            for batch in range(total_batch):
                count += 1
                merge = tf.summary.merge_all()
                sumario, _ = sess.run([merge, treinamento])
                summary_writer.add_summary(sumario, count)
                loss, acc = sess.run([custo, accuracy])
                avg_cost += loss / total_batch
                avg_acc += acc / total_batch
            print("Iter " + str(i) + ", Loss= " + \
                  "{:.6f}".format(avg_cost) + ", Training Accuracy= " + \
                  "{:.5f}".format(avg_acc))
        save_path = saver.save(sess, "Output/model.ckpt")
        print("Model saved in path: %s" % save_path)
        coord.request_stop()
        coord.join(threads)


def main(argv=None):  # pylint: disable=unused-argument
    # cifar10.maybe_download_and_extract()
    # if tf.gfile.Exists(FLAGS.train_dir):
    #   tf.gfile.DeleteRecursively(FLAGS.train_dir)
    # tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == "__main__":
    tf.app.run()
