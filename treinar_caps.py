import tensorflow as tf

from config import cfg
from input import get_batch_data
from modeloCapsulas import CapsNet


def train():
    iterator = get_batch_data(cfg.dataset, cfg.batch_size, is_training=True)

    imagem, label = iterator.get_next()

    capsNet = CapsNet()

    capsNet.create_network(imagem, label)

    loss, train_ops = capsNet.train()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        total_batch = 55000 // cfg.batch_size
        for i in range(cfg.epoch):
            avg_cost = 0.
            avg_acc = 0.
            for batch in range(total_batch):
                _, custo, acc = sess.run([train_ops, loss, capsNet.accuracy])
                avg_cost += custo / total_batch
                avg_acc += acc / total_batch
            print("Epoch " + str(i) + ", Custo= {:.6f}".format(avg_cost) + ", Precisao do treinamento= {:.5f}".format(
                avg_acc))


def main(argv=None):
    # cifar10.maybe_download_and_extract()
    # if tf.gfile.Exists(FLAGS.train_dir):
    #   tf.gfile.DeleteRecursively(FLAGS.train_dir)
    # tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == "__main__":
    tf.app.run()
