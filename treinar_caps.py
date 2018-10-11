"""
This module is used to execute the training of the CapsNet model.
"""
import tensorflow as tf

import utils
from config import CFG
from input import get_batch_data
from modeloCapsulas import CapsNet


def train():
    """
    Train the CapsNet model
    """
    iterator = get_batch_data(CFG.dataset, CFG.batch_size, is_training=True)

    imagem, label = iterator.get_next()

    num_canais, num_caracteristicas, num_classes, num_input = utils.get_model_hyperparameter(is_training=True)

    caps_net = CapsNet(num_caracteristicas, num_caracteristicas, num_canais, num_classes)

    loss, accuracy, train_ops, summary_ops = caps_net.train(imagem, label)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(utils.get_results_path_caps(is_training=True), sess.graph)
        total_batch = num_input // CFG.batch_size
        resultado = []
        for i in range(CFG.epoch):
            avg_cost = 0.
            avg_acc = 0.
            for _ in range(total_batch):
                _, custo, acc, summary, step = sess.run([train_ops, loss, accuracy, summary_ops,
                                                         caps_net.global_step])
                summary_writer.add_summary(summary, step)
                avg_cost += custo / total_batch
                avg_acc += acc / total_batch
            resultado.append([i + 1, avg_cost, avg_acc])
            print("Epoch " + str(i) + ", Custo= {:.6f}".format(avg_cost) + ", Precisao do treinamento= {:.5f}".format(
                avg_acc))
        save_path = saver.save(sess, CFG.results + "/model.ckpt")
        print("Modelo salvo em: %s" % save_path)
        caminho = utils.save_results_caps(resultado, is_training=True)
        print("CSV salvo em {}".format(caminho))

def main(argv=None):
    """
    Initiate the training.
    """
    train()

if __name__ == "__main__":
    tf.app.run()
