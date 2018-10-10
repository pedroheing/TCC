"""
This module is used to execute the training of the CNN model.
"""
import tensorflow as tf

import utils
from config import CFG
from input import get_batch_data
from modeloCnn import ConvolutionalNeuralNetwork


def train():
    """
    Train the CNN model
    """
    iterator = get_batch_data(CFG.dataset, CFG.batch_size, is_training=True)
    imagem, label = iterator.get_next()

    num_canais, num_caracteristicas, num_classes, num_input = utils.get_model_hyperparameter(is_training=True)

    cnn = ConvolutionalNeuralNetwork(num_canais, num_caracteristicas, num_classes)

    cnn.construir_arquitetura(imagem, label)

    loss, accuracy, train_ops, summary_ops = cnn.train()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(CFG.results + '/treinamento', sess.graph)
        total_batch = num_input // CFG.batch_size
        resultado = []
        for i in range(CFG.epoch):
            avg_cost = 0.
            avg_acc = 0.
            for batch in range(total_batch):
                _, cost, acc, summary, step = sess.run([train_ops, loss, accuracy, summary_ops,
                                                        cnn.global_step])
                summary_writer.add_summary(summary, step)
                avg_cost += cost / total_batch
                avg_acc += acc / total_batch
            resultado.append([i + 1, avg_cost, avg_acc])
            print("Epoch " + str(i) + ", Custo= " + \
                  "{:.6f}".format(avg_cost) + ", Precisao do treinamento= " + \
                  "{:.5f}".format(avg_acc))
        save_path = saver.save(sess, CFG.results + "/model.ckpt")
        print("Modelo salvo em: %s" % save_path)
        caminho = utils.save_results_cnn(resultado, is_training=True)
        print("CSV salvo em {}".format(caminho))


def main(argv=None):
    """
    Initiate the training.
    """
    train()


if __name__ == "__main__":
    tf.app.run()
