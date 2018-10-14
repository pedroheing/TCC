"""
This module is used to execute the training of the CapsNet model.
"""
import tensorflow as tf

from shared import Utils
from shared.Config import CFG
from shared.Input import get_batch_data


class TrainModel:

    def __init__(self, model):
        num_canais, num_caracteristicas, num_classes, num_input = Utils.get_model_hyperparameter(is_training=True)
        self.model = model(num_canais, num_caracteristicas, num_classes, CFG.batch_size)

    def train(self, result_path):
        """
        Train the CapsNet model
        """
        iterator = get_batch_data(CFG.dataset, CFG.batch_size, is_training=True)
        imagem, label = iterator.get_next()
        loss, accuracy, train_ops, summary_ops = self.model.train(imagem, label)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)
            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(result_path, sess.graph)
            total_batch = self.model.num_input // CFG.batch_size
            resultado = []
            for i in range(CFG.epoch):
                avg_cost = 0.
                avg_acc = 0.
                for _ in range(total_batch):
                    _, custo, acc, summary, step = sess.run([train_ops, loss, accuracy, summary_ops,
                                                             self.model.global_step])
                    summary_writer.add_summary(summary, step)
                    avg_cost += custo / total_batch
                    avg_acc += acc / total_batch
                resultado.append([i + 1, avg_cost, avg_acc])
                print(
                    "Epoch " + str(i) + ", Custo= {:.6f}".format(avg_cost) + ", Precisao do treinamento= {:.5f}".format(
                        avg_acc))
            save_path = saver.save(sess, CFG.results + "/model.ckpt")
            print("Modelo salvo em: %s" % save_path)
            caminho = result_path + "/resultado.csv"
            print("CSV salvo em {}".format(caminho))