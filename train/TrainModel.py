"""
This module is used to execute the training of the CapsNet model.
"""
from datetime import datetime

import tensorflow as tf

from shared import Utils
from shared.Config import CFG
from shared.Input import get_batch_data


class TrainModel:

    def __init__(self, model):
        num_canais, num_caracteristicas, num_classes = Utils.get_model_hyperparameters()
        self.model = model(num_caracteristicas, num_canais, num_classes, CFG.batch_size)

    def train(self, result_path):
        """
        Train the CapsNet model
        """
        result_path += CFG.result_dir

        with tf.device('/cpu:0'):
            iterator = get_batch_data(CFG.dataset, self.model.batch_size, is_training=True)
            imagem, label = iterator.get_next()

        loss, accuracy, train_ops, summary_ops = self.model.train(imagem, label)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)
            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(result_path, sess.graph)
            total_batch = Utils.get_num_examples_in_dataset(is_training=True) // self.model.batch_size
            resultado = []
            for i in range(CFG.epoch):
                avg_cost = 0.
                avg_acc = 0.
                init_time = datetime.now()
                for e in range(total_batch):
                    _, custo, acc, summary, step = sess.run([train_ops, loss, accuracy, summary_ops,
                                                             self.model.global_step])
                    summary_writer.add_summary(summary, step)
                    avg_cost += custo / total_batch
                    avg_acc += acc / total_batch
                end_time = datetime.now()
                diff_time = end_time - init_time
                resultado.append([i + 1, avg_cost, avg_acc, Utils.format_timestamp(init_time),
                                  Utils.format_timestamp(end_time), str(diff_time)])
                print(
                    "Epoch " + str(i) + ", Custo= {:.6f}".format(avg_cost) + ", Precisao do treinamento= {:.5f}".format(
                        avg_acc))
            save_path = saver.save(sess, result_path + "/model.ckpt")
            print("Modelo salvo em: %s" % save_path)
            caminho = Utils.save_results_training(resultado, result_path + "/resultado.csv")
            print("CSV salvo em {}".format(caminho))
