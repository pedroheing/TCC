"""
This module is used to execute the training of the CapsNet model.
"""
<<<<<<< HEAD
=======

import math
>>>>>>> 77c759207b6ac061903ac4009d1e04092c07c4ff
from datetime import datetime

import tensorflow as tf
import math

from shared import Utils
from shared.Config import CFG
from shared.Input import get_batch_data


class TrainModel:

    def __init__(self, model):
        num_canais, num_caracteristicas, num_classes = Utils.get_model_hyperparameters()
        self.model = model(num_caracteristicas, num_canais, num_classes, CFG.batch_size)

    def train(self, result_path, restore_path):
        """
        Train the CapsNet model
        """
        result_path += CFG.result_dir
<<<<<<< HEAD
        print("result_path: {}".format(result_path))
        restore_path += CFG.restore_dir
        print("restore_path: {}".format(restore_path))
=======
>>>>>>> 77c759207b6ac061903ac4009d1e04092c07c4ff

        with tf.device('/cpu:0'):
            iterator = get_batch_data(CFG.dataset, self.model.batch_size, is_training=True)
            imagem, label = iterator.get_next()

        loss, accuracy, error_rate, train_ops, summary_ops = self.model.train(imagem, label)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)
            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(result_path, sess.graph)
<<<<<<< HEAD
            if restore_path != 'results/':
                saver.restore(sess, restore_path + "/model.ckpt")
            total_batch = math.ceil(Utils.get_num_examples_in_dataset(is_training=True) / self.model.batch_size)
            print("Total batch {}".format(total_batch))
=======
            total_batch = math.ceil(Utils.get_num_examples_in_dataset(is_training=True) / self.model.batch_size)
>>>>>>> 77c759207b6ac061903ac4009d1e04092c07c4ff
            resultado = []
            for i in range(CFG.epoch):
                avg_cost, avg_acc, avg_err = 0., 0., 0.
                init_time = datetime.now()
                for _ in range(total_batch):
                    _, custo, acc, err, summary, step = sess.run([train_ops, loss, accuracy, error_rate, summary_ops,
                                                                  self.model.global_step])
                    summary_writer.add_summary(summary, step)
                    avg_cost += custo / total_batch
                    avg_acc += acc / total_batch
                    avg_err += err / total_batch
                end_time = datetime.now()
                diff_time = end_time - init_time
                resultado.append([i + 1, avg_cost, avg_acc, avg_err, Utils.format_timestamp(init_time),
                                  Utils.format_timestamp(end_time), str(diff_time).split('.', 2)[0]])
                print(
                    "Epoch " + str(i) + ", Custo= {:.6f}".format(avg_cost) + ", Precisao do treinamento= {:.5f}".format(
                        avg_acc))
            save_path = saver.save(sess, result_path + "/model.ckpt")
            print("Modelo salvo em: %s" % save_path)
            caminho = Utils.save_results_training(resultado, result_path + "/resultado.csv")
            print("CSV salvo em {}".format(caminho))
