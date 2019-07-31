"""
This module is used to evaluate the CNN model.
"""
import math
from datetime import datetime

import tensorflow as tf
import math

from shared import Utils
from shared.Config import CFG
from shared.Input import get_batch_data
from datetime import datetime


class EvaluateModel:

    def __init__(self, model):
        num_canais, num_caracteristicas, num_classes = Utils.get_model_hyperparameters()
        self.model = model(num_caracteristicas, num_canais, num_classes, CFG.batch_size)

    def evaluate(self, result_path, restore_path):
        """
        Evaluate the CNN model.
        """
        result_path += CFG.result_dir
<<<<<<< HEAD
        print("result_path: {}".format(result_path))
        restore_path += CFG.restore_dir
        print("restore_path: {}".format(restore_path))
=======
        restore_path += CFG.restore_dir
>>>>>>> 77c759207b6ac061903ac4009d1e04092c07c4ff

        with tf.device('/cpu:0'):
            iterator = get_batch_data(CFG.dataset, self.model.batch_size, is_training=False)
            imagens, labels = iterator.get_next()
<<<<<<< HEAD
        accuracy, error_rate, summary_ops = self.model.evaluate(imagens, labels)

        with tf.Session() as sess:
            total_batch = math.ceil(Utils.get_num_examples_in_dataset(is_training=False) / self.model.batch_size)
            print("Total batch {}".format(total_batch))
=======

        accuracy, error_rate, summary_ops = self.model.evaluate(imagens, labels)

        with tf.Session() as sess:
            total_batch = math.ceil(Utils.get_num_examples_in_dataset(is_training=True) / self.model.batch_size)
>>>>>>> 77c759207b6ac061903ac4009d1e04092c07c4ff
            avg_acc, avg_err = 0., 0.
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)
            summary_writer = tf.summary.FileWriter(result_path, sess.graph)
            saver.restore(sess, restore_path + "/model.ckpt")
            init_time = datetime.now()
            for batch in range(total_batch):
                acc, err, summary, step = sess.run([accuracy, error_rate, summary_ops, self.model.global_step])
                summary_writer.add_summary(summary, step)
                avg_acc += acc / total_batch
                avg_err += err / total_batch
            end_time = datetime.now()
            diff_time = end_time - init_time
            print("accuracy: {:.5f} error rate: {:.5f}".format(avg_acc, avg_err))
<<<<<<< HEAD
            path = Utils.save_results_evaluating([avg_acc, avg_err,  Utils.format_timestamp(init_time),
=======
            path = Utils.save_results_evaluating([avg_acc, avg_err, Utils.format_timestamp(init_time),
>>>>>>> 77c759207b6ac061903ac4009d1e04092c07c4ff
                                                  Utils.format_timestamp(end_time), str(diff_time).split('.', 2)[0]],
                                                 result_path + "/resultado.csv")
            print("CSV salvo em {}".format(path))
