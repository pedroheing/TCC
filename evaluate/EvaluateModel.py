"""
This module is used to evaluate the CNN model.
"""
import tensorflow as tf

from shared import Utils
from shared.Config import CFG
from shared.Input import get_batch_data


class EvaluateModel:

    def __init__(self, model):
        num_canais, num_caracteristicas, num_classes = Utils.get_model_hyperparameters()
        self.model = model(num_caracteristicas, num_canais, num_classes, CFG.batch_size)

    def evaluate(self, result_path, restore_path):
        """
        Evaluate the CNN model.
        """
        iterator = get_batch_data(CFG.dataset, self.model.batch_size, is_training=False)
        imagens, labels = iterator.get_next()
        accuracy, summary_ops = self.model.evaluate(imagens, labels)

        with tf.Session() as sess:
            total_batch = Utils.get_num_examples_in_dataset(is_training=True) // self.model.batch_size
            avg_acc = 0.
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)
            summary_writer = tf.summary.FileWriter(result_path, sess.graph)
            saver.restore(sess, restore_path + "/model.ckpt")
            for batch in range(total_batch):
                acc, summary, step = sess.run([accuracy, summary_ops, self.model.global_step])
                summary_writer.add_summary(summary, step)
                avg_acc += acc / total_batch
            print("accuracy: {:.5f}".format(avg_acc))
            path = Utils.save_results_evaluating([avg_acc], result_path + "/resultado.csv")
            print("CSV salvo em {}".format(path))
