"""
This module is used to create and get the value of prompt flags.
"""

import tensorflow as tf

FLAGS = tf.app.flags

FLAGS.DEFINE_integer('batch_size', 128, 'tamanho do batch')
FLAGS.DEFINE_integer('epoch', 5, 'epoch')
FLAGS.DEFINE_string('dataset', 'fashionMNIST', 'O nome do dataset [fashionMNIST, traffic_sign')
FLAGS.DEFINE_string('result_dir', '', 'O diretório para guardar os resultados')
FLAGS.DEFINE_string('restore_dir', '', 'O diretório para restaurar os resultados')

CFG = tf.app.flags.FLAGS
