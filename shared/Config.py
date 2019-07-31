"""
This module is used to create and get the value of prompt flags.
"""

import tensorflow as tf

FLAGS = tf.app.flags

FLAGS.DEFINE_integer('batch_size', 128, 'tamanho do batch')
FLAGS.DEFINE_integer('epoch', 5, 'epoch')
<<<<<<< HEAD
FLAGS.DEFINE_string('dataset', 'traffic_sign', 'O nome do dataset [fashionMNIST, traffic_sign')
=======
FLAGS.DEFINE_string('dataset', 'fashionMNIST', 'O nome do dataset [fashionMNIST, traffic_sign')
>>>>>>> 77c759207b6ac061903ac4009d1e04092c07c4ff
FLAGS.DEFINE_string('result_dir', '', 'O diretório para guardar os resultados')
FLAGS.DEFINE_string('restore_dir', '', 'O diretório para restaurar os resultados')

CFG = tf.app.flags.FLAGS
