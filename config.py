"""
This module is used to create and get the value of prompt flags.
"""

import tensorflow as tf

FLAGS = tf.app.flags

FLAGS.DEFINE_integer('batch_size', 128, 'tamanho do batch')
FLAGS.DEFINE_integer('epoch', 50, 'epoch')

FLAGS.DEFINE_string('dataset', 'fashionMNIST', 'O nome do dataset [fashionMNIST, traffic_sign')
FLAGS.DEFINE_boolean('is_training', True, 'Define se o modelo deve ser treinado ou avaliado')
FLAGS.DEFINE_string('logdir', 'logdir', 'Diretório de logs')
FLAGS.DEFINE_string('results', 'results', 'caminho para armazenar os resultados')

CFG = tf.app.flags.FLAGS
