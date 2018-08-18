import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_integer('batch_size', 128, 'tamanho do batch')
flags.DEFINE_integer('epoch', 50, 'epoch')

flags.DEFINE_string('dataset', 'fashionMNIST', 'O nome do dataset [fashionMNIST, traffic_sign')
flags.DEFINE_boolean('is_training', True, 'Define se o modelo deve ser treinado ou avaliado')
flags.DEFINE_integer('num_threads', 8, 'Número de threads para gerenciar os exemplos')
flags.DEFINE_string('logdir', 'logdir', 'Diretório de logs')
flags.DEFINE_string('results', 'results', 'caminho para armazenar os resultados')

cfg = tf.app.flags.FLAGS
