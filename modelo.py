import tensorflow as tf

from config import cfg


class ConvolutionalNeuralNetwork():

    def __init__(self):
        self._definir_hyperparametros()

    def _definir_hyperparametros(self):
        self.num_canais = 1
        self.num_caracteristicas = 28
        if cfg.dataset == "fashionMNIST":
            self.num_classes = 10
        elif cfg.dataset == "traffic_sign":
            self.num_classes = 62
        else:
            raise Exception('Dataset inválido, por favor confirme o nome do dataset:', cfg.dataset)

    def conv2d(self, input, pesos, bias, strides=1):
        x = tf.nn.conv2d(input, pesos, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, bias)
        return tf.nn.relu(x)

    def maxpool2d(self, input, k=2):
        return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    with tf.name_scope("conv_net"):
        def construir_arquitetura(self, imagens):
            with tf.name_scope("conv_net_conv1"):
                peso1 = self._criar_variavel("w0", [5, 5, 1, 256])
                bias1 = self._criar_variavel("b0", [256])
                conv1 = self.conv2d(imagens, peso1, bias1)
                conv1 = self.maxpool2d(conv1, k=2)
            #  conv1 = tf.contrib.layers.conv2d(imagens, num_outputs=256, kernel_size=5, stride=1, padding="SAME", biases_initializer=tf.contrib.layers.xavier_initializer())
            #  conv1 = tf.contrib.layers.dropout(conv1)
            # assert conv1.get_shape() == [cfg.batch_size, self.num_caracteristicas, self.num_caracteristicas, 256]
            #  pool1 = tf.contrib.layers.max_pool2d(conv1, kernel_size=2, stride=1, padding="SAME")

            with tf.name_scope("conv_net_conv2"):
                peso2 = self._criar_variavel("w1", [5, 5, 256, 256])
                bias2 = self._criar_variavel("b1", [256])
                conv2 = self.conv2d(conv1, peso2, bias2)
                conv2 = self.maxpool2d(conv2, k=2)
            # conv2 = tf.contrib.layers.conv2d(pool1, num_outputs=256, kernel_size=5, stride=1, padding="SAME", biases_initializer=tf.contrib.layers.xavier_initializer())
            # conv2 = tf.contrib.layers.dropout(conv2)
            # assert conv2.get_shape() == [cfg.batch_size, self.num_caracteristicas, self.num_caracteristicas, 256]
            # pool2 = tf.contrib.layers.max_pool2d(conv2, kernel_size=2, stride=1, padding="SAME")

            with tf.name_scope("conv_net_conv3"):
                peso3 = self._criar_variavel("w2", [5, 5, 256, 128])
                bias3 = self._criar_variavel("b2", [128])
                conv3 = self.conv2d(conv2, peso3, bias3)
                conv3 = self.maxpool2d(conv3, k=2)
            #   conv3 = tf.contrib.layers.conv2d(pool2, num_outputs=128, kernel_size=5, stride=1, padding="SAME", biases_initializer=tf.contrib.layers.xavier_initializer())
            #   conv3 = tf.contrib.layers.dropout(conv3)
            #   assert conv3.get_shape() == [cfg.batch_size, self.num_caracteristicas, self.num_caracteristicas, 128]
            #   pool3 = tf.contrib.layers.max_pool2d(conv3, kernel_size=2, stride=1, padding="SAME")

            with tf.name_scope("conv_net_fc1"):
                peso4 = self._criar_variavel("w3", [4 * 4 * 128, 328])
                bias4 = self._criar_variavel("b3", [328])
                fc1 = tf.contrib.layers.flatten(conv3)
                fc1 = tf.add(tf.matmul(fc1, peso4), bias4)
                fc1 = tf.nn.relu(fc1)
            #  fc1 = tf.contrib.layers.fully_connected(fc1, num_outputs=328, biases_initializer=tf.contrib.layers.xavier_initializer())
            #  fc1 = tf.contrib.layers.dropout(fc1)

            with tf.name_scope("conv_net_fc2"):
                peso5 = self._criar_variavel("w4", [328, 192])
                bias5 = self._criar_variavel("b4", [192])
                fc2 = tf.add(tf.matmul(fc1, peso5), bias5)
                fc2 = tf.nn.relu(fc2)
            # fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=192, biases_initializer=tf.contrib.layers.xavier_initializer())
            # fc2 = tf.contrib.layers.dropout(fc2)

            with tf.name_scope("conv_net_out"):
                peso6 = self._criar_variavel("w5", [192, self.num_classes])
                bias6 = self._criar_variavel("b5", [self.num_classes])
                out = tf.add(tf.matmul(fc2, peso6), bias6)
            #  fc3 = tf.contrib.layers.fully_connected(fc2, activation_fn=None, num_outputs=self.num_classes, biases_initializer=tf.contrib.layers.xavier_initializer())
            #  fc3 = tf.contrib.layers.dropout(fc3)

            return out

    with tf.name_scope("custo"):
        def custo(self, logits, labels):
            #  cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels, name='cross_entropy')
            #  custo = tf.reduce_mean(cross_entropy, name="reduce_mean")
            custo = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
            tf.summary.scalar("custo", custo)
            return custo

    with tf.name_scope("treino"):
        def treinar(self, custo):
            return tf.train.AdamOptimizer().minimize(custo)

    with tf.name_scope("accuracy"):
        def accuracy(self, logits, labels):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", accuracy)
            return accuracy

    def _criar_variavel(self, nome, shape):
        variavel = tf.get_variable(name=nome, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        tf.summary.histogram(nome, variavel)
        return variavel
