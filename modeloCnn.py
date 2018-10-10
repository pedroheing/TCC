"""
This module is used to create the CNN model.
"""
import tensorflow as tf


class ConvolutionalNeuralNetwork():

    def __init__(self, num_canais, num_caracteristicas, num_classes):
        self.num_canais = num_canais
        self.num_caracteristicas = num_caracteristicas
        self.num_classes = num_classes

    def __conv2d(self, input, num_filters, kernel_size):
        """
        Return the result of a conv2d operation.

        Args:
            input: input images.
            num_filters: number of filters to be used.
            kernel_size: the kernel size.
        """
        conv = tf.layers.conv2d(inputs=input,
                                filters=num_filters,
                                kernel_size=kernel_size,
                                padding="same",
                                activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer())
        return conv

    def __maxpool2d(self, input):
        """
        Return the max_pooling2d operation.

        Args:
            Input: the result of an conv2d operation.
        """
        return tf.layers.max_pooling2d(inputs=input, pool_size=[2, 2], strides=2, padding="same")

    def conv_layer(self, input, num_filters, kernel_size):
        """
        Return the full operation of a convolutional layer.

        Args:
            input: input images.
            num_filters: number of filters to be used.
            kernel_size: the kernel size.
        """
        conv = self.__conv2d(input, num_filters, kernel_size)
        return self.__maxpool2d(conv)

    def construir_arquitetura(self, images, labels):
        """
        Create the architecture for the CNN and returns its logits.

        Args:
            images: the input images.
        """
        with tf.name_scope("conv_net"):
            tf.summary.image('images', images)
            with tf.name_scope("conv_net_conv1"):
                conv1 = self.conv_layer(images, 256, (5, 5))

            with tf.name_scope("conv_net_conv2"):
                conv2 = self.conv_layer(conv1, 256, (5, 5))

            with tf.name_scope("conv_net_conv3"):
                conv3 = self.conv_layer(conv2, 128, (5, 5))

            with tf.name_scope("conv_net_fc1"):
                fc1 = tf.contrib.layers.flatten(conv3)
                fc1 = tf.layers.dense(fc1, units=328, activation=tf.nn.relu)

            with tf.name_scope("conv_net_fc2"):
                fc2 = tf.layers.dense(fc1, units=192, activation=tf.nn.relu)

            with tf.name_scope("conv_net_out"):
                out = tf.layers.dense(fc2, units=self.num_classes)

            self.logits = out
            self.labels = labels

            return out

    def _loss(self):
        """
        Return the loss of the model.
        """
        with tf.name_scope("custo"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.labels))
            tf.summary.scalar("custo", loss)
            return loss

    def accuracy(self):
        """
        Return the accuracy of the model.
        """
        with tf.name_scope("precisao"):
            correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("precisao", accuracy)
            return accuracy

    def _criar_variavel(self, nome, shape):
        """
        Return the
        :param nome:
        :param shape:
        :return:
        """
        with tf.name_scope("criar_variavel"):
            variavel = tf.get_variable(name=nome, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram(nome, variavel)
            return variavel

    def train(self):
        """
        Train the model and return its accuracy, loss and the summary operations.

        Returns:
            total_loss: the total loss of the model.
            accuracy: the accuracy of the model.
            tain_ops: the operation to train the model.
            summary_ops: the operation to merge all the summaries of the model.
        """
        with tf.name_scope("train"):
            self.global_step = tf.train.get_or_create_global_step()
            total_loss = self._loss()
            accuracy = self.accuracy()
            train_ops = tf.train.AdamOptimizer().minimize(total_loss, global_step=self.global_step)
            summary_ops = tf.summary.merge_all()

            return total_loss, accuracy, train_ops, summary_ops
