"""
This module is used to create the CNN model.
"""
import tensorflow as tf

from interface import implements
from model.IModel import IModel


class ConvolutionalNeuralNetwork(implements(IModel)):

    def __init__(self, num_caracteristicas, num_canais, num_classes, batch_size):
        self.channels = num_canais
        self.num_characteristics = num_caracteristicas
        self.num_class = num_classes
        self.batch_size = batch_size

    def _conv2d(self, input, num_filters, kernel_size):
        """
        Return the result of a conv2d operation.

        Args:
            input: input images.
            num_filters: number of filters to be used.
            kernel_size: the kernel size.

        Return:
            conv: the result of as conv2d operation.
        """
        conv = tf.layers.conv2d(inputs=input, filters=num_filters, kernel_size=kernel_size, padding="same",
                                activation=tf.nn.relu)
        return conv

    def _maxpool2d(self, input):
        """
        Return the max_pooling2d operation.

        Args:
            Input: the result of an conv2d operation.
        """
        return tf.layers.max_pooling2d(inputs=input, pool_size=[2, 2], strides=2,  padding="same")

    def _conv_layer(self, input, num_filters, kernel_size):
        """
        Return the full operation of a convolutional layer.

        Args:
            input: input images.
            num_filters: number of filters to be used.
            kernel_size: the kernel size.
        """
        conv = self._conv2d(input, num_filters, kernel_size)
        return self._maxpool2d(conv)

    def _process_images(self, images, is_training):
        """
        Create the architecture for the CNN and returns its logits.

        Args:
            images: the input images.
            is_training: indicate if the model is training, it defines if the dropout will be used
            or not.

        Returns:
            logits: the classification result of the model for the given examples without being normalized.
        """
        with tf.name_scope("conv_net"):
            tf.summary.image('images', images, self.batch_size)

            with tf.name_scope("conv_net_conv1"):
                first_conv = self._conv_layer(images, 256, (5, 5))

            with tf.name_scope("conv_net_conv2"):
                second_conv = self._conv_layer(first_conv, 256, (5, 5))

            with tf.name_scope("conv_net_conv3"):
                third_conv = self._conv_layer(second_conv, 128, (5, 5))

            with tf.name_scope("flatted_conv"):
                flatted_conv = tf.contrib.layers.flatten(third_conv)

            with tf.name_scope("conv_net_fc1"):
                first_fc = tf.layers.dense(flatted_conv, units=328, activation=tf.nn.relu)

            with tf.name_scope("conv_net_fc2"):
                second_fc = tf.layers.dense(first_fc, units=192, activation=tf.nn.relu)

            with tf.name_scope("conv_net_dropout"):
                dropout = tf.layers.dropout(second_fc, rate=0.4, training=is_training)

            with tf.name_scope("conv_net_out"):
                logits = tf.layers.dense(dropout, units=self.num_class)

            return logits

    def _error_rate(self, accuracy):
        with tf.name_scope("error_rate"):
            error_rate = 1. - accuracy
            tf.summary.scalar("error_rate", error_rate)
            return error_rate

    def _loss(self, logits, labels):
        """
        Return the loss of the model.

        Args:
            logits: the result of the images processing.
            labels: the label of the logits.

        Returns:
            loss: the loss of the model.
        """
        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
            tf.summary.scalar("loss", loss)
            return loss

    def _accuracy(self, logits, labels):
        """
        Return the accuracy of the model.

        Args:
            logits: the result of the images processing.
            labels: the label of the logits.

        Returns:
            accuracy: the accuracy of the model, varying between 0 and 1.
        """
        with tf.name_scope("accuracy"):
            pred = tf.nn.softmax(logits)
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", accuracy)
            return accuracy

    def evaluate(self, images, labels):
        """
        Return the accuracy of the model.

        Args:
            images: the images to evaluate the model.
            labels: the labels of the images.

        Returns:
            accuracy: the accuracy of the model for the given examples.
        """
        with tf.name_scope("evaluate"):
            self.global_step = tf.train.get_or_create_global_step()
            logits = self._process_images(images, False)
            accuracy = self._accuracy(logits, labels)
            error_rate = self._error_rate(accuracy)
            summary_ops = tf.summary.merge_all()

            return accuracy, error_rate, summary_ops

    def train(self, images, labels):
        """
        Train the model and return its accuracy, loss and the summary operations.

        Args:
            images: the batch images to train the model.
            labels: the labels of the batch images.

        Returns:
            total_loss: the total loss of the model.
            accuracy: the accuracy of the model.
            tain_ops: the operation to train the model.
            summary_ops: the operation to merge all the summaries of the model.
        """
        with tf.name_scope("train"):
            self.global_step = tf.train.get_or_create_global_step()
            logits = self._process_images(images, True)
            total_loss = self._loss(logits, labels)
            accuracy = self._accuracy(logits, labels)
            error_rate = self._error_rate(accuracy)
            train_ops = tf.train.AdamOptimizer().minimize(total_loss, global_step=self.global_step)
            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)
            summary_ops = tf.summary.merge_all()

            return total_loss, accuracy, error_rate, train_ops, summary_ops
