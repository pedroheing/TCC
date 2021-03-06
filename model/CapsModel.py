"""
This module is used to create the CapsNet model.
"""
import numpy as np
import tensorflow as tf
from interface import implements

import capslayer as cl
from model.IModel import IModel


class CapsNet(implements(IModel)):

    def __init__(self, num_caracteristicas, channels, num_label, batch_size):
        self.height = num_caracteristicas
        self.width = num_caracteristicas
        self.channels = channels
        self.num_label = num_label
        self.batch_size = batch_size

    def _process_images(self, inputs, labels_one_hoted):
        """
        Setup capsule network.
        Args:
            inputs: Tensor or array with shape [batch_size, height, width, channels]
            or [batch_size, height * width * channels].
            labels: Tensor or array with shape [batch_size].
        Returns:
            poses: [batch_size, num_label, 16, 1].
            probs: Tensor with shape [batch_size, num_label], the probability of entity presence.
        """
        with tf.name_scope("caps_net"):
            tf.summary.image("input_image", inputs, self.batch_size, family="input_image")

            with tf.name_scope('conv1_layer'):
                conv1 = tf.layers.conv2d(inputs, filters=256, kernel_size=9, strides=1, padding='valid',
                                         activation=tf.nn.relu)

            with tf.name_scope('primaryCaps_layer'):
                primary_caps, activation = cl.layers.primaryCaps(conv1, filters=32, kernel_size=9, strides=2,
                                                                 out_caps_dims=[8, 1])

            with tf.name_scope('classCaps_layer'):
                # cl.shape(primary_caps) = [<tf.Tensor 'train/ClassCaps_layer/shape_1/strided_slice:0' shape=() dtype=int32>, 6, 6, 32, 8, 1]
                # cl.shape(primary_caps)[1:4] = [6, 6, 32]
                # num_inputs = 1152
                num_inputs = np.prod(cl.shape(primary_caps)[1:4])
                primary_caps = tf.reshape(primary_caps, shape=[-1, num_inputs, 8, 1])
                activation = tf.reshape(activation, shape=[-1, num_inputs])
                # The routing goes three times and return the result
                poses, probs = cl.layers.dense(primary_caps, activation, num_outputs=self.num_label,
                                               out_caps_dims=[16, 1],
                                               routing_method="DynamicRouting")

            with tf.name_scope('decoder'):
                masked_caps = tf.multiply(poses, labels_one_hoted)
                num_inputs = np.prod(masked_caps.get_shape().as_list()[1:])
                active_caps = tf.reshape(masked_caps, shape=(-1, num_inputs))
                fc1 = tf.layers.dense(active_caps, units=512, activation=tf.nn.relu)
                fc2 = tf.layers.dense(fc1, units=1024, activation=tf.nn.relu)
                num_outputs = self.height * self.width * self.channels
                recon_imgs = tf.layers.dense(fc2, units=num_outputs, activation=tf.sigmoid)
                imgs = tf.reshape(recon_imgs, shape=[-1, self.height, self.width, self.channels])

                tf.summary.image("reconstructed_images", imgs, self.batch_size, family="reconstructed_images")

        return poses, probs, recon_imgs

    def _error_rate(self, accuracy):
        with tf.name_scope("error_rate"):
            error_rate = 1. - accuracy
            tf.summary.scalar("error_rate", error_rate)
            return error_rate
    
    def _accuracy(self, probs, labels):
        """
        Return the accuracy of the model.
        Returns:
            accuracy: the accuracy of the model.
        """
        with tf.name_scope('accuracy'):
            pred = tf.nn.softmax(probs)
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", accuracy)
            return accuracy

    def _loss(self, images, labels_one_hotted, probs, recon_imgs):
        """
        Return the total loss of the model.
        Args:
            images: the batch images used for the training.
            labels_one_hotted: the labels of the batch images.
            probs: the probabilities calculated by the model.
            recon_imgs: the reconstructed images of the decoder.
        Returns:
            total_loss: the total loss of the model.
        """
        with tf.name_scope("loss"):
            # Margin loss
            margin_loss = cl.losses.margin_loss(logits=probs, labels=tf.squeeze(labels_one_hotted, axis=(2, 3)))
            tf.summary.scalar("classification_loss", margin_loss)

            # Reconstruction loss
            origin = tf.reshape(images, shape=(-1, self.height * self.width * self.channels))
            squared = tf.square(recon_imgs - origin)
            reconstruction_err = tf.reduce_sum(squared)
            tf.summary.scalar("reconstruction_loss", reconstruction_err)

            # Total loss
            total_loss = margin_loss + 0.0005 * reconstruction_err

            tf.summary.scalar("total_loss", total_loss)
            return total_loss

    def evaluate(self, images, labels):
        """
        Return the accuracy of the model for the given examples.
        Args:
            images: the batch images to train the model.
            labels: the labels of the batch images.
        Returns:
            accuracy: the accuracy of the model for the given examples.
        """
        with tf.name_scope("evaluate"):
            self.global_step = tf.train.get_or_create_global_step()
            labels_one_hotted = tf.reshape(labels, (-1, self.num_label, 1, 1))
            _, probs, _ = self._process_images(images, labels_one_hotted)
            accuracy = self._accuracy(probs, labels)
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

            labels_one_hotted = tf.reshape(labels, (-1, self.num_label, 1, 1))
            poses, probs, reconstructed_imgs = self._process_images(images, labels_one_hotted)
            total_loss = self._loss(images, labels_one_hotted, probs, reconstructed_imgs)
            accuracy = self._accuracy(probs, labels)
            error_rate = self._error_rate(accuracy)
            optimizer = tf.train.AdamOptimizer()
            train_ops = optimizer.minimize(total_loss, global_step=self.global_step)
            summary_ops = tf.summary.merge_all()

            return total_loss, accuracy, error_rate, train_ops, summary_ops