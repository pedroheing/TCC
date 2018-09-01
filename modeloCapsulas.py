import numpy as np
import tensorflow as tf

import capslayer as cl


class CapsNet(object):
    def __init__(self, height=28, width=28, channels=1, num_label=10):
        '''
        Args:
            height: Integer, the height of inputs.
            width: Integer, the width of inputs.
            channels: Integer, the channels of inputs.
            num_label: Integer, the category number.
        '''
        self.height = height
        self.width = width
        self.channels = channels
        self.num_label = num_label

    def create_network(self, inputs, labels):
        """ Setup capsule network.
        Args:
            inputs: Tensor or array with shape [batch_size, height, width, channels] or [batch_size, height * width * channels].
            labels: Tensor or array with shape [batch_size].
        Returns:
            poses: [batch_size, num_label, 16, 1].
            probs: Tensor with shape [batch_size, num_label], the probability of entity presence.
        """
        self.raw_imgs = inputs
        self.labels_one_hoted = tf.reshape(labels, (-1, 10, 1, 1))
        self.labels = tf.argmax(labels, 1)
        with tf.variable_scope('Conv1_layer'):
            # Conv1, return with shape [batch_size, 20, 20, 256]
            inputs = tf.reshape(self.raw_imgs, shape=[-1, self.height, self.width, self.channels])
            conv1 = tf.layers.conv2d(inputs,
                                     filters=256,
                                     kernel_size=9,
                                     strides=1,
                                     padding='VALID',
                                     activation=tf.nn.relu)

        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps, activation = cl.layers.primaryCaps(conv1,
                                                            filters=32,
                                                            kernel_size=9,
                                                            strides=2,
                                                            out_caps_dims=[8, 1],
                                                            method="norm")

        with tf.variable_scope('DigitCaps_layer'):
            routing_method = "DynamicRouting"
            num_inputs = np.prod(cl.shape(primaryCaps)[1:4])
            primaryCaps = tf.reshape(primaryCaps, shape=[-1, num_inputs, 8, 1])
            activation = tf.reshape(activation, shape=[-1, num_inputs])
            self.poses, self.probs = cl.layers.dense(primaryCaps,
                                                     activation,
                                                     num_outputs=self.num_label,
                                                     out_caps_dims=[16, 1],
                                                     routing_method=routing_method)

        # Decoder structure
        # Reconstructe the inputs with 3 FC layers
        with tf.variable_scope('Decoder'):
            #      labels = tf.one_hot(self.labels, depth=self.num_label, axis=-1, dtype=tf.float32)
            masked_caps = tf.multiply(self.poses, self.labels_one_hoted)
            num_inputs = np.prod(masked_caps.get_shape().as_list()[1:])
            active_caps = tf.reshape(masked_caps, shape=(-1, num_inputs))
            fc1 = tf.layers.dense(active_caps, units=512, activation=tf.nn.relu)
            fc2 = tf.layers.dense(fc1, units=1024, activation=tf.nn.relu)
            num_outputs = self.height * self.width * self.channels
            self.recon_imgs = tf.layers.dense(fc2,
                                              units=num_outputs,
                                              activation=tf.sigmoid)
            recon_imgs = tf.reshape(self.recon_imgs, shape=[-1, self.height, self.width, self.channels])

        with tf.variable_scope('accuracy'):
            logits_idx = tf.to_int32(tf.argmax(cl.softmax(self.probs, axis=1), 1))
            correct_prediction = tf.equal(tf.to_int32(self.labels), logits_idx)
            correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
            self.accuracy = tf.reduce_mean(correct / tf.cast(tf.shape(self.probs)[0], tf.float32))

        return self.poses, self.probs

    def _loss(self):
        with tf.variable_scope("loss"):
            # 1. Margin loss
            margin_loss = cl.losses.margin_loss(logits=self.probs,
                                                labels=tf.squeeze(self.labels_one_hoted, axis=(2, 3)))

            # 2. The reconstruction loss
            orgin = tf.reshape(self.raw_imgs, shape=(-1, self.height * self.width * self.channels))
            squared = tf.square(self.recon_imgs - orgin)
            reconstruction_err = tf.reduce_mean(squared)

            # 3. Total loss
            # The paper uses sum of squared error as reconstruction error, but we
            # have used reduce_mean in `# 2 The reconstruction loss` to calculate
            # mean squared error. In order to keep in line with the paper,the
            # regularization scale should be 0.0005*784=0.392
            total_loss = margin_loss + 0.392 * reconstruction_err

            return total_loss

    def train(self):
        self.global_step = tf.Variable(1, name='global_step', trainable=False)
        total_loss = self._loss()
        optimizer = tf.train.AdamOptimizer()
        train_ops = optimizer.minimize(total_loss, global_step=self.global_step)

        return (total_loss, train_ops)
