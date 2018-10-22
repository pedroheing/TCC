"""
    This module is used to get and prepare the images and labels from the dataset.
"""
import os

import numpy as np
import skimage.data
import tensorflow as tf
from skimage import transform
from tensorflow.examples.tutorials.mnist import input_data


def read_fashionMNIST(is_training=True):
    """
    Return the images and labels from the fashionMNIST dataset.

    Args:
        is_training: a boolean indicating if the model is in the training phase.

    Returns:
        train_x: the images of the dataset for training.
        traing_Y: the labels of the dataset for training.

        or

        test_x: the images in the dataset for testing.
        test_Y: the labels of the dataset for training.
    """
    with tf.name_scope("read_fashionMNIST"):
        data = input_data.read_data_sets('data/fashion', one_hot=True, validation_size=0)
        train_x = data.train.images.reshape(-1, 28, 28, 1)
        test_x = data.test.images.reshape(-1, 28, 28, 1)
        train_y = data.train.labels
        test_y = data.test.labels

        if is_training:
            return train_x, train_y

        return test_x, test_y


def read_traffic_signs(is_training=True):
    """
    Return the images and labels from the trafficSigns dataset.

    Args:
        is_training: a boolean indicating if the model is in the training phase.

    Returns:
        images: the images from the dataset without any treatment.
        labels: the labels from the dataset without any treatment.
    """
    with tf.name_scope("read_trafficSigns"):
        data_directory = "data/traffic/"

        if is_training:
            data_directory += "Training"
        elif not is_training:
            data_directory += "Testing"

        directories = [d for d in os.listdir(data_directory)
                       if os.path.isdir(os.path.join(data_directory, d))]
        labels = []
        images = []
        for d in directories:
            label_directory = os.path.join(data_directory, d)
            file_names = [os.path.join(label_directory, f)
                          for f in os.listdir(label_directory)
                          if f.endswith(".ppm")]
            for f in file_names:
                images.append(skimage.data.imread(f))
                labels.append(int(d))
        return images, labels


def load_data(dataset, is_training=True):
    """
    Return the imagens and labels from the dataset.

    Args:
        dataset: name of the dataset.
        is_training: a boolean indicating if the model is in the training phase.

    Returns:
        images: the images of the dataset treated.
        labels: the labels one-hotted fot he dataset treated.

    Raises:
        invalid_dataset: the dataset's name is invalid.
    """
    with tf.name_scope("load_data"):
        if dataset == 'fashionMNIST':
            return read_fashionMNIST(is_training)
        if dataset == 'traffic_sign':
            data, label = read_traffic_signs(is_training)
            data = [transform.resize(image, (28, 28)) for image in data]
            data = np.array(data)
            labels = tf.one_hot(label, depth=62, dtype=tf.float32)
            return tf.image.rgb_to_grayscale(data), labels
        raise Exception('Dataset inválido, por favor confirme o nome do dataset:', dataset)


def get_batch_data(dataset, batch_size, is_training=True):
    """
    Return the initializable iterator to get the batches of the dataset.

    Args:
        dataset: the name of the dataset.
        batch_size: the size of the batch.
        is_training: a boolean indicating if the model is in the training phase.

    Returns:
        An initializable itaretor to get the batches of the dataset.

    Raises:
        invalid_dataset: the dataset's name is invalid.
    """
    with tf.name_scope("get_batch_data"):
        dados, labels = load_data(dataset, is_training)
        dados = tf.cast(dados, tf.float32)
        labels = tf.cast(labels, tf.float32)
        dados = tf.data.Dataset.from_tensor_slices(dados)
        labels = tf.data.Dataset.from_tensor_slices(labels)

        with tf.device('/cpu:0'):
            if is_training:
                dataset = tf.data.Dataset.zip((dados, labels)).shuffle(5000).repeat().batch(batch_size).prefetch(1)
            else:
                dataset = tf.data.Dataset.zip((dados, labels)).repeat().batch(batch_size).prefetch(1)

        return dataset.make_initializable_iterator()
