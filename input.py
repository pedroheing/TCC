import csv
import os

import numpy as np
import skimage.data
import tensorflow as tf
from skimage import transform
from tensorflow.examples.tutorials.mnist import input_data

with tf.name_scope("read_fashionMNIST"):
    def __read_fashionMNIST(is_training=True):
        """
        Returns:
            Os conjuntos de dados de treino ou teste
        """
        # (trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()
        data = input_data.read_data_sets('data/fashion', one_hot=True)
        train_X = data.train.images.reshape(-1, 28, 28, 1)
        test_X = data.test.images.reshape(-1, 28, 28, 1)
        train_y = data.train.labels
        test_y = data.test.labels

        if is_training:
            return train_X, train_y
        else:
            return test_X, test_y

with tf.name_scope("read_trafficSigns_2"):
    def _read_trafficSigns_2(is_training=True):
        '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

        Arguments: path to the traffic sign data, for example './GTSRB/Training'
        Returns:   list of images, list of corresponding labels'''

        root = "C:\\Users\\pedro\\Desktop\\bases\\"

        if is_training:
            root += "training"
        else:
            root += "test"

        images = []  # images
        labels = []  # corresponding labels
        # loop over all 42 classes
        for c in range(0, 43):
            prefix = root + '/' + format(c, '05d') + '/'  # subdirectory for class
            gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')  # annotations file
            gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
            next(gtReader)  # skip header
            # loop over all images in current annotations file
            for row in gtReader:
                images.append(skimage.data.imread(prefix + row[0]))  # the 1th column is the filename
                labels.append(row[7])  # the 8th column is the label
            gtFile.close()
        return images, labels

with tf.name_scope("read_trafficSigns"):
    def __read_trafficSigns(is_training=True):
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

with tf.name_scope("load_data"):
    def load_data(dataset, is_training=True):
        if dataset == 'fashionMNIST':
            return __read_fashionMNIST(is_training)
        elif dataset == 'traffic_sign':
            data, label = __read_trafficSigns(is_training)
            data = [transform.resize(image, (28, 28)) for image in data]
            data = np.array(data)
            labels = tf.one_hot(label, depth=62, dtype=tf.float32)
            return tf.image.rgb_to_grayscale(data), labels
        else:
            raise Exception('Dataset inválido, por favor confirme o nome do dataset:', dataset)

with tf.name_scope("get_batch_data"):
    def get_batch_data(dataset, batch_size, is_training=True):
        dados, labels = load_data(dataset, is_training)
        dados = tf.cast(dados, tf.float32)
        labels = tf.cast(labels, tf.float32)
        dados = tf.data.Dataset.from_tensor_slices(dados)
        labels = tf.data.Dataset.from_tensor_slices(labels)

        if is_training:
            train_dataset = tf.data.Dataset.zip((dados, labels)).shuffle(5000).repeat().batch(batch_size)
        else:
            train_dataset = tf.data.Dataset.zip((dados, labels)).repeat().batch(batch_size)

        return train_dataset.make_initializable_iterator()
