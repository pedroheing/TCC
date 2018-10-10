"""
This module is used to get the models hyperparameters and record the results.
"""
import pandas as pd

from config import CFG


def get_model_hyperparameter(is_training):
    """
    Return the hyperparameters that will be used in the process of the model

    Args:
        is_training: a boolean indicating if the model is in the training phase.

    Returns:
        num_channels: the number of channels of the imagens in the dataset.
        num_characteristics: the number of characteristics in each image of the dataset.
        num_classes: the number of possible classification classes.
        num_examples: the number of imagens in the dataset.
    """
    num_channels = get_num_channels()
    num_characteristics = get_num_characteristics()
    num_classes = get_num_classes()
    num_examples = get_num_examples_in_dataset(is_training)
    return num_channels, num_characteristics, num_classes, num_examples


def get_num_channels():
    """
    Return the number of channels of the imagens in the dataset.
    """
    return 1


def get_num_characteristics():
    """
    Return the number of characteristics in each image of the dataset.
    """
    return 28

def get_num_classes():
    """
    Return the number of possible classification classes.

    Raises:
        invalid_dataset: the dataset's name is invalid.
    """
    if CFG.dataset == "fashionMNIST":
        return 10
    if CFG.dataset == 'traffic_sign':
        return 62
    raise Exception('Invalid dataset name, please confirm the inserted name: ', CFG.dataset)


def get_num_examples_in_dataset(is_training):
    """
    Return the number of examples in the dataset.

    Args:
        is_training: a boolean indicating if the model is in the training phase.

    Raises:
        invalid_dataset: the dataset's name is invalid.
    """
    if CFG.dataset == "fashionMNIST":
        if is_training:
            return 55000
        return 10000
    if CFG.dataset == 'traffic_sign':
        if is_training:
            return 4575
        return 2520
    raise Exception('Invalid dataset name, please confirm the inserted name: ', CFG.dataset)


def get_results_path_cnn(is_training=True):
    """
    Return the path of the result folder for the CNN model.

    Args:
        is_training: a boolean indicating if the model is in the training phase.
    """
    if is_training:
        return CFG.results + '/treinamentoCNN'
    return CFG.results + '/avaliacaoCNN'


def get_results_path_caps(is_training=True):
    """
    Return the path of the result folder for the CapsNet model.

    Args:
        is_training: a boolean indicating if the model is in the training phase.
    """
    if is_training:
        return CFG.results + '/treinamentoCaps'
    return CFG.results + '/avaliacaoCaps'


def save_results_cnn(results, is_training):
    """
    Saves the results of the CNN model in a CSV file.

    Args:
        results: a matrix where each row should have the value of three columns. Number of the
        epoch, the cost and the precision.
        is_training: a boolean indicating if the model is in the training phase.

    Returns:
        path: the path where the CSV file was saved.
    """
    path = get_results_path_cnn(is_training) + "/resultado.csv"
    save_results_general(results, path)
    return path


def save_results_caps(results, is_training):
    """
    Saves the results of the CapsNet model in a CSV file.

    Args:
        results: a matrix where each row should have the value of three columns. Number of the
        epoch, the cost and the precision.
        is_training: a boolean indicating if the model is in the training phase.

    Returns:
        path: the path where the CSV file was saved.
    """
    path = get_results_path_caps(is_training) + "/resultado.csv"
    save_results_general(results, path)
    return path


def save_results_general(results, path):
    """
    Saves the results in a CSB file.

    Args:
        results: a matrix where each row should have the value of three columns. Number of the
        epoch, the cost and the precision.
        path: the path where the CSV file should be created.
    """
    data_frame = pd.DataFrame(results, columns=['Epoch', 'Cost', 'Precision'])
    data_frame.to_csv(path, index=False)
