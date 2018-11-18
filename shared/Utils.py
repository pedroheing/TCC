"""
This module is used to get the models hyperparameters and record the results.
"""
import pandas as pd

from shared.Config import CFG


def get_model_hyperparameters():
    """
    Return the hyperparameters that will be used in the process of the model

    Args:
        is_training: a boolean indicating if the model is in the training phase.

    Returns:
        num_channels: the number of channels of the imagens in the dataset.
        num_characteristics: the number of characteristics in each image of the dataset.
        num_classes: the number of possible classification classes.
        num_examples: the number of imagens in the dataset.get_model_hyperparameter
    """
    num_channels = get_num_channels()
    num_characteristics = get_num_characteristics()
    num_classes = get_num_classes()
    return num_channels, num_characteristics, num_classes


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
            return 60000
        return 10000
    if CFG.dataset == 'traffic_sign':
        if is_training:
            return 4575
        return 2520
    raise Exception('Invalid dataset name, please confirm the inserted name: ', CFG.dataset)


def format_timestamp(timestamp):
    return timestamp.strftime('%H:%M:%S')


def save_results_evaluating(result, path):
    data_frame = pd.DataFrame([result], columns=['Accuracy', 'Error rate', 'Initial time', 'End time',
                                                 'Consumed time'])
    data_frame.to_csv(path, index=False)
    return path


def save_results_training(results, path):
    """
    Saves the results in a CSB file.

    Args:
        results: a matrix where each row should have the value of six columns. The number of the
        epoch, the cost, the precision, the initial time, the end time, and the consumed time
        path: the path where the CSV file should be created.
    """
    data_frame = pd.DataFrame(results, columns=['Epoch', 'Cost', 'Precision', 'Error rate', 'Initial time', 'End time',
                                                'Consumed time'])
    data_frame.to_csv(path, index=False)
    return path
