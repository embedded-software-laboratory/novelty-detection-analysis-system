import pandas as pd

from ndas.extensions import data
from ndas.utils import logger
from ndas.algorithms import *

_available_algorithms = []
_active_instance = None
_detected_novelties = {}


def init_algorithms(config):
    """
    Initialize the algorithms helper

    Parameters
    ----------
    config

    """
    classes = ([subclasses.__name__ for subclasses in BaseDetector.__subclasses__()])
    for klass in classes:
        logger.init.debug("Loaded algorithm: %s" % klass)
        _available_algorithms.append(klass)
    set_specific_algorithm_instance(get_available_algorithms()[0], data.get_dataframe())


def get_available_algorithms():
    """
    Returns the currently loaded algorithms
    """
    return _available_algorithms


def get_instance():
    """
    Returns the active instance
    """
    global _active_instance
    return _active_instance


def set_specific_algorithm_instance(klass: str, datasets, additional_args=None):
    """
    Sets a new algorithm as active instance

    Parameters
    ----------
    klass
    datasets
    additional_args
    """
    if additional_args is None:
        additional_args = {}
    global _active_instance
    if klass not in _available_algorithms:
        raise Exception("Unknown algorithm")
    else:
        obj = globals()[klass]
        for arg in additional_args:
            logger.algorithms.debug("Received additional argument %s = %s", arg, additional_args[arg])
        _active_instance = obj(datasets, **additional_args)


def get_specific_algorithm_instance(klass: str, datasets, additional_args=None):
    """
    Gets a specific algorithm instance

    Parameters
    ----------
    klass
    datasets
    additional_args
    """
    if additional_args is None:
        additional_args = {}
    if klass not in _available_algorithms:
        raise Exception("Unknown algorithm")
    else:
        obj = globals()[klass]
        for arg in additional_args:
            logger.algorithms.debug("Received additional argument %s = %s", arg, additional_args[arg])
        return obj(datasets, **additional_args)


def get_algorithm_required_arguments(klass):
    """
    Returns the required arguments of an algorithm

    Parameters
    ----------
    klass
    """
    if klass not in _available_algorithms:
        raise Exception("Unknown algorithm")
    else:
        obj = globals()[klass]
        return obj(pd.DataFrame).get_required_arguments()


def set_detected_novelties(plot_name, novelties):
    """
    Sets the detected novelties for a plot

    Parameters
    ----------
    plot_name
    novelties
    """
    global _detected_novelties
    _detected_novelties[plot_name] = novelties


def get_detected_novelties(plot_name):
    """
    Returns the detected novelties for a plot

    Parameters
    ----------
    plot_name
    """
    global _detected_novelties
    return _detected_novelties[plot_name]


def restore_from_save(data: list):
    """
    Loads algorithm results from save file

    Parameters
    ----------
    data
    """
    global _detected_novelties
    _detected_novelties = data


def format_for_save():
    """
    Formats algorithm results for save file
    """
    global _detected_novelties
    return _detected_novelties
