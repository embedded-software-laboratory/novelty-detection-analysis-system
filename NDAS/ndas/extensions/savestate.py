import hickle as hkl
import pandas
import os
from PyQt5.QtWidgets import QMessageBox

from ndas.extensions import annotations, data, plots
from ndas.utils import logger

current_state = None


def init_state_handler(config):
    """
    Initializes the savestate helper

    Parameters
    ----------
    config
    """
    global current_state
    current_state = State()


def get_current_state():
    """
    Returns the current save state
    """
    global current_state
    return current_state


def save_state(state: 'State', loc: str, patientinformation: str):
    """
    Saves the current save state to a save file

    Parameters
    ----------
    state
    loc
    """
    save_data = state.get_save_data(patientinformation)
    _save_object(save_data, loc)
    logger.savestate.debug("Current state saved to file.")


def restore_state(loc: str):
    """
    Restores the state from a save file

    Parameters
    ----------
    loc
    """
    restore_data = _read_object(loc)
    try:
        if restore_data["data"] is None:
            logger.savestate.error("No data to restore: Canceling")
            return False
        else:
            data.restore_from_save(restore_data["data"])

            if restore_data["labels"] is not None:
                annotations.restore_from_save(restore_data["labels"])

            if restore_data["novelties"] is not None:
                plots.restore_from_save(restore_data["novelties"])

            logger.savestate.debug("State restored from file.")
            try:
                return True, restore_data["patientinformation"]
            except KeyError:
                return True, None
    except KeyError:
        if restore_data["'data'"] is None:
            logger.savestate.error("No data to restore: Canceling")
            return False
        else:
            data.restore_from_save(restore_data["'data'"])

            if restore_data["'labels'"] is not None:
                annotations.restore_from_save(list(restore_data["'labels'"].values()))

            if restore_data["'novelties'"] is not None:
                plots.restore_from_save(restore_data["'novelties'"], True)

            logger.savestate.debug("State restored from file.")
            try:
                return True, restore_data["'patientinformation'"]
            except KeyError:
                return True, None

def restore_additional_lables(loc: str, patientinformation: str):
    """
    Restores additional lables and adds them to the current plots
    """
    restore_data = _read_object(loc)

    try:
        if restore_data["patientinformation"] != patientinformation:
            QMessageBox.critical(None, "Patients not matching", "ERROR: Patients do not match, no additional labels were loaded", QMessageBox.Ok)
        else:

            if restore_data["data"] is None:
                logger.savestate.error("No data to restore: Canceling")
                return False
            else:
                if restore_data["labels"] is not None:
                    annotations.restore_additional_labels(restore_data["labels"], os.path.basename(os.path.normpath(loc)))

                logger.savestate.debug("Additional lables restored from file.")
                return True

    except KeyError:
        QMessageBox.critical(None, "Incompatible dataformat", "ERROR: Dataformat incompatible", QMessageBox.Ok)

def _save_object(obj: dict, filename: str):
    """
    Writes the HDF5-file

    Parameters
    ----------
    obj
    filename
    """
    logger.savestate.info("Writing state to %s" % filename)
    hkl.dump(obj, filename, mode='w', compression='lzf',
             shuffle=True, fletcher32=True)  # HDF5


def _read_object(filename: str):
    """
    Reads the HDF5-file

    Parameters
    ----------
    filename
    """
    logger.savestate.info("Loading state from %s" % filename)
    return hkl.load(filename)


class State(object):
    """
    A State consisting of all required information to restore
    """

    def __init__(self, data: dict = None, novelties=None, labels=None):
        self.data = data
        self.novelties = novelties
        self.labels = labels

    def set_data(self, data):
        self.data = data

    def set_novelties(self, novelties):
        self.novelties = novelties

    def set_labels(self, labels):
        self.labels = labels

    def get_save_data(self, patientinformation):
        return {'patientinformation': patientinformation,
                'data': self.data,
                'novelties': self.novelties,
                'labels': self.labels}
