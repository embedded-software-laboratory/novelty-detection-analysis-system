import hickle as hkl
import pickle
import numpy
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
    print("---------------------------------")
    print(type(save_data))
    print(save_data)
    print("---------------------------------")
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
    except KeyError: #for some older ndas files, there are differences in the exact data structure. To preserve backward compatibility with the older files, this case is catched and handled here
        try:
            if restore_data["'data'"] is None:
                logger.savestate.error("No data to restore: Canceling")
                return False
            else:
                dataframe = []
                if type(restore_data["'data'"]["'dataframe'"]) == hkl.lookup.RecoveredDataset:
                    for set in list(restore_data["'data'"]["'dataframe'"].astype(list)):
                        dataframe.append(list(set.astype(list)))         
                else:
                    dataframe = restore_data["'data'"]["'dataframe'"][0]
                dataframe_labels_list = []
                for label in list(restore_data["'data'"]["'dataframe_labels'"].astype(list)):
                    dataframe_labels_list.append(label.decode("utf-8"))
                new_data_dict = {'dataframe': dataframe, 'dataframe_labels': dataframe_labels_list, 'data_slider_start': int(restore_data["'data'"]["'data_slider_start'"].astype(int)), 'data_slider_end': int(restore_data["'data'"]["'data_slider_end'"].astype(int)), 'imputed_dataframe': numpy.ndarray(shape=(0, 0), dtype=numpy.float64), 'mask_dataframe': numpy.ndarray(shape=(0, 0), dtype=numpy.float64)}
                data.restore_from_save(new_data_dict)

                if restore_data["'labels'"] is not None:
                    new_label_array = []
                    for key in restore_data["'labels'"]:
                        try:
                            new_label_array.append({'value': float(restore_data["'labels'"][key]["'value'"].astype(float)), 
                                                    'x':  float(restore_data["'labels'"][key]["'x'"].astype(float)), 
                                                    'index':  restore_data["'labels'"][key]["'index'"][0], 
                                                    'label':  str(restore_data["'labels'"][key]["'label'"].astype(str)), 
                                                    'plot_name':  str(restore_data["'labels'"][key]["'plot_name'"].astype(str))})
                        except AttributeError:
                            new_label_array.append({'value': restore_data["'labels'"][key]["'value'"][0], 'x':  restore_data["'labels'"][key]["'x'"][0], 'index':  restore_data["'labels'"][key]["'index'"][0], 'label':  str(restore_data["'labels'"][key]["'label'"].astype(str)), 'plot_name':  str(restore_data["'labels'"][key]["'plot_name'"].astype(str))})
                        except IndexError:
                            new_label_array.append({'value': float(restore_data["'labels'"][key]["'value'"].astype(float)), 
                                                    'x':  float(restore_data["'labels'"][key]["'x'"].astype(float)), 
                                                    'index':  int(restore_data["'labels'"][key]["'index'"].astype(int)), 
                                                    'label':  str(restore_data["'labels'"][key]["'label'"].astype(str)), 
                                                    'plot_name':  str(restore_data["'labels'"][key]["'plot_name'"].astype(str))})
                    annotations.restore_from_save(new_label_array)

                if restore_data["'novelties'"] is not None:
                    plots.restore_from_save(restore_data["'novelties'"])

                logger.savestate.debug("State restored from file.")
                try:
                    return True, restore_data["'patientinformation'"]
                except KeyError:
                    return True, None
        except KeyError:
            if restore_data['"data"'] is None:
                logger.savestate.error("No data to restore: Canceling")
                return False
            else:
                dataframe = []
                if type(restore_data['"data"']['"dataframe"']) == hkl.lookup.RecoveredDataset:
                    for set in list(restore_data['"data"']['"dataframe"'].astype(list)):
                        dataframe.append(list(set.astype(list)))         
                else:
                    dataframe = restore_data['"data"']['"dataframe"'][0]
                dataframe_labels_list = []
                for key in restore_data['"data"']['"dataframe_labels"']:
                    temp = list(restore_data['"data"']['"dataframe_labels"'][key][0].astype(list))
                    temp2 = []
                    for label in temp:
                        temp2.append(label.decode("utf-8", "replace"))
                    dataframe_labels_list.append(''.join(temp2))
                new_data_dict = {'dataframe': dataframe, 'dataframe_labels': dataframe_labels_list, 'data_slider_start': int(restore_data['"data"']['"data_slider_start"'].astype(int)), 'data_slider_end': int(restore_data['"data"']['"data_slider_end"'].astype(int)), 'imputed_dataframe': numpy.ndarray(shape=(0, 0), dtype=numpy.float64), 'mask_dataframe': numpy.ndarray(shape=(0, 0), dtype=numpy.float64)}
                data.restore_from_save(new_data_dict)

                if restore_data['"labels"'] is not None:
                    new_label_array = []
                    for key in restore_data['"labels"']:
                        try:
                            new_label_array.append({'value': float(restore_data['"labels"'][key]['"value"'].astype(float)), 
                                                    'x':  float(restore_data['"labels"'][key]['"x"'].astype(float)), 
                                                    'index':  restore_data['"labels"'][key]['"index"'][0], 
                                                    'label':  str(restore_data['"labels"'][key]['"label"'].astype(str)), 
                                                    'plot_name':  str(restore_data['"labels"'][key]['"plot_name"'].astype(str))})
                        except AttributeError:
                            new_label_array.append({'value': restore_data['"labels"'][key]['"value"'][0], 'x':  restore_data['"labels"'][key]['"x"'][0], 'index':  restore_data['"labels"'][key]['"index"'][0], 'label':  str(restore_data['"labels"'][key]['"label"'].astype(str)), 'plot_name':  str(restore_data['"labels"'][key]['"plot_name"'].astype(str))})
                        except IndexError:
                            new_label_array.append({'value': float(restore_data['"labels"'][key]['"value"'].astype(float)), 
                                                    'x':  float(restore_data['"labels"'][key]['"x"'].astype(float)), 
                                                    'index':  int(restore_data['"labels"'][key]['"index"'].astype(int)), 
                                                    'label':  str(restore_data['"labels"'][key]['"label"'].astype(str)), 
                                                    'plot_name':  str(restore_data['"labels"'][key]['"plot_name"'].astype(str))})
                    annotations.restore_from_save(new_label_array)

                if restore_data['"novelties"'] is not None:
                    plots.restore_from_save(restore_data['"novelties"'])

                logger.savestate.debug("State restored from file.")
                try:
                    return True, restore_data['"patientinformation"']
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
    hkl.dump(obj, filename, mode='wb', compression='lzf',
             shuffle=True, fletcher32=True)  # HDF5
    file = open(filename, mode="wb")
    pickle.dump(obj, file)
    file.close()


def _read_object(filename: str):
    """
    Reads the HDF5-file

    Parameters
    ----------
    filename
    """
    logger.savestate.info("Loading state from %s" % filename)
    # Because of compatibility issues that existed between files saved by the compiled version and files by the script version, we switched from hickle to pickle to save the data.
    # So, we first try to load the file using pickle. But, to preserve backward compatibility with older files, we use hickle if the loading process fails with pickle. 
    try:
        file = open(filename, mode="rb")
        load = pickle.load(file)
        file.close()
        return load
    except pickle.UnpicklingError:
        return(hkl.load(filename))


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
