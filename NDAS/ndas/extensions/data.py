import pandas as pd
from ndas.importer import *
from ndas.utils import logger

_dataframe = None
_imputed_dataframe = pd.DataFrame()
_dataframe_labels = None
_available_importer = []
_active_instance = None
data_slider_start = 0
data_slider_end = 0
truncate_size = 0


def init_data_importer(config):
    """
    Initializes the data importer helper
    Parameters
    ----------
    config
    """
    global _available_importer, truncate_size

    if config["auto_truncate_size"]:
        truncate_size = int(config["auto_truncate_size"])

    classes = ([subclasses.__name__ for subclasses in BaseImporter.__subclasses__()])
    for klass in classes:
        logger.init.debug("Loaded importer: %s" % klass)
        _available_importer.append(klass)


def get_importer(klass, files):
    """
    Returns the selected importer instance
    Parameters
    ----------
    klass
    files
    """
    global _active_instance
    if klass not in _available_importer:
        raise Exception("Unknown importer")
    else:
        obj = globals()[klass]
        _active_instance = obj(files)
        return _active_instance


def set_slice(start, end):
    """
    Slices the data from start to end
    Parameters
    ----------
    start
    end
    """
    global data_slider_start, data_slider_end

    start = int(start)
    end = int(end)

    if end > get_dataframe_length():
        end = get_dataframe_length()

    if start > get_dataframe_length():
        start = get_dataframe_length()

    if end < start:
        end = start

    if end < 0:
        end = 0

    if start < 0:
        start = 0

    logger.data.debug("Slicing data: %i-%i (Available: %i-%i)" % (start, end, 0, get_dataframe_length()))
    data_slider_start = start
    data_slider_end = end


def set_instance(klass, files):
    """
    Sets the current importer instance
    Parameters
    ----------
    klass
    files
    """
    global _active_instance
    if klass not in _available_importer:
        raise Exception("Unknown importer")
    else:
        obj = globals()[klass]
        _active_instance = obj(files)


def get_instance():
    """
    Returns the current instance
    """
    global _active_instance
    return _active_instance


def set_dataframe(df, labels):
    """
    Sets the loaded data
    Parameters
    ----------
    df
    labels
    """
    global _dataframe, data_slider_end, _dataframe_labels, truncate_size
    _dataframe = df

    if get_dataframe_length() > truncate_size:
        logger.data.info(
            "Dataframe length is exceeding " + str(truncate_size) + ". The dataframe set is truncated to " + str(
                truncate_size) + " data points and can be enlarged in the slicer settings.")
        data_slider_end = truncate_size
    else:
        data_slider_end = get_dataframe_length()

    if len(labels) == len(df.columns):
        _dataframe_labels = labels
        logger.data.debug("Received custom labels from dataframe import.")
    else:
        _dataframe_labels = df.columns.tolist()
        logger.data.debug("Using default dataframe columns as labels.")


def get_dataframe():
    """
    Returns the current data
    """
    global _dataframe, data_slider_start, data_slider_end
    if _dataframe is not None:
        return _dataframe.iloc[data_slider_start:data_slider_end]
    else:
        return None


def get_full_dataframe():
    """
    Returns the full data

    Returns
    -------

    """
    global _dataframe
    if _dataframe is not None:
        return _dataframe
    else:
        return None


def set_imputed_dataframe(df):
    """
    Sets the loaded data

    Parameters
    ----------
    df

    Returns
    -------

    """
    global _imputed_dataframe
    _imputed_dataframe = df


def reset_imputed_dataframe():
    """
    Resets the imputed dataframe if a new regular dataframe is set
    """
    global _imputed_dataframe
    _imputed_dataframe = None


def get_imputed_dataframe():
    """
    Returns the full data

    Returns
    -------

    """
    global _imputed_dataframe
    if _imputed_dataframe is not None and not _imputed_dataframe.empty:
        return _imputed_dataframe
    else:
        return None


def get_dataframe_labels():
    """
    Returns the column labels
    """
    global _dataframe, _dataframe_labels
    if _dataframe is not None:
        return _dataframe_labels
    return None


def get_dataframe_columns():
    """
    Returns the column identifier
    """
    global _dataframe
    if _dataframe is not None:
        return _dataframe.columns.tolist()
    return False


def get_dataframe_index_column():
    """
    Returns the first column identifier
    """
    global _dataframe
    if _dataframe is not None:
        return _dataframe.columns.tolist()[0]
    return False


def get_dataframe_length():
    """
    Returns the length of the data
    """
    global _dataframe
    if _dataframe is not None:
        return len(_dataframe.index)
    return False


def get_dataframe_current_length():
    """
    Returns the length of the data after slicing
    """
    global _dataframe
    if _dataframe is not None:
        return len(_dataframe.iloc[data_slider_start:data_slider_end].index)
    return False


def format_for_save():
    """
    Formats the data for the save file
    """
    global _dataframe, _imputed_dataframe, _dataframe_labels, data_slider_start, data_slider_end
    return {'dataframe': _dataframe.to_numpy(),
            'dataframe_labels': _dataframe_labels,
            'data_slider_start': data_slider_start,
            'data_slider_end': data_slider_end, 
            'imputed_dataframe': _imputed_dataframe.to_numpy()}


def restore_from_save(data: dict):
    """
    Restores the data from a save file
    Parameters
    ----------
    data
    """
    global _dataframe, _imputed_dataframe, _dataframe_labels, data_slider_start, data_slider_end
    _dataframe_labels = data['dataframe_labels']
    _dataframe = pd.DataFrame.from_records(data['dataframe'], columns=_dataframe_labels)
    data_slider_start = data['data_slider_start']
    data_slider_end = data['data_slider_end']
    _imputed_dataframe = pd.DataFrame.from_records(data['imputed_dataframe'], columns=_dataframe_labels)
