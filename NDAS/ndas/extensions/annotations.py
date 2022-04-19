from ndas.extensions import data, savestate
from ndas.utils import logger
import copy

_current_point_selection = []
_current_point_labels = {}
_current_point_labels_history = []
_current_point_history_index = -1
_available_labels = []


def init_annotations(config):
    """
    Initializes the annotations helper

    Parameters
    ----------
    config
    """
    global _current_point_selection, _current_point_labels, _available_labels
    clear_selections()
    clear_history()

    _available_labels = config["labels"]
    for label in _available_labels:
        logger.annotations.debug("Loaded available annotation label: %s" % str(label))


def register_plot_annotation(plot_name: str = None):
    """
    Registers a plot for annotation capabilities

    Parameters
    ----------
    plot_name
    """
    global _current_point_labels
    if plot_name is not None and plot_name not in _current_point_labels.keys():
        _current_point_labels[plot_name] = []
        update_history()
    else:
        df = data.get_dataframe()
        for column in df.columns[1:]:
            _current_point_labels[column] = []
        clear_history()
        update_history()


def add_selection_point(selected_point: 'SelectedPoint') -> bool:
    """
    Adds a point as selected point

    Parameters
    ----------
    selected_point
    """
    if not is_point_selected(selected_point):
        _current_point_selection.append(selected_point)
        logger.annotations.debug(
            "Added selected-point %i (%0.02f,%0.02f)" % (selected_point.index, selected_point.x, selected_point.val))
        return True
    else:
        return False


def is_point_selected(selected_point: 'SelectedPoint') -> bool:
    """
    Returns if a point is currently selected

    Parameters
    ----------
    selected_point
    """
    return selected_point in _current_point_selection


def is_point_labeled(plot_name: str, labeled_point: 'LabeledPoint') -> bool:
    """
    Returns if a point is currently labeled

    Parameters
    ----------
    plot_name
    labeled_point
    """
    return labeled_point in _current_point_labels[plot_name]


def get_all_selection_points():
    """
    Returns all selected points
    """
    return _current_point_selection


def add_labels_current_selection(plot_name: str, label: str):
    """
    Adds a label to the current point selection

    Parameters
    ----------
    plot_name
    label
    """
    for selected_point in _current_point_selection:
        add_label(plot_name, selected_point, label)
    update_history()


def delabel_selected(plot_name: str):
    """
    Removes label from the current selection

    Parameters
    ----------
    plot_name
    """
    for selected_point in get_all_selection_points():
        for labeled_point in get_labeled_points(plot_name):
            if selected_point == labeled_point:
                remove_labeled_point(plot_name, labeled_point)
    update_history()


def add_label_unselected(x, y, index, label, plot_name):
    """
    Adds labels to unselected points, e.g. after restoring

    Parameters
    ----------
    x
    y
    index
    label
    plot_name
    """
    lp = LabeledPoint(y, x, index, label, plot_name)
    _current_point_labels[plot_name].append(lp)
    update_history()


def restore_from_save(label_data):
    """
    Restores the annotations from save file

    Parameters
    ----------
    label_data
    """
    clear_selections()
    clear_history()

    for cl in data.get_dataframe_columns():
        register_plot_annotation(cl)

    for single_labeled_point in label_data:
        plot_name = single_labeled_point["plot_name"]
        lp = LabeledPoint(single_labeled_point["value"], single_labeled_point["x"], single_labeled_point["index"],
                          single_labeled_point["label"], plot_name)
        if plot_name not in _current_point_labels:
            _current_point_labels[plot_name] = []
        _current_point_labels[plot_name].append(lp)
    update_history()


def format_for_save():
    """
    Formats the annotations for the save file
    """
    output = []
    for key, val in _current_point_labels.items():
        for single_labeled_point in _current_point_labels[key]:
            output.append({'value': single_labeled_point.val,
                           'x': single_labeled_point.x,
                           'index': single_labeled_point.index,
                           'label': single_labeled_point.label,
                           'plot_name': single_labeled_point.plot_name})
    return output


def get_all_labeled_points():
    """
    Returns all labeled points
    """
    return _current_point_labels


def get_labeled_points(plot_name: str):
    """
    Returns all labeled points for a specific plot

    Parameters
    ----------
    plot_name
    """
    if plot_name in _current_point_labels:
        return _current_point_labels[plot_name]
    else:
        return []


def remove_labeled_point(plot_name: str, labeled_point: 'LabeledPoint') -> bool:
    """
    Removes a label from a point

    Parameters
    ----------
    plot_name
    labeled_point
    """
    if is_point_labeled(plot_name, labeled_point):
        _current_point_labels[plot_name].remove(labeled_point)
        logger.annotations.debug("Removed labeled-point %i (%s) (%0.02f,%0.02f) in plot %s" % (
            labeled_point.index, labeled_point.label, labeled_point.x, labeled_point.val, plot_name))
        return True
    else:
        return False


def add_label(plot_name: str, selected_point: 'SelectedPoint', label: str):
    """
    Adds a label to a point

    Parameters
    ----------
    plot_name
    selected_point
    label
    """
    lp = cast_clicked_to_labeled(plot_name, selected_point, label)
    if lp not in _current_point_labels[plot_name]:
        _current_point_labels[plot_name].append(lp)
        logger.annotations.debug(
            "Added labeled-point %i (%s) (%0.02f,%0.02f) in plot %s" % (lp.index, lp.label, lp.x, lp.val, plot_name))
    else:
        _current_point_labels[plot_name].remove(lp)
        _current_point_labels[plot_name].append(lp)
        logger.annotations.debug(
            "Changed labeled-point %i (%s) (%0.02f,%0.02f) in plot %s" % (lp.index, lp.label, lp.x, lp.val, plot_name))

    savestate.get_current_state().set_labels(format_for_save())


def get_number_selected():
    """
    Returns the number of selected points
    """
    return len(_current_point_selection)


def get_number_labeled(active_plot):
    """
    Returns the number of labeled points

    Parameters
    ----------
    active_plot
    """
    if active_plot:
        return len(_current_point_labels[active_plot])
    return 0


def remove_selected_point(selected_point: 'SelectedPoint') -> bool:
    """
    Removes the selection of a point

    Parameters
    ----------
    selected_point
    """
    if is_point_selected(selected_point):
        _current_point_selection.remove(selected_point)
        logger.annotations.debug(
            "Removed selection-point %i (%0.02f,%0.02f)" % (selected_point.index, selected_point.x, selected_point.val))
        return True
    else:
        return False


def get_available_labels():
    """
    Returns the currently available labels

    """
    global _available_labels
    return _available_labels


def cast_clicked_to_labeled(plot_name: str, selected_point: 'SelectedPoint', label: str):
    """
    Casts selected points to labeled points

    Parameters
    ----------
    plot_name
    selected_point
    label
    """
    return LabeledPoint(selected_point.val, selected_point.x, selected_point.index, label, plot_name)


def clear_selections():
    """
    Deselects all selected points
    """
    global _current_point_selection, _current_point_labels
    _current_point_selection = []


def clear_history():
    """
    Deletes history of Annotation-States
    """
    global _current_point_labels_history, _current_point_history_index
    _current_point_labels_history = []
    _current_point_history_index = -1
    logger.annotations.debug("Cleared History, now length 0")


def update_history():
    """
    Appends current Annotation-State to the history, removes history ahead of current index
    """
    global _current_point_labels, _current_point_labels_history, _current_point_history_index
    del _current_point_labels_history[_current_point_history_index+1:]
    _current_point_labels_history.append(copy.deepcopy(_current_point_labels))
    _current_point_history_index += 1
    logger.annotations.debug("Updated History, now length "+str(len(_current_point_labels_history)))


def apply_selected_history():
    """
    Sets Annotation_State to State stored at selected index
    """
    global _current_point_labels, _current_point_labels_history, _current_point_history_index
    _current_point_labels = _current_point_labels_history[_current_point_history_index]
    format_for_save()
    logger.annotations.debug("Changed History index to "+str(_current_point_history_index))


def history_go_step_back():
    """
    Goes back to previous state of history
    """
    global _current_point_history_index
    if _current_point_history_index > 0:
        _current_point_history_index -= 1
        apply_selected_history()


def history_go_step_forward():
    """
    Goes forward to next state of history
    """
    global _current_point_history_index, _current_point_labels_history
    if _current_point_history_index < len(_current_point_labels_history)-1:
        _current_point_history_index += 1
        apply_selected_history()


class Point(object):
    """
    A representation of a single point
    """

    def __init__(self, val: float, x: float, index: int):
        self.val = val
        self.x = x
        self.index = index

    def __eq__(self, other):
        return self.index == other.index

    def __getstate__(self):
        return [self.val, self.x, self.index]

    def __setstate__(self, state):
        self.val = state[0]
        self.x = state[1]
        self.index = state[2]


class LabeledPoint(Point):
    """
    A representation of a labeled point
    """

    def __init__(self, val: float, x: float, index: int, label: str, plot_name: str):
        super(LabeledPoint, self).__init__(val, x, index)
        self.label = label
        self.plot_name = plot_name

    def __getstate__(self):
        return [self.val, self.x, self.index, self.label, self.plot_name]

    def __setstate__(self, state):
        self.val = state[0]
        self.x = state[1]
        self.index = state[2]
        self.label = state[3]
        self.plot_name = state[4]

    def __eq__(self, other):
        return self.index == other.index and self.plot_name == other.plot_name

    def show(self):
        self.visible = True

    def hide(self):
        self.visible = False

    def get_data(self):
        return self.val, self.x, self.index, self.label


class SelectedPoint(Point):
    """
    A representation of a selected point
    """

    def __init__(self, val: float, x: float, index: int):
        super(SelectedPoint, self).__init__(val, x, index)
