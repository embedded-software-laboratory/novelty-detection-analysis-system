import sys
import traceback
from abc import abstractmethod

import numpy as np
import pandas as pd
from PyQt5.QtCore import QObject, pyqtSignal, QRunnable

from ndas.extensions import physiologicallimits
from ndas.misc import parameter
from ndas.utils import logger


class BaseDetector(QRunnable):
    """ Base class for all detectors

    Subclass of QRunnable because the detection is run on multiple threads.
    """

    @abstractmethod
    def __init__(self, datasets, *args, **kwargs):
        """ Initialise the Detector as subclass of QRunnable

        Parameters
        ----------
        datasets
        args
        kwargs
        """
        super(BaseDetector, self).__init__()
        self.required_arguments = []
        self.datasets = datasets
        self.args = args
        self.kwargs = kwargs

        self.signals = DetectorSignals()  # Required to send signals

    @abstractmethod
    def detect(self, datasets, **kwargs):
        """ Start the detection of novelties in given dataset

        Abstract method, requires to be implemented in subclass of BaseDetector.

        Parameters
        ----------
        datasets
        kwargs

        Returns
        -------
        novelties : dict
            A dict with observation offset as key and bool as value for detected novelties
        """
        raise NotImplementedError('Subclasses must override detect()!')

    def run(self):
        """ Start the detection

        QRunnable require a run method for multithreading. This method calls the detect method of the
        active BaseDetector subclass. The result is send to slot and not returned.
        """
        try:
            self.signal_percentage(5)
            result = self.detect(self.datasets, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signal_error(str(value))
        else:
            self.signal_result(result)
        finally:
            self.signal_percentage(100)

    def register_parameter(self, argument_name: str, type: parameter.ArgumentType = None, default: any = None,
                           min=None, max=None, tooltip=None):
        """ Register a new required argument for the Detector implementation

        Parameters
        ----------
        tooltip
        argument_name : str
            The name of the argument
        type: str
            The type of the argument (bool, string, integer, float)
        default:
            The default value of the parameter
        min:
            The minimum of the integer/float parameter
        max:
            The maximum of the integer/float parameter
        """

        if type == parameter.ArgumentType.FLOAT or type == parameter.ArgumentType.INTEGER:
            arg = parameter.AdditionalNumberParameter(argument_name)

            if min is not None:
                arg.set_minimum(min)

            if max is not None:
                arg.set_maximum(max)

        else:
            arg = parameter.AdditionalParameter(argument_name)

        if type is not None:
            arg.set_type(type)

        if default is not None:
            arg.set_default(default)

        if tooltip is not None:
            arg.set_tooltip(tooltip)

        if arg not in self.required_arguments:
            self.required_arguments.append(arg)

    def are_required_arguments_satisfied(self, **args) -> bool:
        """ Check if the given arguments satisfy the required arguments for the active Detector.

        Parameters
        ----------
        args

        Returns
        -------
        status : bool
            True if all required arguments are satisfied, otherwise False
        """
        if not all(arg.argument_name in args for arg in self.required_arguments):
            return False
        else:
            return True

    def get_required_arguments(self) -> list:
        """ Returns the required arguments list

        Returns
        -------
        required arguments : list
            The registered arguments of the active Detector implementation.
        """
        return self.required_arguments

    def signal_error(self, msg: str):
        """ Send a error message that requires confirmation from user.

        Parameters
        ----------
        msg : str
            The message that is displayed on the error message window.
        """
        self.signals.error_signal.emit(msg)

    def signal_percentage(self, pct: int):
        """ Set the percentage of the progress bar in GUI.

        Parameters
        ----------
        pct : int
            The percentage of the progress bar to be set.
        """
        self.signals.status_signal.emit(pct)

    def signal_result(self, result: dict):
        """ Send the result of the detection to the slot of the mainwindow class.

        Parameters
        ----------
        result : dict
            The dict of novelties after the detection.
        """
        self.signals.result_signal.emit(result)

    def signal_add_plot(self, name, x_data, y_data, x_label, y_label):
        """ Adds a new plot to the plot list

        Parameters
        ----------
        name
        x_data
        y_data
        x_label
        y_label
        """
        self.signals.add_plot_signal.emit(name, x_data, y_data, x_label, y_label)

    def signal_add_line(self, plot_name, line_name, x_data, y_data):
        """ Adds a line to an existing plot

        Parameters
        ----------
        plot_name
        line_name
        x_data
        y_data
        """
        if isinstance(x_data, list):
            x_data = pd.Series(x_data)

        if isinstance(y_data, list):
            y_data = pd.Series(y_data)

        if isinstance(x_data, np.ndarray):
            x_data = pd.Series(x_data)
            y_data = pd.Series(y_data)

        self.signals.add_line_signal.emit(plot_name, line_name, x_data, y_data)

    def signal_add_infinite_line(self, plot_name, line_name, y):
        """ Adds a labeled infinite line to an existing plot

        Parameters
        ----------
        plot_name
        line_name
        y
        """
        self.signals.add_infinite_line_signal.emit(plot_name, line_name, y)

    def log(self, msg):
        """ Send a message to the logger

        Parameters
        ----------
        msg: str
            The message
        """
        message = "[Running: " + self.__repr__() + "] " + msg
        logger.algorithms.debug(message)

    def __repr__(self) -> str:
        """ Returns a printable representation of the given object

        Returns
        -------
        class name : str
            The printable representation of the class.
        """
        class_name = self.__class__.__name__
        return '%s' % class_name

    def get_physiological_information(self, name) -> physiologicallimits.PhysiologicalDataType:
        """ Returns the physiological information/limits of the given name/alias

        Parameters
        ----------
        name : str
            The name or the alias of the requested physical information

        Returns
        -------
        physical information type : PhysicalDataType
            The PhysicalDataType of the requested name/alias

        """
        return physiologicallimits.get_physical_dt(name)


class DetectorSignals(QObject):
    """ Class for the signals that are send by the Detector implementation.

    Signals can only be sent from subclasses of QObject. The detectors are subclasses of
    QRunnable to run in multithreading.

    This class is purely used to send signals.
    """
    status_signal = pyqtSignal(int)
    error_signal = pyqtSignal(str)
    result_signal = pyqtSignal(dict)
    add_plot_signal = pyqtSignal(str, pd.Series, pd.Series, str, str)
    add_line_signal = pyqtSignal(str, str, pd.Series, pd.Series)
    add_infinite_line_signal = pyqtSignal(str, str, float)
