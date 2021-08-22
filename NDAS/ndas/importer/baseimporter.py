import sys
import traceback
from abc import abstractmethod

import pandas as pd
from PyQt5.QtCore import QObject, pyqtSignal, QRunnable

from ndas.utils import logger


class BaseImporter(QRunnable):
    """
    Base class, that has to be extended with implemented importers
    """

    @abstractmethod
    def __init__(self, files, *args, **kwargs):
        """
        Abstract method of the init
        Registers importer signals.

        Parameters
        ----------
        files
        args
        kwargs
        """
        super(BaseImporter, self).__init__(*args, **kwargs)

        self.files = files
        self.signals = ImporterSignals()

    @abstractmethod
    def get_dataframe(self, file):
        """
        Abstract method to return a dataframe with loaded data

        Parameters
        ----------
        file

        """
        # return list of DataSets
        raise NotImplementedError('Subclasses must override get_dataframe()!')

    @abstractmethod
    def get_labels(self, files):
        """
        Abstract method to return data labels

        Parameters
        ----------
        files
        """
        return []

    def run(self):
        """
        Re-implemented run method for QRunnable
        Is executed in multithreading.
        """
        try:
            self.signal_percentage(15)
            result = self.get_dataframe(self.files)
            labels = self.get_labels(self.files)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signal_error(str(value))
        else:
            self.signal_result(result, labels)
        finally:
            self.signal_percentage(100)

    def __repr__(self):
        """
        Representer method for autoloading

        """
        class_name = self.__class__.__name__
        return '%s' % class_name

    def log(self, msg):
        """
        Send a message to the logger

        Parameters
        ----------
        msg: str
            The message
        """
        message = "[Importer: " + self.__repr__() + "] " + msg
        logger.importer.debug(message)

    def signal_percentage(self, pct: int):
        """
        Set the percentage of the progress bar in GUI.

        Parameters
        ----------
        pct : int
            The percentage of the progress bar to be set.
        """
        self.signals.status_signal.emit(pct)

    def signal_error(self, msg: str):
        """
        Send a error message that requires confirmation from user.

        Parameters
        ----------
        msg : str
            The message that is displayed on the error message window.
        """
        self.signals.error_signal.emit(msg)

    def signal_result(self, df, labels):
        self.signals.result_signal.emit(df, labels)


class ImporterSignals(QObject):
    """
    Class for the signals that are send by the Importer implementation.

    Signals can only be sent from subclasses of QObject.
    The importers are subclasses of QRunnable to run in multithreading.

    This class is purely used to send signals.
    """
    status_signal = pyqtSignal(int)
    error_signal = pyqtSignal(str)
    result_signal = pyqtSignal(pd.DataFrame, list)
