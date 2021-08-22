import logging

from PyQt5 import QtWidgets
from PyQt5.QtCore import (pyqtSlot, pyqtSignal)


class QPlainTextEditLogger(QtWidgets.QPlainTextEdit, logging.Handler):
    """
    A logger widget for the annotation tab
    """
    appendPlainTextSignal = pyqtSignal(str)

    def __init__(self):
        """
        Connects the QplainTextEdit with the logger handler
        """
        super().__init__()
        self.setReadOnly(True)
        self.ensureCursorVisible()

        self.verticalScrollBar().rangeChanged.connect(self.update_scrollbar)
        self.appendPlainTextSignal.connect(self.appendPlainText)

        self.setFormatter(logging.Formatter('%(asctime)s - %(message)s', "%H:%M"))
        logging.getLogger().addHandler(self)
        logging.getLogger().setLevel(logging.DEBUG)

    def emit(self, record):
        """
        Emits if new log is available

        Parameters
        ----------
        record
        """
        msg = self.format(record)
        self.appendPlainTextSignal.emit(msg)

    @pyqtSlot(int, int)
    def update_scrollbar(self, min, max):
        """
        Updates the scrollbar
        (Autoscroll)

        Parameters
        ----------
        min
        max
        """
        self.verticalScrollBar().setSliderPosition(self.verticalScrollBar().maximum())
