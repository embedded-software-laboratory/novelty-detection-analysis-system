from PyQt5.QtCore import QAbstractTableModel, Qt, QSortFilterProxyModel, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import *
import pandas as pd
import numpy as np
from ndas.extensions import physiologicallimits, data
from ndas.utils import logger


class DataInspectionWidget(QWidget):
    """
    Widget to visualize the imported or generated data in table view
    """

    data_edit_signal = pyqtSignal(pd.DataFrame, pd.DataFrame)

    def __init__(self):
        """
        Creates the layout and the table
        """
        super().__init__()

        self.layout = QGridLayout(self)
        self.tableView = QTableView(self)
        self.tableView.setSortingEnabled(True)
        self.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.apply_edits_button = QPushButton("Apply Changes (may take a few seconds)")
        self.apply_edits_button.clicked.connect(lambda: self.apply_edits())
        self.layout.addWidget(self.apply_edits_button, 0, 0)
        self.layout.addWidget(self.tableView, 1, 0)

        self.model = None
        self.proxy_model = None

    def set_data(self, dataframe):
        """
        Loads the current dataframe into the table

        Parameters
        ----------
        dataframe
        """
        self.model = DataframeModel(dataframe)
        self.model.table_change_signal.connect(lambda: self.tableView.viewport().update())

        ''' Enable sorting by using a proxy model'''
        self.proxy_model = QSortFilterProxyModel()
        self.proxy_model.setSourceModel(self.model)

        self.tableView.setModel(self.proxy_model)

    def apply_edits(self):
        if self.model:
            current_data_mask = data.get_mask_dataframe()
            if isinstance(current_data_mask, pd.DataFrame):
                updated_mask = current_data_mask.mask(self.model.get_internal_mask().applymap(bool), 1)
            else:
                updated_mask = self.model.get_internal_mask().copy(deep=True)
            data.set_mask_dataframe(updated_mask)
            self.data_edit_signal.emit(self.model.get_internal_data(), self.model.get_internal_mask())

    def history_forward(self):
        if self.model:
            self.model.history_forward()

    def history_backward(self):
        if self.model:
            self.model.history_backward()


class DataframeModel(QAbstractTableModel):
    """
    Model for visualizing dataframes
    """
    table_change_signal = pyqtSignal()

    def __init__(self, data):
        """
        Initializes the QAbstractTableModel

        Parameters
        ----------
        data
        """
        QAbstractTableModel.__init__(self)
        self._data = data.copy(deep=True)
        self._ref_data = data
        self._changes = pd.DataFrame(0, index=data.index, columns=data.columns)
        self.history = [(self._data.copy(deep=True), self._changes.copy(deep=True))]
        self.history_index = 0

    def flags(self, index):
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable

    def setData(self, index, value, role):
        """
        Sets data-point at index to value.
        Column ID uses int values
        All other values use float
        Nan can be set by putting a number above 2^32 (~4.3e9)

        Parameters
        ----------
        index, value, role
        """
        if role == Qt.EditRole:
            if value >= 4294967296:
                value = np.nan
            if "ID" in self._data.columns[index.column()]:
                self._data.iloc[index.row(), index.column()] = int(value)
            else:
                self._data.iloc[index.row(), index.column()] = float(value)
            if self._ref_data.iloc[index.row(), index.column()] == self._data.iloc[index.row(), index.column()] or (np.isnan(self._ref_data.iloc[index.row(), index.column()]) and np.isnan(self._data.iloc[index.row(), index.column()])):
                self._changes.iloc[index.row(), index.column()] = 0
            else:
                self._changes.iloc[index.row(), index.column()] = 1
            del self.history[self.history_index + 1:]
            history_data = self._data.copy(deep=True)
            history_changes = self._changes.copy(deep=True)
            self.history.append((history_data, history_changes))
            self.history_index += 1
            logger.inspector.debug("Updated History, now length " + str(len(self.history)))
            return True

    def apply_history(self):
        self._data = self.history[self.history_index][0].copy(deep=True)
        self._changes = self.history[self.history_index][1].copy(deep=True)
        self.table_change_signal.emit()
        logger.inspector.debug("Changed History index to "+str(self.history_index))

    def history_backward(self):
        """
        Goes back to previous state of history
        """
        if self.history_index > 0:
            self.history_index -= 1
            self.apply_history()

    def history_forward(self):
        """
        Goes forward to next state of history
        """
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.apply_history()

    def get_internal_data(self):
        return self._data

    def get_internal_mask(self):
        return self._changes

    def rowCount(self, parent=None):
        """
        Returns the number of rows
        Required to reimplement.

        Parameters
        ----------
        parent
        """
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        """
        Returns the number of columns
        Required to reimplement.

        Parameters
        ----------
        parnet
        """
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        """
        Highlights selected data
        Required to reimplement.

        Use int for ID to avoid scientific notation

        Edited values are shown in light yellow
        Values outside the physiological limits are shown in light red
        NAN are shown with light grey text

        Parameters
        ----------
        index
        role
        """
        if index.isValid():
            if role == Qt.DisplayRole or role == Qt.EditRole:
                if "ID" in self._data.columns[index.column()]:
                    return int(self._data.iloc[index.row(), index.column()])
                else:
                    return float(self._data.iloc[index.row(), index.column()])
            if role == Qt.BackgroundRole:
                phys_limits = physiologicallimits.get_physical_dt(self._data.columns[index.column()])
                if phys_limits and not np.isnan(self._data.iloc[index.row(), index.column()]) and not phys_limits.low <= self._data.iloc[index.row(), index.column()] <= phys_limits.high:
                    return QColor(255, 204, 204)
                if self._changes.iloc[index.row(), index.column()]:
                    return QColor(255, 255, 204)
            if role == Qt.ForegroundRole:
                if np.isnan(self._data.iloc[index.row(), index.column()]):
                    return QColor(192, 192, 192)
        return None

    def headerData(self, section, orientation, role):
        """
        Loads header data into the table.
        Required to reimplement.

        Parameters
        ----------
        section
        orientation
        role
        """
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[section]
        return None
