from PyQt5.QtCore import QAbstractTableModel, Qt, QSortFilterProxyModel, pyqtSignal
from PyQt5.QtWidgets import *
import pandas as pd


class DataInspectionWidget(QWidget):
    """
    Widget to visualize the imported or generated data in table view
    """

    data_edit_signal = pyqtSignal(pd.DataFrame)

    def __init__(self):
        """
        Creates the layout and the table
        """
        super().__init__()

        self.layout = QGridLayout(self)
        self.tableView = QTableView(self)
        self.tableView.setSortingEnabled(True)
        self.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.layout.addWidget(self.tableView, 0, 0)

        self.model = None
        self.proxy_model = None

    def set_data(self, dataframe):
        """
        Loads the current dataframe into the table

        Parameters
        ----------
        dataframe
        """
        self.model = DataframeModel(dataframe, self.data_edit_signal)

        ''' Enable sorting by using a proxy model'''
        self.proxy_model = QSortFilterProxyModel()
        self.proxy_model.setSourceModel(self.model)

        self.tableView.setModel(self.proxy_model)


class DataframeModel(QAbstractTableModel):
    """
    Model for visualizing dataframes
    """

    def __init__(self, data, data_edit_signal):
        """
        Initializes the QAbstractTableModel

        Parameters
        ----------
        data
        """
        QAbstractTableModel.__init__(self)
        self._data = data
        self.data_edit_signal = data_edit_signal

    def flags(self, index):
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable

    def setData(self, index, value, role):
        if role == Qt.EditRole:
            print(value, type(value))
            if "ID" in self._data.columns[index.column()]:
                self._data.iloc[index.row(), index.column()] = int(value)
            else:
                self._data.iloc[index.row(), index.column()] = float(value)
            self.data_edit_signal.emit(self._data)
            return True

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
