from PyQt5.QtCore import QAbstractTableModel, Qt, QSortFilterProxyModel
from PyQt5.QtWidgets import *


class DataInspectionWidget(QWidget):
    """
    Widget to visualize the imported or generated data in table view
    """

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
        self.model = DataframeModel(dataframe)

        ''' Enable sorting by using a proxy model'''
        self.proxy_model = QSortFilterProxyModel()
        self.proxy_model.setSourceModel(self.model)

        self.tableView.setModel(self.proxy_model)


class DataframeModel(QAbstractTableModel):
    """
    Model for visualizing dataframes
    """

    def __init__(self, data):
        """
        Initializes the QAbstractTableModel

        Parameters
        ----------
        data
        """
        QAbstractTableModel.__init__(self)
        self._data = data

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
            if role == Qt.DisplayRole:
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
