import csv

from PyQt5.QtCore import (QEvent, Qt)
from PyQt5.QtWidgets import *
from pyqtgraph.Qt import QtGui


class ImportWindow(QWidget):
    """
    Widget to specify import of CSV files
    Current not in use.
    """

    def __init__(self, file, *args, **kwargs):
        """
        Creates the window and the table view of the selected data

        Parameters
        ----------
        file
        args
        kwargs
        """
        super(ImportWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle('File Importer')
        self.setMinimumSize(600, 400)

        self.file_name = file

        self.model = QtGui.QStandardItemModel(self)

        self.tableView = QTableView(self)
        self.tableView.setModel(self.model)
        self.tableView.horizontalHeader().setStretchLastSection(True)
        self.tableView.setSelectionMode(QAbstractItemView.MultiSelection)

        self.layoutVertical = QVBoxLayout(self)
        self.layoutVertical.addWidget(self.tableView)
        # self.loadCsv(self.file_name)

        # delegate = CheckBoxDelegate(None)
        # self.tableView.setItemDelegateForColumn(1, delegate)

        with open(self.file_name, "r") as fileInput:
            for row in csv.reader(fileInput):
                item_list = []
                for field in row:
                    rowItem = QtGui.QStandardItem(field)
                    # rowItem.setSelectable(False)
                    item_list.append(rowItem)
                self.model.appendRow(item_list)

        selectionModel = self.tableView.selectionModel()
        selectionModel.selectionChanged.connect(self.selChanged)

        # btn = QPushButton('Hello')
        # self.setCellWidget(0, 0, btn)

    def selChanged(self, selected, deselected):
        """
        Updates the selection in the table view

        Parameters
        ----------
        selected
        deselected
        """
        for index in sorted(self.tableView.selectionModel().selectedColumns()):
            column = index.column()
            header = self.model.data(self.model.index(0, column))
            print(header)


class CheckBoxDelegate(QItemDelegate):
    """
    Custom implementation to display checkbox in column headers
    """

    def __init__(self, parent):
        """
        Creates the QItemDelegate

        Parameters
        ----------
        parent
        """
        QItemDelegate.__init__(self, parent)

    def createEditor(self, parent, option, index):
        """
        Parameters
        ----------
        parent
        option
        index
        """
        return None

    def paint(self, painter, option, index):
        """
        Paints the checkbox

        Parameters
        ----------
        painter
        option
        index
        """
        self.drawCheck(painter, option, option.rect, Qt.Unchecked if int(index.data()) == 0 else Qt.Checked)

    def editorEvent(self, event, model, option, index):
        """
        Updates the model if checkbox checked

        Parameters
        ----------
        event
        model
        option
        index
        """
        if not int(index.flags() & Qt.ItemIsEditable) > 0:
            return False

        if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
            self.setModelData(None, model, index)
            return True

        return False

    def setModelData(self, editor, model, index):
        """
        Sets the model

        Parameters
        ----------
        editor
        model
        index
        """
        model.setData(index, 1 if int(index.data()) == 0 else 0, Qt.EditRole)
