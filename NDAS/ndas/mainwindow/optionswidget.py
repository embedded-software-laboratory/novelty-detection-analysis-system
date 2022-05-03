from PyQt5.QtCore import QAbstractTableModel, Qt, QSortFilterProxyModel, pyqtSignal, QRect, QPoint
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pandas as pd
import numpy as np
import yaml
import qtwidgets
import math


class OptionsWindow(QMainWindow):
    def __init__(self, parent=None):
        self.parent = parent
        super().__init__(parent)
        self.setCentralWidget(OptionsWidget(self))
        self.setWindowTitle("Configure settings")
        self.move(parent.frameGeometry().center() - QRect(QPoint(), self.sizeHint()).center())

    def closeEvent(self, event):
        self.parent.optionsopened = False
        event.accept()


class OptionsWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__()
        self.layout = QVBoxLayout()
        self.config = yaml.safe_load(open("ndas/config/config.yml"))
        self.groupbox_dict = {}
        self.setLayout(self.layout)
        for k, v in self.config.items():
            if v:
                groupbox = QGroupBox(k)
                groupbox_layout = QFormLayout()
                groupbox_layout.setFieldGrowthPolicy(1)
                groupbox_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                groupbox_layout.setLabelAlignment(Qt.AlignLeft)
                groupbox.setLayout(groupbox_layout)
                self.groupbox_dict[k] = groupbox_layout
                self.layout.addWidget(groupbox)
                for k2, v2 in v.items():
                    if k == "physiologicalinfo":
                        print(k2)
                    elif k == "colors":
                        print(k2)
                    else:
                        if isinstance(v2, list):
                            label = QLabel(k2)
                            label.setMinimumWidth(120)
                            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                            list_w = QListWidget()
                            list_w.addItems(v2)
                            list_w.setMinimumWidth(list_w.sizeHintForColumn(0))
                            list_w.setMinimumHeight(list_w.sizeHintForRow(0)*(2 + list_w.count()))
                            list_w.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
                            buttons_layout = QHBoxLayout()
                            new_item = QLineEdit()
                            a_button = QPushButton("Add as new item")
                            r_button = QPushButton("Remove Selection")
                            r_button.clicked.connect(lambda: self.remove_selected(list_w))
                            buttons_layout.addWidget(new_item)
                            buttons_layout.addWidget(a_button)
                            buttons_layout.addWidget(r_button)
                            list_edit_layout = QVBoxLayout()
                            list_edit_layout.addWidget(list_w)
                            list_edit_layout.addLayout(buttons_layout)
                            groupbox_layout.addRow(label, list_edit_layout)
                        elif isinstance(v2, bool):
                            label = QLabel(k2)
                            label.setMinimumWidth(120)
                            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                            label.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
                            toggling_layout = QHBoxLayout()
                            widget = qtwidgets.Toggle(handle_color=Qt.gray, bar_color=Qt.lightGray, checked_color=(self.palette().color(QPalette.Highlight)))
                            toggling_layout.addItem(QSpacerItem(5, 5, hPolicy=QSizePolicy.MinimumExpanding))
                            toggling_layout.addWidget(widget)
                            toggling_layout.addItem(QSpacerItem(5, 5, hPolicy=QSizePolicy.MinimumExpanding))
                            widget.setChecked(v2)
                            groupbox_layout.addRow(label, toggling_layout)
                        elif isinstance(v2, int):
                            label = QLabel(k2)
                            label.setMinimumWidth(120)
                            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                            widget = QSpinBox()
                            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                            widget.setMaximum(10000000)
                            widget.setSingleStep(10**int(math.log10(v2)))
                            widget.setValue(v2)
                            groupbox_layout.addRow(label, widget)
                        elif isinstance(v2, str):
                            label = QLabel(k2)
                            label.setMinimumWidth(120)
                            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                            widget = QLineEdit()
                            widget.setText(v2)
                            groupbox_layout.addRow(label, widget)
                        else:
                            print("hope not")

    def remove_selected(self, list_in):
        selection = list_in.selectedItems()
        print(selection)
        if not selection:
            return
        for item in selection:
            list_in.takeItem(list_in.row(item))
