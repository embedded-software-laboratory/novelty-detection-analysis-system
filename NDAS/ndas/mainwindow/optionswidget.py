from PyQt5.QtCore import QAbstractTableModel, Qt, QSortFilterProxyModel, pyqtSignal, QRect, QPoint, pyqtSlot, QAbstractAnimation, QPropertyAnimation, QParallelAnimationGroup
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pandas as pd
import numpy as np
import yaml
import qtwidgets
import math
import copy
from ndas.extensions import data, annotations, plots, physiologicallimits
from ndas.utils import logger
from ndas.misc import colors


class OptionsWindow(QMainWindow):
    def __init__(self, parent=None):
        self.parent = parent
        super().__init__(parent)
        self.container = QWidget()
        self.container_lay = QVBoxLayout()
        self.container.setLayout(self.container_lay)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.setCentralWidget(self.container)
        self.container_lay.addWidget(self.scroll)
        self.optionswidget = OptionsWidget(self)
        self.scroll.setWidget(self.optionswidget)
        width = self.optionswidget.geometry().width()*1.1
        height = self.parent.frameGeometry().height()*0.9
        self.scroll.setMinimumWidth(width)
        self.resize(width, height)
        self.setWindowTitle("Configure settings")
        self.move(parent.frameGeometry().center() - QRect(QPoint(), self.size()).center())

    def closeEvent(self, event):
        self.parent.optionsopened = False
        event.accept()


class OptionsWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.layout = QVBoxLayout()
        self.config = yaml.safe_load(open("ndas/config/config.yml"))
        self.setLayout(self.layout)
        self.groupbox_dict = {}
        self.create_settings_from_dict(self.layout, self.config, self.groupbox_dict)
        self.button_layout = QHBoxLayout()
        self.apl_cur_ses_btn = QPushButton("Apply Settings for Current Session")
        self.apl_cur_ses_btn.clicked.connect(lambda: self.apply_settings_to_session())
        self.sav_set_btn = QPushButton("Apply Settings and Save for future Sessions")
        self.sav_set_btn.clicked.connect(lambda: self.save_settings_and_apply())
        self.button_layout.addWidget(self.apl_cur_ses_btn)
        self.button_layout.addWidget(self.sav_set_btn)
        if parent:
            parent.container_lay.addLayout(self.button_layout)
        else:
            self.layout.addLayout(self.button_layout)

    def create_settings_from_dict(self, layout, dict_in, dict_of_fields, collapsible=False):
        element_number = 0
        for k, v in dict_in.items():
            if v:
                local_dict = {}
                if collapsible:
                    groupbox = CollapsibleBox(k)
                elif k == "physiologicalinfo":
                    groupbox = QGroupBox(k)
                    gr_layout = QVBoxLayout()
                    gr_scrollarea = QScrollArea()
                    gr_scrollarea.setFrameShape(QFrame.NoFrame)
                    gr_layout.addWidget(gr_scrollarea)
                    groupbox.setLayout(gr_layout)
                    gr_scrollarea.setWidgetResizable(True)
                    scroll_content = QWidget()
                    gr_scrollarea.setWidget(scroll_content)
                else:
                    groupbox = QGroupBox(k)
                if k == "physiologicalinfo":
                    groupbox_layout = QVBoxLayout()
                    self.create_settings_from_dict(groupbox_layout, v, local_dict, collapsible=True)
                elif k == "colors":
                    groupbox_layout = QGridLayout()
                    element_number = 0
                else:
                    groupbox_layout = QFormLayout()
                    groupbox_layout.setFieldGrowthPolicy(1)
                    groupbox_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                    groupbox_layout.setLabelAlignment(Qt.AlignLeft)
                layout.addWidget(groupbox)
                for k2, v2 in v.items():
                    if k == "physiologicalinfo":
                        skip = 1
                    elif k == "colors":
                        label = QLabel(k2)
                        groupbox_layout.addWidget(label, element_number // 3, 2*(element_number % 3))
                        color_button = QPushButton()
                        self.set_style_sheet(color_button, "#"+v2)
                        color_button.clicked.connect(lambda ignore, button_local=color_button: self.change_color(button_local))
                        groupbox_layout.addWidget(color_button, element_number // 3, 1 + 2*(element_number % 3))
                        local_dict[k2] = color_button
                        element_number += 1
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
                            new_item.setPlaceholderText("Title of new item")
                            a_button = QPushButton("Add as new item")
                            a_button.clicked.connect(lambda ignore, list_local=list_w, text_field_local=new_item: self.add_at_selected(list_local, text_field_local))
                            r_button = QPushButton("Remove Selected")
                            r_button.clicked.connect(lambda ignore, list_local=list_w: self.remove_selected(list_local))
                            buttons_layout.addWidget(new_item)
                            buttons_layout.addWidget(a_button)
                            buttons_layout.addWidget(r_button)
                            list_edit_layout = QVBoxLayout()
                            list_edit_layout.addWidget(list_w)
                            list_edit_layout.addLayout(buttons_layout)
                            groupbox_layout.addRow(label, list_edit_layout)
                            local_dict[k2] = list_w
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
                            local_dict[k2] = widget
                        elif isinstance(v2, int):
                            label = QLabel(k2)
                            label.setMinimumWidth(120)
                            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                            widget = QSpinBox()
                            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                            widget.setMinimum(-100000000)
                            widget.setMaximum(10000000)
                            if not v2:
                                widget.setSingleStep(1)
                            else:
                                widget.setSingleStep(10**int(math.log10(abs(v2))))
                            widget.setValue(v2)
                            groupbox_layout.addRow(label, widget)
                            local_dict[k2] = widget
                        elif isinstance(v2, float):
                            label = QLabel(k2)
                            label.setMinimumWidth(120)
                            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                            widget = QDoubleSpinBox()
                            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                            widget.setMinimum(-100000000)
                            widget.setMaximum(10000000)
                            widget.setDecimals(3)
                            if not v2:
                                widget.setSingleStep(1)
                            else:
                                widget.setSingleStep(10**int(math.log10(abs(v2))))
                            widget.setValue(v2)
                            groupbox_layout.addRow(label, widget)
                            local_dict[k2] = widget
                        elif isinstance(v2, str):
                            label = QLabel(k2)
                            label.setMinimumWidth(120)
                            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                            widget = QLineEdit()
                            widget.setText(v2)
                            groupbox_layout.addRow(label, widget)
                            local_dict[k2] = widget
                        else:
                            print(k2, "- This type of setting is not yet supported by the settings editor.")
                            local_dict[k2] = v2
                dict_of_fields[k] = self.copy_dict(local_dict)
                if collapsible:
                    groupbox.setContentLayout(groupbox_layout)
                elif k == "physiologicalinfo":
                    scroll_content.setLayout(groupbox_layout)
                    gr_scrollarea.setMinimumHeight(scroll_content.geometry().width()*0.5)
                else:
                    groupbox.setLayout(groupbox_layout)

    def remove_selected(self, list_in):
        selection = list_in.selectedItems()
        if not selection:
            return
        for item in selection:
            list_in.takeItem(list_in.row(item))

    def add_at_selected(self, list_in, text_field_in):
        list_in.addItem(text_field_in.text())
        text_field_in.setText("")

    def change_color(self, button):
        color = QColorDialog.getColor(button.palette().color(QPalette.Background), self)
        if color.isValid():
            self.set_style_sheet(button, color.name())

    def set_style_sheet(self, button, color_string):
        button.setStyleSheet("QAbstractButton"
                             "{"
                             "background-color:" + color_string + ";"
                             "border-style:outset;"
                             "border-color:grey;"
                             "border-width:1px;"
                             "border-radius:5px;"
                             "}"
                             "QAbstractButton:pressed"
                             "{"
                             "border-style:inset;"
                             "}"
                             )

    def apply_settings_to_session(self):
        changed_settings = self.get_settings_from_dict(self.groupbox_dict)
        self._update_modules(changed_settings)

    def save_settings_and_apply(self):
        self.apply_settings_to_session()
        changed_settings = self.get_settings_from_dict(self.groupbox_dict)
        with open("ndas/config/config.yml", 'w') as outfile:
            yaml.dump(changed_settings, outfile, sort_keys=False)

    def get_settings_from_dict(self, dict_in):
        result = {}
        for k, v in dict_in.items():
            if isinstance(v, dict):
                result[k] = self.get_settings_from_dict(v)
            elif isinstance(v, QListWidget):
                list_of_items = []
                for i in range(v.count()):
                    list_of_items.append(v.item(i).text())
                result[k] = list_of_items
            elif isinstance(v, qtwidgets.Toggle):
                result[k] = v.isChecked()
            elif isinstance(v, QSpinBox):
                result[k] = v.value()
            elif isinstance(v, QDoubleSpinBox):
                result[k] = v.value()
            elif isinstance(v, QLineEdit):
                result[k] = v.text()
            elif isinstance(v, QPushButton):
                result[k] = v.palette().color(QPalette.Background).name()[1:]
            else:
                result[k] = v
        return result

    def copy_dict(self, dict_in):
        result = {}
        for k, v in dict_in.items():
            if isinstance(v, dict):
                result[k] = self.copy_dict(v)
            else:
                result[k] = v
        return result

    def _update_modules(self, config):
        """
        Updates all modules with config

        Parameters
        ----------
        config
        """
        logger.init.debug("Updating annotations...")
        annotations.set_available_labels(config["annotation"])
        self.parent.parent.update_labels()

        logger.init.debug("Updating importer...")
        data.update_data_importer(config["data"])

        logger.init.debug("Updating physiological info...")
        physiologicallimits.update_physiological_info(config["physiologicalinfo"])

        logger.init.debug("Updating plots...")
        plots.update_graphs(config["plots"])  # darkmode requires app restart

        logger.init.debug("Updating colors...")
        colors.init_colors(config["colors"])  # colors are shown with next loaded plot


class CollapsibleBox(QWidget):
    def __init__(self, title="", parent=None):
        super(CollapsibleBox, self).__init__(parent)

        self.toggle_button = QToolButton(
            text=title, checkable=True, checked=True
        )
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.clicked.connect(self.on_pressed)

        self.toggle_animation = QParallelAnimationGroup(self)

        self.content_area = QScrollArea(
            maximumHeight=0, minimumHeight=0
        )
        self.content_area.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        self.content_area.setWidgetResizable(True)
        self.content_area.setFrameShape(QFrame.NoFrame)

        lay = QVBoxLayout(self)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toggle_button)
        lay.addWidget(self.content_area)

        self.toggle_animation.addAnimation(
            QPropertyAnimation(self, b"minimumHeight")
        )
        self.toggle_animation.addAnimation(
            QPropertyAnimation(self, b"maximumHeight")
        )
        self.toggle_animation.addAnimation(
            QPropertyAnimation(self.content_area, b"maximumHeight")
        )

    @pyqtSlot()
    def on_pressed(self):
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(
            Qt.DownArrow if not checked else Qt.RightArrow
        )
        self.toggle_animation.setDirection(
            QAbstractAnimation.Forward
            if not checked
            else QAbstractAnimation.Backward
        )
        self.toggle_animation.start()

    def setContentLayout(self, layout):
        lay = self.content_area.layout()
        del lay
        self.content_area.setLayout(layout)
        collapsed_height = (
            self.sizeHint().height() - self.content_area.maximumHeight()
        )
        content_height = layout.sizeHint().height()
        for i in range(self.toggle_animation.animationCount()):
            animation = self.toggle_animation.animationAt(i)
            animation.setDuration(500)
            animation.setStartValue(collapsed_height)
            animation.setEndValue(collapsed_height + content_height)

        content_animation = self.toggle_animation.animationAt(
            self.toggle_animation.animationCount() - 1
        )
        content_animation.setDuration(500)
        content_animation.setStartValue(0)
        content_animation.setEndValue(content_height)
