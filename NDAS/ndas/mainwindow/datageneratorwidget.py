import pandas as pd
from PyQt5.QtCore import (pyqtSignal)
from PyQt5.QtWidgets import *

from ndas.extensions import annotations
from ndas.misc import datageneratorform
from ndas.utils import datagenerator


class DataGeneratorWidget(QWidget):
    """
    Widget to generate arbitrary test data
    """
    generated_data_signal = pyqtSignal(pd.DataFrame, list)
    register_annotation_plot_signal = pyqtSignal(str)
    update_labels_signal = pyqtSignal()

    def __init__(self, *args, **kwargs):
        """
        Creates visual elements for the widget

        Parameters
        ----------
        args
        kwargs
        """
        super(DataGeneratorWidget, self).__init__(*args, **kwargs)

        self.elements = []

        self.prepare_groupbox = QGroupBox("Main Settings")
        self.prepare_groupbox_layout = QHBoxLayout()
        self.prepare_groupbox.setLayout(self.prepare_groupbox_layout)

        self.number_of_dimensions_layout = QHBoxLayout()
        self.number_of_dimensions_label = QLabel("Number of Dimensions:")
        self.number_of_dimensions = QSpinBox()
        self.number_of_dimensions.setMinimumWidth(100)
        self.number_of_dimensions.setMinimum(2)
        self.number_of_dimensions.setMaximum(8)
        self.number_of_dimensions.setValue(5)
        self.number_of_dimensions_layout.addWidget(self.number_of_dimensions_label)
        self.number_of_dimensions_layout.addWidget(self.number_of_dimensions)
        self.number_of_dimensions.valueChanged.connect(lambda val: self.number_dimensions_changed(val))

        self.dataset_length_layout = QHBoxLayout()
        self.dataset_length_label = QLabel("Entries per Dimension:")
        self.dataset_length = QSpinBox()
        self.dataset_length.setMinimumWidth(100)
        self.dataset_length.setMinimum(100)
        self.dataset_length.setMaximum(10000)
        self.dataset_length.setValue(1000)
        self.dataset_length_layout.addWidget(self.dataset_length_label)
        self.dataset_length_layout.addWidget(self.dataset_length)

        self.dataset_length_observation_step_layout = QHBoxLayout()
        self.dataset_length_observation_step_label = QLabel("Observation Step")
        self.dataset_length_observation_step = QSpinBox()
        self.dataset_length_observation_step.setMinimumWidth(100)
        self.dataset_length_observation_step.setMinimum(1)
        self.dataset_length_observation_step.setMaximum(100)
        self.dataset_length_observation_step.setValue(1)
        self.dataset_length_observation_step_layout.addWidget(self.dataset_length_observation_step_label)
        self.dataset_length_observation_step_layout.addWidget(self.dataset_length_observation_step)

        self.dataset_seed_layout = QHBoxLayout()
        self.dataset_seed_label = QLabel("Seed")
        self.dataset_seed = QSpinBox()
        self.dataset_seed.setMinimum(1)
        self.dataset_seed.setMaximum(99999)
        self.dataset_seed.setValue(datagenerator.get_random_int(10000, 99999))
        self.dataset_seed.setSingleStep(1)
        self.dataset_seed_layout.addWidget(self.dataset_seed_label)
        self.dataset_seed_layout.addWidget(self.dataset_seed)

        self.data_modification_groupbox = QGroupBox("Data Modification")
        self.data_modification_groupbox_layout = QVBoxLayout()
        self.data_modification_groupbox.setLayout(self.data_modification_groupbox_layout)

        self.main_layout = QVBoxLayout(self)
        self.generate_button = QPushButton("Generate Data")
        self.generate_button.setMinimumWidth(100)
        self.generate_button.clicked.connect(lambda: self.generate_test_data())

        self.prepare_groupbox_layout.addLayout(self.number_of_dimensions_layout)
        self.prepare_groupbox_layout.addLayout(self.dataset_length_layout)
        self.prepare_groupbox_layout.addLayout(self.dataset_length_observation_step_layout)

        self.horizontal_spacer = QSpacerItem(100, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.prepare_groupbox_layout.addItem(self.horizontal_spacer)

        self.prepare_groupbox_layout.addLayout(self.dataset_seed_layout)
        self.prepare_groupbox_layout.addWidget(self.generate_button)

        self.main_layout.addWidget(self.prepare_groupbox)
        self.main_layout.addWidget(self.data_modification_groupbox)

        self.vertical_spacer = QSpacerItem(50, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.main_layout.addItem(self.vertical_spacer)
        self.number_dimensions_changed(5)

    def generate_test_data(self):
        """
        Generates the requested data depending on the given parameters.
        """
        df = pd.DataFrame()

        range_start = 0
        range_stop = int(self.dataset_length.text()) * int(
            self.dataset_length_observation_step.text())
        range_step = int(self.dataset_length_observation_step.text())
        df["observation"] = range(range_start, range_stop, range_step)

        label_list = []
        for element in self.elements:
            self.register_annotation_plot_signal.emit(element.get_name())
            data = datagenerator.generate_data(self.dataset_seed.value(), element.get_distribution(), element.get_mu(),
                                               element.get_sigma(),
                                               int(self.dataset_length.text()))

            flow = element.get_flow()
            data = datagenerator.generate_flow(self.dataset_seed.value(), data, flow)

            for novelty_generator in element.novelty_generators:
                if novelty_generator.get_groupbox_state():
                    novelties = {}

                    if isinstance(novelty_generator, datageneratorform.DataInputForm.PointNoveltyGeneratorForm):
                        number_outlier = int(len(data) * novelty_generator.get_quota())
                        novelties = datagenerator.generate_point_outliers(data,
                                                                          number_outlier,
                                                                          novelty_generator.get_range_start(),
                                                                          novelty_generator.get_range_end())

                    elif isinstance(novelty_generator, datageneratorform.DataInputForm.CollectiveNoveltyGeneratorForm):
                        number_outlier = int(len(data) * novelty_generator.get_quota())
                        novelties = datagenerator.generate_collective_outliers(data,
                                                                               number_outlier,
                                                                               novelty_generator.get_range_start(),
                                                                               novelty_generator.get_range_end())

                    elif isinstance(novelty_generator,
                                    datageneratorform.DataInputForm.MissingDataConditionChangeNoveltyGeneratorForm):
                        novelties = datagenerator.generate_condition_change_gap(data,
                                                                                novelty_generator.get_range_start(),
                                                                                novelty_generator.get_range_end())

                    for k, v in novelties.items():
                        data[k] = v

                    if novelty_generator.get_label_state():
                        label_list.append({element.get_name(): novelties})

            df[element.get_name()] = data
        self.generated_data_signal.emit(df, [])

        for label_dict in label_list:
            for plot_name, novelty_dict in label_dict.items():
                for k, v in novelty_dict.items():
                    annotations.add_label_unselected(df["observation"][k], v, k, "O", plot_name)
        self.update_labels_signal.emit()

    def number_dimensions_changed(self, val):
        """
        Slot if the selection of number of dimensions has changed.
        Add or remove new data selectors, depending on current selector count.

        Parameters
        ----------
        val
        """
        if val < len(self.elements):
            elements_to_remove = len(self.elements) - val
            for _ in range(elements_to_remove):
                self.delete_last_item_of_layout(self.data_modification_groupbox_layout)
            self.elements = self.elements[:-elements_to_remove]
        elif val > len(self.elements):
            elements_to_add = val - len(self.elements)
            for _ in range(elements_to_add):
                new = datageneratorform.DataInputForm(did=len(self.elements) + 1)
                self.elements.append(new)
                self.data_modification_groupbox_layout.addLayout(new.get_layout())

    def delete_last_item_of_layout(self, layout):
        """
        Deletes the last item of a layout and its children.

        Parameters
        ----------
        layout
        """
        if layout is not None:
            item = layout.takeAt(layout.count() - 1)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
            else:
                self.delete_items_of_layout(item.layout())

    def delete_items_of_layout(self, layout):
        """
        Deletes all items of a layout recursively.

        Parameters
        ----------
        layout
        """
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                else:
                    self.delete_items_of_layout(item.layout())
