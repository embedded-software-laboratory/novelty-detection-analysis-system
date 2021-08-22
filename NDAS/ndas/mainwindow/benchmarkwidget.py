import numpy as np
import pandas as pd
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import (pyqtSignal)
from PyQt5.QtWidgets import *

from ndas.extensions import algorithms
from ndas.mainwindow import benchmarkplotwidget
from ndas.misc import datageneratorform, algorithmselectorform
from ndas.utils import datagenerator


class BenchmarkWidget(QStackedWidget):
    """
    Widget to benchmark algorithms
    """

    def __init__(self, threadpool_obj, *args, **kwargs):
        """
        Adding the pages to the StackedWidget

        Parameters
        ----------
        threadpool_obj
        args
        kwargs
        """
        super(BenchmarkWidget, self).__init__(*args, **kwargs)

        self.data_generator_settings_widget = DataSelectionStackedWidgetPage()
        self.data_generator_settings_widget.next_page_signal.connect(lambda: self.to_next_page())
        self.addWidget(self.data_generator_settings_widget)

        self.thread_pool = threadpool_obj

        self.layout = QGridLayout(self)
        self.setLayout(self.layout)

        self.algorithm_selection_widget = None
        self.result_view_widget = None

    def to_next_page(self):
        """
        Show the next page of the StackWidget
        """

        if self.currentIndex() == 0:
            self.algorithm_selection_widget = AlgorithmSelectionStackedWidgetPage()
            self.algorithm_selection_widget.next_page_signal.connect(lambda: self.to_next_page())
            self.algorithm_selection_widget.previous_page_signal.connect(lambda: self.to_previous_page())
            self.addWidget(self.algorithm_selection_widget)

        if self.currentIndex() == 1:
            self.result_view_widget = ResultStackedWidgetPage(self.thread_pool, self.data_generator_settings_widget,
                                                              self.algorithm_selection_widget)
            self.result_view_widget.restart_signal.connect(lambda: self.restart_benchmark())
            self.addWidget(self.result_view_widget)

        self.setCurrentIndex((self.currentIndex() + 1) % 3)

    def restart_benchmark(self):
        """
        Restart the benchmark process (select new data, algorithms etc)
        """
        if self.thread_pool.activeThreadCount() == 0:
            self.setCurrentIndex(0)

            if self.currentIndex() == 0:
                self.removeWidget(self.algorithm_selection_widget)
                self.removeWidget(self.result_view_widget)
                self.algorithm_selection_widget.deleteLater()
                self.result_view_widget.deleteLater()

    def to_previous_page(self):
        """
        Show the previous page of the StackWidget
        """
        if self.currentIndex() > 0:

            self.setCurrentIndex(self.currentIndex() - 1)

            if self.currentIndex() == 0:
                self.removeWidget(self.algorithm_selection_widget)
                self.algorithm_selection_widget.deleteLater()


class MasterStackedWidgetPage(QWidget):
    """
    Base class for single pages of QStackedWidget
    """
    next_page_signal = pyqtSignal()
    previous_page_signal = pyqtSignal()
    restart_signal = pyqtSignal()

    def __init__(self, parent=None):
        """
        Creates the main layout

        Parameters
        ----------
        parent
        """
        super().__init__(parent)

        self.main_layout = QGridLayout(self)
        self.setLayout(self.main_layout)

    def delete_last_item_of_layout(self, layout):
        """
        Deletes a single last item and all of its children of the given layout.
        Calls delete_items_of_layout to delete the children recursively.

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
        Deletes all children of a layout recursively

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


class DataSelectionStackedWidgetPage(MasterStackedWidgetPage):
    """
    Widget Page for selection of data generator parameters
    """

    def __init__(self, parent=None):
        """
        Create buttons and selectors

        Parameters
        ----------
        parent
        """
        super().__init__(parent)

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

        self.dataset_passes_layout = QHBoxLayout()
        self.dataset_passes_label = QLabel("Passes:")
        self.dataset_passes = QSpinBox()
        self.dataset_passes.setMinimumWidth(100)
        self.dataset_passes.setMinimum(1)
        self.dataset_passes.setMaximum(100)
        self.dataset_passes.setValue(1)
        self.dataset_passes_layout.addWidget(self.dataset_passes_label)
        self.dataset_passes_layout.addWidget(self.dataset_passes)

        self.data_modification_groupbox = QGroupBox("Data Modification")
        self.data_modification_groupbox_layout = QVBoxLayout()
        self.data_modification_groupbox.setLayout(self.data_modification_groupbox_layout)

        self.prepare_groupbox_layout.addLayout(self.number_of_dimensions_layout)
        self.prepare_groupbox_layout.addLayout(self.dataset_length_layout)
        self.prepare_groupbox_layout.addLayout(self.dataset_length_observation_step_layout)

        self.horizontal_spacer = QSpacerItem(100, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.prepare_groupbox_layout.addItem(self.horizontal_spacer)

        self.prepare_groupbox_layout.addLayout(self.dataset_passes_layout)

        self.main_layout.addWidget(self.prepare_groupbox, 0, 0, 1, 3)
        self.main_layout.addWidget(self.data_modification_groupbox, 1, 0, 1, 3)

        self.vertical_spacer = QSpacerItem(50, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.main_layout.addItem(self.vertical_spacer)
        self.number_dimensions_changed(5)

        self.next_button = QPushButton("Next")
        self.next_button.setIcon(self.style().standardIcon(getattr(QStyle, 'SP_CommandLink')))
        self.main_layout.addWidget(self.next_button, 3, 2, QtCore.Qt.AlignBottom)
        self.next_button.clicked.connect(lambda: self.next_page_signal.emit())

    def number_dimensions_changed(self, val):
        """
        Activated, if the number of dimensions changed.
        Adds or remove options for additional dimensions.

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
                new = datageneratorform.DataInputForm(did=len(self.elements) + 1, labeling_option=False, quota=True)
                self.elements.append(new)
                self.data_modification_groupbox_layout.addLayout(new.get_layout())


class AlgorithmSelectionStackedWidgetPage(MasterStackedWidgetPage):
    """
    Widget page, that allows selecting multiple algorithms to benchmark
    """

    def __init__(self, parent=None):
        """
        Create the visual layout and buttons

        Parameters
        ----------
        parent
        """
        super().__init__(parent)

        self.elements = []

        self.prepare_algorithm_groupbox = QGroupBox("Main Settings")
        self.prepare_algorithm_groupbox_layout = QHBoxLayout()
        self.prepare_algorithm_groupbox.setLayout(self.prepare_algorithm_groupbox_layout)

        self.number_of_algorithms_layout = QHBoxLayout()
        self.number_of_algorithms_label = QLabel("Number of Algorithms:")
        self.number_of_algorithms = QSpinBox()
        self.number_of_algorithms.setMinimumWidth(100)
        self.number_of_algorithms.setMinimum(1)
        self.number_of_algorithms.setMaximum(8)
        self.number_of_algorithms.setValue(3)
        self.number_of_algorithms_layout.addWidget(self.number_of_algorithms_label)
        self.number_of_algorithms_layout.addWidget(self.number_of_algorithms)
        self.number_of_algorithms.valueChanged.connect(lambda val: self.number_algorithms_changed(val))

        self.algorithm_modification_groupbox = QGroupBox("Algorithm Parameters")
        self.algorithm_modification_groupbox_layout = QVBoxLayout()
        self.algorithm_modification_groupbox.setLayout(self.algorithm_modification_groupbox_layout)

        self.prepare_algorithm_groupbox_layout.addLayout(self.number_of_algorithms_layout)

        self.algorithm_horizontal_spacer = QSpacerItem(100, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.prepare_algorithm_groupbox_layout.addItem(self.algorithm_horizontal_spacer)

        self.main_layout.addWidget(self.prepare_algorithm_groupbox, 0, 0, 1, 3)
        self.main_layout.addWidget(self.algorithm_modification_groupbox, 1, 0, 1, 3)

        self.algorithm_vertical_spacer = QSpacerItem(50, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.main_layout.addItem(self.algorithm_vertical_spacer)
        self.number_algorithms_changed(3)

        self.back_button = QPushButton("Back (Discard)")
        self.next_button = QPushButton("Run (May take a while)")
        self.next_button.setIcon(self.style().standardIcon(getattr(QStyle, 'SP_CommandLink')))
        self.main_layout.addWidget(self.back_button, 3, 1, QtCore.Qt.AlignBottom)
        self.main_layout.addWidget(self.next_button, 3, 2, QtCore.Qt.AlignBottom)
        self.next_button.clicked.connect(lambda: self.next_page_signal.emit())
        self.back_button.clicked.connect(lambda: self.previous_page_signal.emit())

    def number_algorithms_changed(self, val):
        """
        Triggered, if the number of algorithms to benchmark changes
        Adds or removes additional algorithm parameter options

        Parameters
        ----------
        val
        """
        if val < len(self.elements):
            elements_to_remove = len(self.elements) - val
            for _ in range(elements_to_remove):
                self.delete_last_item_of_layout(self.algorithm_modification_groupbox_layout)
            self.elements = self.elements[:-elements_to_remove]
        elif val > len(self.elements):
            elements_to_add = val - len(self.elements)
            for _ in range(elements_to_add):
                new = algorithmselectorform.AlgorithmInputForm(aid=len(self.elements) + 1)
                self.elements.append(new)
                self.algorithm_modification_groupbox_layout.addLayout(new.get_layout())


class ResultStackedWidgetPage(MasterStackedWidgetPage):
    """
    Widget page for the visualization of the results
    """

    def __init__(self, thread_pool, data_widget, algorithm_widget):
        """
        Adds visual elements to the widget page

        Parameters
        ----------
        thread_pool
        data_widget
        algorithm_widget
        """
        super().__init__()

        self.thread_pool = thread_pool
        algorithm_count = len(algorithm_widget.elements)

        cfg = []
        for algorithm in algorithm_widget.elements:
            cfg.append(algorithm.get_conf())

        dim_count = len(data_widget.elements)
        passes = data_widget.dataset_passes.value()
        length = data_widget.dataset_length.value()
        step = data_widget.dataset_length_observation_step.value()

        self.data_store = {}
        self.quick_plot = None

        self.table_widget = QTableWidget()
        self.table_widget.setRowCount(algorithm_count * passes * (dim_count + 1))
        self.table_widget.setColumnCount(15)
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_widget.setHorizontalHeaderItem(0, QTableWidgetItem("Name"))
        self.table_widget.setHorizontalHeaderItem(1, QTableWidgetItem("Algorithm"))
        self.table_widget.setHorizontalHeaderItem(2, QTableWidgetItem("Status"))
        self.table_widget.setHorizontalHeaderItem(3, QTableWidgetItem("Dimension"))
        self.table_widget.setHorizontalHeaderItem(4, QTableWidgetItem("Points"))
        self.table_widget.setHorizontalHeaderItem(5, QTableWidgetItem("Novelties"))
        self.table_widget.setHorizontalHeaderItem(6, QTableWidgetItem("Discovered"))
        self.table_widget.setHorizontalHeaderItem(7, QTableWidgetItem("Training"))
        self.table_widget.setHorizontalHeaderItem(8, QTableWidgetItem("TP (TPR)"))
        self.table_widget.setHorizontalHeaderItem(9, QTableWidgetItem("TN (TNR)"))
        self.table_widget.setHorizontalHeaderItem(10, QTableWidgetItem("FP (FPR)"))
        self.table_widget.setHorizontalHeaderItem(11, QTableWidgetItem("FN (FNR)"))
        self.table_widget.setHorizontalHeaderItem(12, QTableWidgetItem("F1"))
        self.table_widget.setHorizontalHeaderItem(13, QTableWidgetItem("MCC"))
        self.table_widget.setHorizontalHeaderItem(14, QTableWidgetItem("Plot"))

        self.single_run_progress_bar = []
        self.single_run_total_dimensions = []
        self.single_run_total_points = []
        self.single_run_total_novelties = []
        self.single_run_total_discovered = []
        self.single_run_total_training = []
        self.single_run_total_tp = []
        self.single_run_total_tn = []
        self.single_run_total_fp = []
        self.single_run_total_fn = []
        self.single_run_total_score = []
        self.single_run_total_mcc = []
        self.single_run_total_plot = []

        self.single_dimension_dim = {}
        self.single_dimension_points = {}
        self.single_dimension_novelties = {}
        self.single_dimension_discovered = {}
        self.single_dimension_training = {}
        self.single_dimension_true_positives = {}
        self.single_dimension_true_negatives = {}
        self.single_dimension_false_positives = {}
        self.single_dimension_false_negatives = {}
        self.single_dimension_score = {}
        self.single_dimension_mcc = {}
        self.single_dimension_view = {}

        run_index = 0
        dim_index = 0

        for single_algorithm in cfg:
            for single_pass in range(passes):

                self.data_store[run_index] = {}

                self.single_run_progress_bar.append(QProgressBar())
                self.single_run_progress_bar[run_index].setValue(0)
                self.single_run_progress_bar[run_index].setAlignment(QtCore.Qt.AlignCenter)
                self.single_run_progress_bar[run_index].setTextVisible(True)

                self.single_run_total_dimensions.append(QTableWidgetItem(""))
                self.single_run_total_points.append(QTableWidgetItem(""))
                self.single_run_total_novelties.append(QTableWidgetItem(""))
                self.single_run_total_discovered.append(QTableWidgetItem(""))
                self.single_run_total_training.append(QTableWidgetItem(""))
                self.single_run_total_tp.append(QTableWidgetItem(""))
                self.single_run_total_tn.append(QTableWidgetItem(""))
                self.single_run_total_fp.append(QTableWidgetItem(""))
                self.single_run_total_fn.append(QTableWidgetItem(""))
                self.single_run_total_score.append(QTableWidgetItem(""))
                self.single_run_total_mcc.append(QTableWidgetItem(""))
                self.single_run_total_plot.append(QTableWidgetItem(""))

                self.table_widget.setItem(run_index * dim_count + run_index, 0,
                                          QTableWidgetItem(single_algorithm['name']))
                self.table_widget.setItem(run_index * dim_count + run_index, 1,
                                          QTableWidgetItem(single_algorithm['klass']))
                self.table_widget.setCellWidget((run_index * dim_count + run_index), 2,
                                                self.single_run_progress_bar[run_index])
                self.table_widget.setItem((run_index * dim_count + run_index), 3,
                                          self.single_run_total_dimensions[run_index])
                self.table_widget.setItem((run_index * dim_count + run_index), 4,
                                          self.single_run_total_points[run_index])
                self.table_widget.setItem((run_index * dim_count + run_index), 5,
                                          self.single_run_total_novelties[run_index])
                self.table_widget.setItem((run_index * dim_count + run_index), 6,
                                          self.single_run_total_discovered[run_index])
                self.table_widget.setItem((run_index * dim_count + run_index), 7,
                                          self.single_run_total_training[run_index])
                self.table_widget.setItem((run_index * dim_count + run_index), 8, self.single_run_total_tp[run_index])
                self.table_widget.setItem((run_index * dim_count + run_index), 9, self.single_run_total_tn[run_index])
                self.table_widget.setItem((run_index * dim_count + run_index), 10, self.single_run_total_fp[run_index])
                self.table_widget.setItem((run_index * dim_count + run_index), 11, self.single_run_total_fn[run_index])
                self.table_widget.setItem((run_index * dim_count + run_index), 12,
                                          self.single_run_total_score[run_index])
                self.table_widget.setItem((run_index * dim_count + run_index), 13,
                                          self.single_run_total_mcc[run_index])
                self.table_widget.setItem((run_index * dim_count + run_index), 14,
                                          self.single_run_total_plot[run_index])
                self.color_row(self.table_widget, (run_index * dim_count + run_index), QtGui.QColor(227, 227, 227))

                self.single_dimension_dim[run_index] = {}
                self.single_dimension_points[run_index] = {}
                self.single_dimension_novelties[run_index] = {}
                self.single_dimension_discovered[run_index] = {}
                self.single_dimension_training[run_index] = {}
                self.single_dimension_true_positives[run_index] = {}
                self.single_dimension_true_negatives[run_index] = {}
                self.single_dimension_false_positives[run_index] = {}
                self.single_dimension_false_negatives[run_index] = {}
                self.single_dimension_score[run_index] = {}
                self.single_dimension_mcc[run_index] = {}
                self.single_dimension_view[run_index] = {}

                for dim_ in range(dim_count):
                    self.single_dimension_dim[run_index][dim_index] = QTableWidgetItem("")
                    self.single_dimension_points[run_index][dim_index] = QTableWidgetItem("")
                    self.single_dimension_novelties[run_index][dim_index] = QTableWidgetItem("")
                    self.single_dimension_discovered[run_index][dim_index] = QTableWidgetItem("")
                    self.single_dimension_training[run_index][dim_index] = QTableWidgetItem("")
                    self.single_dimension_true_positives[run_index][dim_index] = QTableWidgetItem("")
                    self.single_dimension_true_negatives[run_index][dim_index] = QTableWidgetItem("")
                    self.single_dimension_false_positives[run_index][dim_index] = QTableWidgetItem("")
                    self.single_dimension_false_negatives[run_index][dim_index] = QTableWidgetItem("")
                    self.single_dimension_score[run_index][dim_index] = QTableWidgetItem("")
                    self.single_dimension_mcc[run_index][dim_index] = QTableWidgetItem("")
                    self.single_dimension_view[run_index][dim_index] = QPushButton("View")
                    self.single_dimension_view[run_index][dim_index].setEnabled(False)

                    self.table_widget.setItem(run_index * dim_count + dim_index + run_index + 1, 0,
                                              QTableWidgetItem(single_algorithm['name']))
                    self.table_widget.setItem(run_index * dim_count + dim_index + run_index + 1, 1,
                                              QTableWidgetItem(single_algorithm['klass']))
                    self.table_widget.setItem(run_index * dim_count + dim_index + run_index + 1, 3,
                                              self.single_dimension_dim[run_index][dim_index])
                    self.table_widget.setItem(run_index * dim_count + dim_index + run_index + 1, 4,
                                              self.single_dimension_points[run_index][dim_index])
                    self.table_widget.setItem(run_index * dim_count + dim_index + run_index + 1, 5,
                                              self.single_dimension_novelties[run_index][dim_index])
                    self.table_widget.setItem(run_index * dim_count + dim_index + run_index + 1, 6,
                                              self.single_dimension_discovered[run_index][dim_index])
                    self.table_widget.setItem(run_index * dim_count + dim_index + run_index + 1, 7,
                                              self.single_dimension_training[run_index][dim_index])
                    self.table_widget.setItem(run_index * dim_count + dim_index + run_index + 1, 8,
                                              self.single_dimension_true_positives[run_index][dim_index])
                    self.table_widget.setItem(run_index * dim_count + dim_index + run_index + 1, 9,
                                              self.single_dimension_true_negatives[run_index][dim_index])
                    self.table_widget.setItem(run_index * dim_count + dim_index + run_index + 1, 10,
                                              self.single_dimension_false_positives[run_index][dim_index])
                    self.table_widget.setItem(run_index * dim_count + dim_index + run_index + 1, 11,
                                              self.single_dimension_false_negatives[run_index][dim_index])
                    self.table_widget.setItem(run_index * dim_count + dim_index + run_index + 1, 12,
                                              self.single_dimension_score[run_index][dim_index])
                    self.table_widget.setItem(run_index * dim_count + dim_index + run_index + 1, 13,
                                              self.single_dimension_mcc[run_index][dim_index])
                    self.table_widget.setCellWidget((run_index * dim_count + dim_index + run_index + 1), 14,
                                                    self.single_dimension_view[run_index][dim_index])

                    dim_index = (dim_index + 1) % dim_count

                run_index = run_index + 1

        run_index = 0
        for single_pass in range(passes):
            df, all_novelties = self.generate_test_data(data_widget.elements, length, step)

            for single_algorithm in cfg:
                self.run_detection(single_algorithm, df, all_novelties, run_index=run_index)
                for i in range(len(data_widget.elements) + 1):
                    self.data_store[run_index][i] = df[df.columns[i]]

                run_index = run_index + 1

        self.main_layout.addWidget(self.table_widget, 0, 0, 9, 2)

        self.back_button = QPushButton("Restart")
        self.back_button.setEnabled(False)
        self.back_button.setMaximumWidth(150)
        self.main_layout.addWidget(self.back_button, 10, 1, QtCore.Qt.AlignBottom)
        self.back_button.clicked.connect(lambda: self.restart_signal.emit())

    def generate_test_data(self, elements, length, observation_step):
        """
        Generates test data samples based on the given parameters.

        Parameters
        ----------
        elements
        length
        observation_step
        """
        df = pd.DataFrame()

        range_start = 0
        range_stop = int(length) * int(observation_step)
        range_step = int(observation_step)

        df["observation"] = range(range_start, range_stop, range_step)

        all_dim_novelties = []

        for i, element in enumerate(elements):
            all_dim_novelties.append(i)
            all_dim_novelties[i] = []

            data = datagenerator.generate_data(datagenerator.get_random_int(1337, 420420), element.get_distribution(),
                                               element.get_mu(), element.get_sigma(), int(length))

            flow = element.get_flow()
            data = datagenerator.generate_flow(datagenerator.get_random_int(1337, 420420), data, flow)

            for novelty_generator in element.novelty_generators:
                if novelty_generator.get_groupbox_state():
                    novelties = {}

                    if isinstance(novelty_generator, datageneratorform.DataInputForm.PointNoveltyGeneratorForm):
                        number_outlier = int(len(data) * novelty_generator.get_quota())
                        novelties = datagenerator.generate_point_outliers(data,
                                                                          number_outlier,
                                                                          novelty_generator.get_range_start(),
                                                                          novelty_generator.get_range_end())

                    elif (
                            isinstance(novelty_generator,
                                       datageneratorform.DataInputForm.CollectiveNoveltyGeneratorForm)):
                        number_outlier = int(len(data) * novelty_generator.get_quota())
                        novelties = datagenerator.generate_collective_outliers(data,
                                                                               number_outlier,
                                                                               novelty_generator.get_range_start(),
                                                                               novelty_generator.get_range_end())

                    elif (isinstance(novelty_generator,
                                     datageneratorform.DataInputForm.MissingDataConditionChangeNoveltyGeneratorForm)):
                        novelties = datagenerator.generate_condition_change_gap(data,
                                                                                novelty_generator.get_range_start(),
                                                                                novelty_generator.get_range_end())

                    for k, v in novelties.items():
                        data[k] = v

                    all_dim_novelties[i].append(novelties)

            df[element.get_name()] = data
        return df, all_dim_novelties

    def run_detection(self, cfg, data, all_dim_novelties, run_index):

        klass = cfg["klass"]
        parameter = cfg["parameter"]

        inc = algorithms.get_specific_algorithm_instance(klass, data, parameter)
        inc.signals.status_signal.connect(lambda val: self.progress_bar_update_slot(val, run_index))
        inc.signals.result_signal.connect(lambda val: self.process_result(val, all_dim_novelties, run_index))
        inc.signals.error_signal.connect(lambda val: self.progress_bar_error_slot(val, run_index))

        if not inc.are_required_arguments_satisfied(**parameter):
            if self._confirm_error("Error", "Missing required additional parameters to run this algorithm."):
                return

        if data is None or data.empty:
            if self._confirm_error("Error", "No data imported."):
                return

        self.thread_pool.start(inc)

    def plot_single_dimension(self, title, run_index, dim_index, novelties):
        """
        Calls the BenchmarkPlotWidget to plot a single dimension with highlighted novelties.

        Parameters
        ----------
        title
        run_index
        dim_index
        novelties
        """
        time_column = self.data_store[run_index][0]
        data_column = self.data_store[run_index][dim_index + 1]
        self.quick_plot = benchmarkplotwidget.BenchmarkPlotWindow(title, time_column, data_column, novelties)
        self.quick_plot.show()

    def progress_bar_update_slot(self, val, run_index):
        """
        Updates the progress bar to the current progress value.

        Parameters
        ----------
        val
        run_index
        """
        self.single_run_progress_bar[run_index].setValue(val)

    def progress_bar_error_slot(self, val, run_index):
        """
        Changes the progress bar to signal the error.

        Parameters
        ----------
        val
        run_index

        """
        print("ERROR: %s" % str(val))
        self.single_run_progress_bar[run_index].setFormat("ERROR")
        self.single_run_progress_bar[run_index].setValue(0)
        self.single_run_progress_bar[run_index].setStyleSheet("QProgressBar::chunk ""{""background-color: red;""}")

    def process_result(self, val, all_dim_novelties, run_index):
        """
        The result is processed by counting TP, TN, FP, FN, etc.

        Parameters
        ----------
        val
        all_dim_novelties
        run_index
        """
        dim_index = 0

        total_points = 0
        total_novelties = 0
        total_discovered = 0
        total_training = 0
        total_tp = 0
        total_tn = 0
        total_fp = 0
        total_fn = 0

        for k, v in val.items():

            if dim_index >= len(all_dim_novelties):
                continue

            concat_novelty_dict = {}
            for novelty_dict in all_dim_novelties[dim_index]:
                concat_novelty_dict.update(novelty_dict)

            remove_from_novelty_count = 0
            true_positives = 0
            true_negatives = 0
            false_positives = 0
            false_negatives = 0
            for time_offset, value in v.items():
                if (value == -1 or value == -2) and time_offset in concat_novelty_dict:
                    remove_from_novelty_count = remove_from_novelty_count + 1
                if value == 1 and time_offset in concat_novelty_dict:
                    true_positives = true_positives + 1
                elif value == 1 and time_offset not in concat_novelty_dict:
                    false_positives = false_positives + 1
                elif (value == 0 or value == 2) and time_offset in concat_novelty_dict:
                    false_negatives = false_negatives + 1
                elif (value == 0 or value == 2) and time_offset not in concat_novelty_dict:
                    true_negatives = true_negatives + 1

            self.single_dimension_dim[run_index][dim_index].setText(str(k))
            self.single_dimension_points[run_index][dim_index].setText(str(len(v)))
            total_points = total_points + len(v)

            dim_novelties = -remove_from_novelty_count
            for novelty_dict in all_dim_novelties[dim_index]:
                dim_novelties = dim_novelties + len(novelty_dict)
            self.single_dimension_novelties[run_index][dim_index].setText(str(dim_novelties))
            total_novelties = total_novelties + dim_novelties

            discovered_novelties = sum(value == 1 for value in v.values())
            self.single_dimension_discovered[run_index][dim_index].setText(str(discovered_novelties))
            total_discovered = total_discovered + discovered_novelties

            training_points = sum(value == -1 for value in v.values())
            self.single_dimension_training[run_index][dim_index].setText(str(training_points))
            total_training = total_training + training_points

            '''
            If (true positives + false negatives) = 0 then no positive cases in the input data, 
            so any analysis of this case has no information, and so no conclusion about how positive cases are handled.
            You want N/A or something similar as the ratio result, avoiding a division by zero error

            If (true positives + false positives) = 0 then all cases have been predicted to be negative: 
            this is one end of the ROC curve. Again, you want to recognise and report this possibility 
            while avoiding a division by zero error.
            @src: todo. pls google
            '''

            if true_positives == false_negatives == 0:
                dim_tpr = "n/a"
                dim_fnr = "n/a"
            else:
                dim_tpr = np.round_((true_positives / (true_positives + false_negatives)) * 100, 2)
                dim_fnr = np.round_((false_negatives / (false_negatives + true_positives)) * 100, 2)

            if true_negatives == false_positives == 0:
                dim_tnr = "n/a"
                dim_fpr = "n/a"
            else:
                dim_tnr = np.round_((true_negatives / (true_negatives + false_positives)) * 100, 2)
                dim_fpr = np.round_((false_positives / (false_positives + true_negatives)) * 100, 2)

            if true_positives == false_positives == false_negatives == 0:
                dim_score = "n/a"
            else:
                dim_score = np.round_(true_positives / (true_positives + 0.5 * (false_positives + false_negatives)), 2)

            if true_negatives + true_positives + false_negatives + false_positives == 0:
                dim_mcc = "n/a"
            else:
                dim_mcc_N = true_negatives + true_positives + false_negatives + false_positives
                dim_mcc_S = (true_positives + false_negatives) / dim_mcc_N
                dim_mcc_P = (true_positives + false_positives) / dim_mcc_N
                if (dim_mcc_S == 0 or dim_mcc_P == 0):
                    dim_mcc = "n/a"
                else:
                    dim_mcc = np.round_(((true_positives / dim_mcc_N) - dim_mcc_S * dim_mcc_P) / np.sqrt(
                        dim_mcc_P * dim_mcc_S * (1 - dim_mcc_S) * (1 - dim_mcc_P)), 2)

            self.single_dimension_true_positives[run_index][dim_index].setText(
                str(true_positives) + " (" + str(dim_tpr) + "%)")
            self.single_dimension_true_negatives[run_index][dim_index].setText(
                str(true_negatives) + " (" + str(dim_tnr) + "%)")
            self.single_dimension_false_positives[run_index][dim_index].setText(
                str(false_positives) + " (" + str(dim_fpr) + "%)")
            self.single_dimension_false_negatives[run_index][dim_index].setText(
                str(false_negatives) + " (" + str(dim_fnr) + "%)")
            self.single_dimension_score[run_index][dim_index].setText(str(dim_score))
            self.single_dimension_mcc[run_index][dim_index].setText(str(dim_mcc))

            self.single_dimension_view[run_index][dim_index].setEnabled(True)
            self.single_dimension_view[run_index][dim_index].clicked.connect(
                lambda checked, title=k, novelties=v, dim_i=dim_index, run_i=run_index: self.plot_single_dimension(
                    title, run_i, dim_i, novelties))

            total_tp = total_tp + true_positives
            total_tn = total_tn + true_negatives
            total_fp = total_fp + false_positives
            total_fn = total_fn + false_negatives

            dim_index = dim_index + 1

        if total_tp == total_fn == 0:
            tpr = "n/a"
            fnr = "n/a"
        else:
            tpr = np.round_((total_tp / (total_tp + total_fn)) * 100, 2)
            fnr = np.round_((total_fn / (total_fn + total_tp)) * 100, 2)

        if total_tn == total_fp == 0:
            tnr = "n/a"
            fpr = "n/a"
        else:
            tnr = np.round_((total_tn / (total_tn + total_fp)) * 100, 2)
            fpr = np.round_((total_fp / (total_fp + total_tn)) * 100, 2)

        if total_tp == total_fp == total_fn == 0:
            total_score = "n/a"
        else:
            total_score = np.round_(total_tp / (total_tp + 0.5 * (total_fp + total_fn)), 2)

        if total_tn + total_tp + total_fn + total_fp == 0:
            total_mcc = "n/a"
        else:
            mcc_N = total_tn + total_tp + total_fn + total_fp
            mcc_S = (total_tp + total_fn) / mcc_N
            mcc_P = (total_tp + total_fp) / mcc_N
            total_mcc = np.round_(
                ((total_tp / mcc_N) - mcc_S * mcc_P) / np.sqrt(mcc_P * mcc_S * (1 - mcc_S) * (1 - mcc_P)), 2)

        self.single_run_total_points[run_index].setText(str(total_points))
        self.single_run_total_novelties[run_index].setText(str(total_novelties))
        self.single_run_total_discovered[run_index].setText(str(total_discovered))
        self.single_run_total_training[run_index].setText(str(total_training))
        self.single_run_total_tp[run_index].setText(str(total_tp) + " (" + str(tpr) + "%)")
        self.single_run_total_tn[run_index].setText(str(total_tn) + " (" + str(tnr) + "%)")
        self.single_run_total_fp[run_index].setText(str(total_fp) + " (" + str(fpr) + "%)")
        self.single_run_total_fn[run_index].setText(str(total_fn) + " (" + str(fnr) + "%)")
        self.single_run_total_score[run_index].setText(str(total_score))
        self.single_run_total_mcc[run_index].setText(str(total_mcc))

        if self.thread_pool.activeThreadCount() == 0:
            self.back_button.setEnabled(True)

    @staticmethod
    def color_row(table, row_index, color):
        """
        Colors a row in the specified color

        Parameters
        ----------
        table
        row_index
        color
        """
        for j in range(table.columnCount()):
            if isinstance(table.item(row_index, j), QTableWidgetItem):
                table.item(row_index, j).setBackground(color)
