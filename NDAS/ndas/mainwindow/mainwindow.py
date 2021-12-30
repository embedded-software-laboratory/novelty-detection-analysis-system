import pandas as pd
from PyQt5.QtCore import (pyqtSlot, QTimer)
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QIcon
from PyQt5.QtWidgets import *

import logging
import numpy as np
import os

from ndas.extensions import algorithms, annotations, data, plots, savestate
from ndas.mainwindow import datamedicalimputationwidget, statgraphwidgets, datainspectionwidget, benchmarkwidget
from ndas.mainwindow.sshsettingswidget import SSHSettingsWindow
from ndas.mainwindow.databasesettingswidget import DatabaseSettingsWindow
from ndas.mainwindow.importdatabasewidget import ImportDatabaseWindow
from ndas.misc import loggerwidget, parameter, rangeslider
from ndas.utils import stats


class MainWindow(QMainWindow):
    """
    The main window with the GUI options
    """
    STYLESHEET = """
        #NDASProgressBar {
            min-height: 12px;
            max-height: 12px;
            border-radius: 1px;
            text-align: center;
            color: #000;
        }
        
        #NDASProgressBar::chunk {
            border-radius: 1px;
            background-color: #4e81bd;
            margin-right: 5px;
        }
    """

    def __init__(self, threadpool_obj, *args, **kwargs):
        """
        Creates the main window with all buttons and options
        Parameters
        ----------
        threadpool_obj
        args
        kwargs
        """
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle('NDAS')

        self.main_widget = QTabWidget()
        self.tab_annotation = QWidget()
        self.tab_annotation.setAutoFillBackground(True)
        self.tab_datamedimputation = datamedicalimputationwidget.DataMedicalImputationWidget(main_window=self)
        self.tab_datamedimputation.setAutoFillBackground(True)
        self.tab_datainspector = datainspectionwidget.DataInspectionWidget()
        self.tab_datainspector.setAutoFillBackground(True)
        # self.tab_datagenerator = datageneratorwidget.DataGeneratorWidget()

        # self.tab_datagenerator.generated_data_signal.connect( lambda df, labels: self.data_import_result_slot(df, labels))
        # self.tab_datagenerator.register_annotation_plot_signal.connect( lambda name: annotations.register_plot_annotation(name))
        # self.tab_datagenerator.update_labels_signal.connect(lambda: self.plot_layout_widget.update_labels())

        # self.tab_datagenerator.setAutoFillBackground(True)
        self.tab_benchmark = benchmarkwidget.BenchmarkWidget(threadpool_obj)
        self.tab_benchmark.setAutoFillBackground(True)
        self.tab_statistics = statgraphwidgets.StatsGraphWidget()
        self.tab_statistics.setAutoFillBackground(True)

        self.main_widget.addTab(self.tab_annotation, "Annotation")
        self.main_widget.addTab(self.tab_datamedimputation, 'Data Imputation')
        self.main_widget.addTab(self.tab_statistics, "Statistics")
        self.main_widget.addTab(self.tab_datainspector, "Data Inspector")
        # self.main_widget.addTab(self.tab_datagenerator, "Data Generator")
        self.main_widget.addTab(self.tab_benchmark, "Benchmark")

        self.main_grid = QGridLayout(self.main_widget)
        self.tab_annotation.setLayout(self.main_grid)

        self.plot_layout_widget = plots.plot_layout_widget
        self.plot_widget = self.plot_layout_widget.main_plot
        self.plot_scroll_widget = self.plot_layout_widget.nav_plot

        self.thread_pool = threadpool_obj
        self.additional_parameters_layout_list = []

        self.annotation_groupbox = QGroupBox("Annotation")
        self.annotation_groupbox_layout = QHBoxLayout()
        self.annotation_groupbox.setLayout(self.annotation_groupbox_layout)

        self.annotation_number_active_label = QLabel("Labeled: ")
        self.annotation_number_active = QLabel("0")
        self.annotation_number_selected_label = QLabel("Selected: ")
        self.annotation_number_selected = QLabel("0")
        self.annotation_active_label_label = QLabel("           Active Label:")
        self.annotation_active_label = QComboBox()
        self.annotation_active_label.currentTextChanged.connect(lambda text: self.update_annotation_options(text))

        self.annotation_active_label_sensor_label = QLabel("Type:")
        self.annotation_active_label_sensor = QComboBox()
        self.sensor_type_labels = ["", "Noise", "Tampering", "Dislocation", "Disconnection", "Stuck Value", "Miscalibration", "Other (comment)"]
        for type_string in self.sensor_type_labels:
            self.annotation_active_label_sensor.addItem(type_string)
        self.annotation_corrected_value_label = QLabel('Corrected Value:')
        self.annotation_corrected_value = QLineEdit()
        self.annotation_corrected_value.setValidator(QDoubleValidator(0, 500, 2))
        self.annotation_detailed_comment_label = QLabel('Comment:')
        self.annotation_detailed_comment = QLineEdit()

        self.annotation_add_label_btn = QPushButton(" Add")
        self.annotation_add_label_btn.setIcon(QIcon('ndas/img/plus-line.svg'))
        self.annotation_remove_label_btn = QPushButton(" Remove")
        self.annotation_remove_label_btn.setIcon(QIcon('ndas/img/minus-line.svg'))
        self.annotation_invert_btn = QPushButton(" Invert")
        self.annotation_invert_btn.setIcon(QIcon('ndas/img/convert-arrow.svg'))
        self.annotate_deselect_btn = QPushButton(" Deselect")
        self.annotate_deselect_btn.setIcon(QIcon('ndas/img/close-line.svg'))

        self.annotation_label_layout = QHBoxLayout()
        self.annotation_label_layout.addWidget(self.annotation_number_active_label)
        self.annotation_label_layout.addWidget(self.annotation_number_active)
        self.annotation_label_layout.addWidget(self.annotation_number_selected_label)
        self.annotation_label_layout.addWidget(self.annotation_number_selected)
        self.annotation_groupbox_layout.addLayout(self.annotation_label_layout)

        self.annotation_spacer = QSpacerItem(20, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.annotation_groupbox_layout.addItem(self.annotation_spacer)

        self.annotation_groupbox_layout.addWidget(self.annotation_active_label_label)
        self.annotation_groupbox_layout.addWidget(self.annotation_active_label)
        self.annotation_groupbox_layout.addWidget(self.annotation_active_label_sensor_label)
        self.annotation_groupbox_layout.addWidget(self.annotation_active_label_sensor)
        self.annotation_groupbox_layout.addWidget(self.annotation_corrected_value_label)
        self.annotation_groupbox_layout.addWidget(self.annotation_corrected_value)
        self.annotation_groupbox_layout.addWidget(self.annotation_detailed_comment_label)
        self.annotation_groupbox_layout.addWidget(self.annotation_detailed_comment)

        self.annotation_groupbox_layout.addWidget(self.annotation_add_label_btn)
        self.annotation_groupbox_layout.addWidget(self.annotation_remove_label_btn)
        self.annotation_groupbox_layout.addWidget(self.annotation_invert_btn)
        self.annotation_groupbox_layout.addWidget(self.annotate_deselect_btn)

        self.novelty_selection_groupbox = QGroupBox("Novelty Selection")
        self.novelty_selection_groupbox_layout = QHBoxLayout()
        self.novelty_selection_groupbox.setLayout(self.novelty_selection_groupbox_layout)

        self.novelty_selection_number_active_label = QLabel("Marked: ")
        self.novelty_selection_number_active = QLabel("0")
        self.novelty_selection_number_selected_label = QLabel("Selected: ")
        self.novelty_selection_number_selected = QLabel("0")

        self.novelty_selection_add_btn = QPushButton(" Mark")
        self.novelty_selection_add_btn.setIcon(QIcon('ndas/img/plus-line.svg'))
        self.novelty_selection_remove_btn = QPushButton(" Unmark")
        self.novelty_selection_remove_btn.setIcon(QIcon('ndas/img/minus-line.svg'))
        self.novelty_selection_invert_btn = QPushButton(" Invert Marking")
        self.novelty_selection_invert_btn.setIcon(QIcon('ndas/img/convert-arrow.svg'))

        self.novelty_selection_label_layout = QHBoxLayout()
        self.novelty_selection_label_layout.addWidget(self.novelty_selection_number_active_label)
        self.novelty_selection_label_layout.addWidget(self.novelty_selection_number_active)
        self.novelty_selection_label_layout.addWidget(self.novelty_selection_number_selected_label)
        self.novelty_selection_label_layout.addWidget(self.novelty_selection_number_selected)
        self.novelty_selection_groupbox_layout.addLayout(self.novelty_selection_label_layout)

        self.novelty_selection_spacer = QSpacerItem(20, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.novelty_selection_groupbox_layout.addItem(self.novelty_selection_spacer)

        self.novelty_selection_groupbox_layout.addWidget(self.novelty_selection_add_btn)
        self.novelty_selection_groupbox_layout.addWidget(self.novelty_selection_remove_btn)
        self.novelty_selection_groupbox_layout.addWidget(self.novelty_selection_invert_btn)

        self.analysis_options_groupbox = QGroupBox("Analysis Settings")
        self.analysis_options_groupbox_layout = QVBoxLayout()
        self.analysis_options_groupbox.setLayout(self.analysis_options_groupbox_layout)

        self.active_analysis_algorithm_layout = QHBoxLayout()
        self.active_analysis_algorithm_label = QLabel("Active ND Algorithm:")
        self.active_analysis_algorithm = QComboBox()
        self.active_analysis_algorithm_layout.addWidget(self.active_analysis_algorithm_label)
        self.active_analysis_algorithm_layout.addWidget(self.active_analysis_algorithm)

        self.analysis_additional_options = QVBoxLayout()
        self.analysis_run_btn = QPushButton("Run")
        self.analysis_run_btn.setIcon(self.style().standardIcon(getattr(QStyle, 'SP_CommandLink')))
        self.analysis_remove_detected_btn = QPushButton("Remove Selected Outliers")

        self.analysis_options_groupbox_layout.addLayout(self.active_analysis_algorithm_layout)
        self.analysis_options_groupbox_layout.addLayout(self.analysis_additional_options)
        self.analysis_options_groupbox_layout.addWidget(self.analysis_run_btn)
        self.analysis_options_groupbox_layout.addWidget(self.analysis_remove_detected_btn)

        self.data_selection_groupbox = QGroupBox("Data Slicing")
        self.data_selection_groupbox_layout = QVBoxLayout()
        self.data_selection_groupbox.setLayout(self.data_selection_groupbox_layout)

        self.data_selection_start = QLineEdit()
        self.data_selection_end = QLineEdit()
        self.data_selection_start.setValidator(QIntValidator(0, 999999999))
        self.data_selection_end.setValidator(QIntValidator(0, 999999999))
        self.data_selection_start.setText("0")
        self.data_selection_end.setText("0")
        self.data_selection_start.setMaximumWidth(80)
        self.data_selection_end.setMaximumWidth(80)
        self.data_selection_start.setDisabled(True)
        self.data_selection_end.setDisabled(True)

        self.data_selection_btn = QPushButton("Slice")
        self.data_selection_reset_btn = QPushButton("Reset")
        self.data_selection_start_label = QLabel("X-Range:")
        self.data_selection_separation_label = QLabel("  -")

        self.data_selection_range_layout = QHBoxLayout()
        self.data_selection_range_layout.addWidget(self.data_selection_start_label)
        self.data_selection_range_layout.addWidget(self.data_selection_start)
        self.data_selection_range_layout.addWidget(self.data_selection_separation_label)
        self.data_selection_range_layout.addWidget(self.data_selection_end)

        self.data_selection_slider = rangeslider.RangeSlider()
        self.data_selection_slider.setDisabled(True)

        self.data_selection_btn_layout = QHBoxLayout()
        self.data_selection_btn_layout.addWidget(self.data_selection_btn)
        self.data_selection_btn_layout.addWidget(self.data_selection_reset_btn)

        self.data_selection_groupbox_layout.addWidget(self.data_selection_slider)
        self.data_selection_groupbox_layout.addLayout(self.data_selection_range_layout)
        self.data_selection_groupbox_layout.addLayout(self.data_selection_btn_layout)

        self.reset_view_btn = QPushButton("Reset View")
        self.visual_options_groupbox = QGroupBox("Visualization")
        self.visual_options_groupbox_layout = QVBoxLayout()
        self.visual_options_groupbox.setLayout(self.visual_options_groupbox_layout)

        self.toggle_vline_btn = QCheckBox("Enable V-Line")
        self.toggle_hline_btn = QCheckBox("Enable H-Line")
        self.toggle_plot_lines = QCheckBox("Show Curves")
        self.toggle_plot_points = QCheckBox("Show Points")
        self.toggle_plot_points.setChecked(1)
        self.toggle_plot_points_layout = QHBoxLayout()
        self.toggle_plot_points_layout.addWidget(self.toggle_plot_points)
        self.toggle_plot_points_layout.addWidget(self.toggle_plot_lines)

        self.coordinate_label = QLabel("Pointer location:")
        self.x_label = QLabel("x=NaN")
        self.y_label = QLabel("y=NaN")
        self.fps_label_label = QLabel("Generating:")
        self.fps_label = QLabel("0 fps")

        self.vh_lines_layout = QHBoxLayout()
        self.vh_lines_layout.addWidget(self.toggle_vline_btn)
        self.vh_lines_layout.addWidget(self.toggle_hline_btn)

        self.plot_selector_layout = QHBoxLayout()
        self.plot_selector_label = QLabel("Active Plot:")
        self.plot_selector = QComboBox()
        self.plot_selector.setDisabled(True)
        self.plot_selector_layout.addWidget(self.plot_selector_label)
        self.plot_selector_layout.addWidget(self.plot_selector)

        self.visual_options_groupbox_layout.addLayout(self.vh_lines_layout)
        self.visual_options_groupbox_layout.addLayout(self.toggle_plot_points_layout)
        self.visual_options_groupbox_layout.addLayout(self.plot_selector_layout)
        self.visual_options_groupbox_layout.addWidget(self.reset_view_btn)

        self.statistic_groupbox = QGroupBox("Stats")
        self.statistic_groupbox_layout = QVBoxLayout()
        self.statistic_groupbox.setLayout(self.statistic_groupbox_layout)

        self.stats_number_of_data_points_layout = QHBoxLayout()
        self.stats_number_of_data_points_label = QLabel("Total Data Points:")
        self.stats_number_of_data_points = QLabel("-")
        self.stats_number_of_data_points_layout.addWidget(self.stats_number_of_data_points_label)
        self.stats_number_of_data_points_layout.addWidget(self.stats_number_of_data_points)

        self.stats_number_of_discovered_novelties_layout = QHBoxLayout()
        self.stats_number_of_discovered_secondary_novelties_layout = QHBoxLayout()
        self.stats_number_of_discovered_novelties_label = QLabel("Tier-I Novelties:")
        self.stats_number_of_discovered_secondary_novelties_label = QLabel("Tier-II Novelties:")
        self.stats_num_primary_novelties = QLabel("-")
        self.stats_num_secondary_novelties = QLabel("-")
        self.stats_number_of_discovered_secondary_novelties_layout.addWidget(
            self.stats_number_of_discovered_secondary_novelties_label)
        self.stats_number_of_discovered_secondary_novelties_layout.addWidget(self.stats_num_secondary_novelties)
        self.stats_number_of_discovered_novelties_layout.addWidget(self.stats_number_of_discovered_novelties_label)
        self.stats_number_of_discovered_novelties_layout.addWidget(self.stats_num_primary_novelties)

        self.stats_data_range_layout = QHBoxLayout()
        self.stats_data_range_label = QLabel("Range:")
        self.stats_data_range = QLabel("-")
        self.stats_data_range_layout.addWidget(self.stats_data_range_label)
        self.stats_data_range_layout.addWidget(self.stats_data_range)

        self.stats_mean_layout = QHBoxLayout()
        self.stats_mean_label = QLabel("Mean μ:")
        self.stats_mean = QLabel("-")
        self.stats_mean_layout.addWidget(self.stats_mean_label)
        self.stats_mean_layout.addWidget(self.stats_mean)

        self.stats_std_layout = QHBoxLayout()
        self.stats_std_label = QLabel("Standard Deviation σ:")
        self.stats_std = QLabel("-")
        self.stats_std_layout.addWidget(self.stats_std_label)
        self.stats_std_layout.addWidget(self.stats_std)

        self.stats_mad_layout = QHBoxLayout()
        self.stats_mad_label = QLabel("Median Absolute Deviation:")
        self.stats_mad = QLabel("-")
        self.stats_mad_layout.addWidget(self.stats_mad_label)
        self.stats_mad_layout.addWidget(self.stats_mad)

        self.statistic_groupbox_layout.addLayout(self.stats_number_of_data_points_layout)
        self.statistic_groupbox_layout.addLayout(self.stats_number_of_discovered_novelties_layout)
        self.statistic_groupbox_layout.addLayout(self.stats_number_of_discovered_secondary_novelties_layout)
        self.statistic_groupbox_layout.addLayout(self.stats_data_range_layout)
        self.statistic_groupbox_layout.addLayout(self.stats_mean_layout)
        self.statistic_groupbox_layout.addLayout(self.stats_std_layout)
        self.statistic_groupbox_layout.addLayout(self.stats_mad_layout)

        self.logger_box = QGroupBox("Logging")
        self.logger_box_layout = QHBoxLayout()
        self.logger_box.setLayout(self.logger_box_layout)
        self.logger_widget = loggerwidget.QPlainTextEditLogger()
        self.logger_box_layout.addWidget(self.logger_widget)

        self.status_bar = QStatusBar()
        self.vertical_spacer = QSpacerItem(200, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.central_layout = QHBoxLayout()
        self.btn_layout = QVBoxLayout()
        self.progress_bar_layout = QVBoxLayout()

        self._add_widgets()
        self._add_algorithms()
        self._add_labels()
        self._add_progressbar()
        self._add_statusbar()
        self._add_menu()

        self._connect_signals()
        self.setCentralWidget(self.main_widget)

    def _add_algorithms(self):
        """
        Adds algorithms to the list of available algorithms
        """
        algorithm_list = algorithms.get_available_algorithms()
        for algorithm in algorithm_list:
            self.active_analysis_algorithm.addItem(algorithm)
        self.change_algorithm_additional_parameters()

    def _add_labels(self):
        """
        Adds labels to the list of available labels.
        """
        label_list = annotations.get_available_labels()
        for label in label_list:
            self.annotation_active_label.addItem(str(label))

    def _add_progressbar(self):
        """
        Adds the progress bar to the GUI
        """
        self.progress_bar = QProgressBar(self, objectName="NDASProgressBar", minimum=0, maximum=100)
        self.status_bar.addPermanentWidget(self.progress_bar, stretch=100)
        self.progress_bar.hide()

    def _add_menu(self):
        """
        Adds the menu to the GUI
        """
        self.main_menu = self.menuBar()
        self.file_menu = self.main_menu.addMenu('&Import')

        open_csv_action = QAction("Import csv", self)
        open_csv_action.triggered.connect(lambda: self.fm_open_csv_action())
        self.file_menu.addAction(open_csv_action)
        open_wfm_wave_action = QAction("Import waveform", self)
        open_wfm_wave_action.triggered.connect(lambda: self.fm_open_wfm_wave_action())
        self.file_menu.addAction(open_wfm_wave_action)
        open_wfm_numeric_action = QAction("Import waveform (numeric)", self)
        open_wfm_numeric_action.triggered.connect(lambda: self.fm_open_wfm_numeric_action())
        self.file_menu.addAction(open_wfm_numeric_action)
        import_patient_action = QAction("Import patient from database", self)
        import_patient_action.triggered.connect(lambda: self.fm_import_patient_action())
        self.file_menu.addAction(import_patient_action)

        open_ndas_action = QAction("Load", self)
        open_ndas_action.triggered.connect(lambda: self.load_ndas_slot())
        self.main_menu.addAction(open_ndas_action)
        save_ndas_action = QAction("Save", self)
        save_ndas_action.triggered.connect(lambda: self.save_ndas_slot())
        self.main_menu.addAction(save_ndas_action)

        export_menu = self.main_menu.addMenu("Export as")
        export_png_action = QAction("PNG", self)
        export_png_action.triggered.connect(lambda: self.export_to_png())
        export_menu.addAction(export_png_action)
        export_svg_action = QAction("SVG", self)
        export_svg_action.triggered.connect(lambda: self.export_to_svg())
        export_menu.addAction(export_svg_action)
        export_mpl_action = QAction("matplotlib", self)
        export_mpl_action.triggered.connect(lambda: self.export_to_mpl())
        export_menu.addAction(export_mpl_action)
        export_csv_action = QAction("CSV", self)
        export_csv_action.triggered.connect(lambda: self.export_to_csv())
        export_menu.addAction(export_csv_action)
        export_csv_w_mask_action = QAction("CSV with Mask", self)
        export_csv_w_mask_action.triggered.connect(lambda: self.export_to_csv_w_mask())
        export_menu.addAction(export_csv_w_mask_action)

        settings_menu = self.main_menu.addMenu("Settings")

        ssh_settings_action = QAction("Configure SSH authentification data", self)
        ssh_settings_action.triggered.connect(lambda: self.change_ssh_settings())
        settings_menu.addAction(ssh_settings_action)

        database_settings_action = QAction("Configure database authentification data", self)
        database_settings_action.triggered.connect(lambda: self.change_database_settings())
        settings_menu.addAction(database_settings_action)

        self.showPointToolTips = True
        activate_tooltips = QAction("Show point tooltips", self, checkable=True)
        activate_tooltips.triggered.connect(lambda: self.toggleTooltipStatus(activate_tooltips.isChecked()))
        activate_tooltips.setChecked(True)
        settings_menu.addAction(activate_tooltips)


        help_menu = self.main_menu.addMenu('&?')
        about_action = QAction("About", self)
        about_action.triggered.connect(lambda: self.open_about_window())
        help_menu.addAction(about_action)

    def _add_widgets(self):
        """
        Adds the main widgets to the GUI
        """
        self.central_layout = QGridLayout()
        self.central_layout.addWidget(self.annotation_groupbox, 0, 0)
        self.central_layout.addWidget(self.novelty_selection_groupbox, 1, 0)
        self.central_layout.addWidget(self.plot_layout_widget, 2, 0)

        self.central_layout.addLayout(self.btn_layout, 0, 2, 3, 3)

        self.main_grid.addLayout(self.central_layout, 0, 0)
        self.main_grid.addLayout(self.progress_bar_layout, 1, 0)

        self.btn_layout.addWidget(self.fps_label)
        self.btn_layout.addWidget(self.analysis_options_groupbox)
        self.btn_layout.addWidget(self.data_selection_groupbox)
        self.btn_layout.addWidget(self.visual_options_groupbox)
        self.btn_layout.addWidget(self.statistic_groupbox)
        self.btn_layout.addWidget(self.logger_box)
        self.btn_layout.addItem(self.vertical_spacer)

    def _add_statusbar(self):
        """
        Adds the status bar to the GUI
        """
        self.status_bar.addPermanentWidget(self.x_label)
        self.status_bar.addPermanentWidget(self.y_label)
        self.status_bar.addPermanentWidget(self.fps_label)
        self.status_bar.showMessage("Ready")
        self.setStatusBar(self.status_bar)

    def save_ndas_slot(self):
        """
        Calls the save option for NDAS files
        """
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog

        save_dialog = QFileDialog()
        save_dialog.setDefaultSuffix('ndas')
        file_name, _ = save_dialog.getSaveFileName(self, "Choose save file location", "",
                                                   "NDAS Files (*.ndas)", options=options)

        if file_name:
            if file_name[-5:] != ".ndas":
                file_name = file_name + ".ndas"

            self.progress_bar_update_slot(5)
            savestate.get_current_state().set_data(data.format_for_save())
            self.progress_bar_update_slot(30)
            savestate.get_current_state().set_labels(annotations.format_for_save())
            self.progress_bar_update_slot(60)
            savestate.get_current_state().set_novelties(plots.format_for_save())
            self.progress_bar_update_slot(90)
            savestate.save_state(savestate.get_current_state(), file_name)
            self.progress_bar_update_slot(100)

    def load_ndas_slot(self):
        """
        Calls the load option for NDAS files
        """
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Choose NDAS file", "",
                                                   "NDAS Files (*.ndas)", options=options)
        if file_name:
            self.progress_bar_update_slot(15)

            if savestate.restore_state(file_name):
                self.progress_bar_update_slot(85)
                self.set_data_selection(data.data_slider_start, data.data_slider_end)
                self.data_selection_slider.setRangeLimit(data.data_slider_start, data.get_dataframe_length())
                self.data_selection_slider.valueChanged.connect(
                    lambda start_value, end_value: self.update_data_selection_text(start_value, end_value))
                self.data_selection_start.textChanged.connect(lambda: self.update_data_selection_slider())
                self.data_selection_end.textChanged.connect(lambda: self.update_data_selection_slider())
                self.update_plot_selector()
                self.update_data_selection_slider()
                datamedicalimputationwidget.DataMedicalImputationWidget.on_import_data(self.tab_datamedimputation)

            self.progress_bar_update_slot(100)

    def _connect_signals(self):
        """
        Connects available signals to the slots
        """
        self.reset_view_btn.clicked.connect(lambda: self.plot_widget.autoRange())
        self.analysis_run_btn.clicked.connect(lambda: self.run_detection())
        self.analysis_remove_detected_btn.clicked.connect(lambda: self.remove_detected_points())

        self.active_analysis_algorithm.currentIndexChanged.connect(lambda: self.change_algorithm_slot())
        self.active_analysis_algorithm.currentIndexChanged.connect(
            lambda: self.change_algorithm_additional_parameters())

        self.toggle_hline_btn.clicked.connect(
            lambda: self.plot_layout_widget.set_h_line_visibility(self.toggle_hline_btn.isChecked()))
        self.toggle_vline_btn.clicked.connect(
            lambda: self.plot_layout_widget.set_v_line_visibility(self.toggle_vline_btn.isChecked()))
        self.toggle_plot_lines.clicked.connect(lambda: self.toggle_plot_lines_slot())
        self.toggle_plot_points.clicked.connect(lambda: self.toggle_plot_points_slot())

        self.annotation_add_label_btn.clicked.connect(
            lambda: self.plot_layout_widget.label_selection(self.annotation_active_label.currentText() + '|' + self.annotation_active_label_sensor.currentText() + '|' + self.annotation_corrected_value.text() + '|' + self.annotation_detailed_comment.text()))
        self.annotation_remove_label_btn.clicked.connect(
            lambda: self.plot_layout_widget.delabel_selection())
        self.annotate_deselect_btn.clicked.connect(lambda: self.plot_layout_widget.deselect_all())
        self.annotation_invert_btn.clicked.connect(lambda: self.plot_layout_widget.invert_selection())

        self.novelty_selection_add_btn.clicked.connect(lambda: self.on_mark_selected_clicked())
        self.novelty_selection_remove_btn.clicked.connect(lambda: self.on_unmark_selected_clicked())
        self.novelty_selection_invert_btn.clicked.connect(lambda: self.on_invert_marking_selected_clicked())

        self.plot_layout_widget.mouse_moved_signal.connect(lambda x, y: self.mouse_moved_slot(x, y))
        self.plot_layout_widget.fps_updated_signal.connect(lambda fps: self.fps_updated_slot(fps))
        self.plot_layout_widget.point_selection_changed_signal.connect(
            lambda num_selected: self.point_selection_changed_slot(num_selected))
        self.plot_layout_widget.point_labeling_changed_signal.connect(
            lambda num_labeled: self.point_labeling_changed_slot(num_labeled))
        self.plot_selector.currentTextChanged.connect(lambda: self.load_plot())

        self.data_selection_btn.clicked.connect(lambda: self.slice_data_slot())
        self.data_selection_reset_btn.clicked.connect(lambda: self.reset_data_slice_slot())

    def delete_items_of_layout(self, layout):
        """
        Deletes the items of the given layout
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

    @pyqtSlot()
    def toggle_plot_lines_slot(self):
        """
        Toggles the visibility of the connecting lines
        """
        current_plot_selection = self.plot_selector.currentText().split()[0]
        current_pl_checkbox_state = self.toggle_plot_lines.isChecked()
        plots.set_plot_line_status(current_plot_selection, current_pl_checkbox_state)
        plots.plot_layout_widget.set_line_item_visibility(current_pl_checkbox_state)

    @pyqtSlot()
    def toggle_plot_points_slot(self):
        """
        Toggles the visibility of the scatter plot points
        """
        current_plot_selection = self.plot_selector.currentText().split()[0]
        current_pp_checkbox_state = self.toggle_plot_points.isChecked()
        plots.set_plot_point_status(current_plot_selection, current_pp_checkbox_state)
        plots.plot_layout_widget.set_point_item_visibility(current_pp_checkbox_state)

    @pyqtSlot()
    def load_plot(self):
        """
        Loads the currently selected active plot
        """
        if not self.plot_selector.currentText().split():
            return

        if not self.plot_selector.currentText().split()[0]:
            plots.set_plot_active("")

        if isinstance(self.plot_selector.currentText().split()[0], str):
            current_plot_selection = self.plot_selector.currentText().split()[0]
            current_pp_checkbox_state = self.toggle_plot_points.isChecked()
            current_pl_checkbox_state = self.toggle_plot_lines.isChecked()
            plots.set_plot_point_status(current_plot_selection, current_pp_checkbox_state)
            plots.set_plot_line_status(current_plot_selection, current_pl_checkbox_state)
            plots.set_plot_active(self.plot_selector.currentText().split()[0])

        else:
            for plot_name in self.plot_selector.currentText().split()[0]:
                plots.set_plot_active(plot_name)

        self.update_statistics()

    @pyqtSlot()
    def progress_bar_update_slot(self, val: int):
        """
        Updates the progress bar slot with the current status percentage
        Parameters
        ----------
        val
        """
        self.progress_bar.show()
        self.progress_bar.setValue(val)
        if val >= 100:
            self.timer = QTimer(self, timeout=self.on_progress_bar_timeout)
            self.timer.start(1000)

    @pyqtSlot()
    def error_msg_slot(self, val: str):
        """
        Shows a error msg
        Parameters
        ----------
        val
        """
        self._confirm_error("Error", val)

    @pyqtSlot()
    def change_algorithm_additional_parameters(self):
        """
        Changes the currently visible additional parameters based on algorithm selection
        """
        for param in self.additional_parameters_layout_list:
            for i in range(self.analysis_additional_options.count()):
                layout_item = self.analysis_additional_options.itemAt(i)
                if layout_item.layout() == param:
                    self.delete_items_of_layout(layout_item.layout())
                    self.analysis_additional_options.removeItem(layout_item)
                    break

        additional_parameters = algorithms.get_instance().get_required_arguments()
        q_layout = QFormLayout()

        for arg in additional_parameters:
            q_label = QLabel(arg.argument_name + " = ")

            if arg.type == parameter.ArgumentType.INTEGER:
                q_input = QSpinBox()
                q_input.setMinimum(arg.minimum)
                q_input.setMaximum(arg.maximum)
                q_input.setValue(arg.default)
            elif arg.type == parameter.ArgumentType.FLOAT:
                q_input = QDoubleSpinBox()
                q_input.setDecimals(3)
                q_input.setValue(arg.default)
                q_input.setSingleStep(0.01)
                q_input.setMinimum(arg.minimum)
                q_input.setMaximum(arg.maximum)
            elif arg.type == parameter.ArgumentType.BOOL:
                q_input = QCheckBox()
                q_input.setChecked(arg.default)
            else:
                q_input = QLineEdit(arg.default)

            if arg.tooltip is not None:
                q_label.setToolTip(arg.tooltip)
                q_input.setToolTip(arg.tooltip)

            q_layout.addRow(q_label, q_input)

        self.additional_parameters_layout_list.append(q_layout)
        self.analysis_additional_options.addLayout(q_layout)

    def get_param_list(self):
        """
        Returns the parameter list for additional parameters of algorithms
        """
        args = {}
        for param in self.additional_parameters_layout_list:
            for i in range(self.analysis_additional_options.count()):
                layout_item = self.analysis_additional_options.itemAt(i)
                if layout_item.layout() == param:
                    for x in range(0, layout_item.layout().count(), 2):
                        label_item = layout_item.layout().itemAt(x).widget()
                        input_item = layout_item.layout().itemAt(x + 1).widget()

                        label_text = label_item.text().replace(" = ", "")

                        if isinstance(input_item, QSpinBox):
                            input_text = input_item.value()
                        elif isinstance(input_item, QDoubleSpinBox):
                            input_text = input_item.value()
                        elif isinstance(input_item, QCheckBox):
                            input_text = input_item.isChecked()
                        else:
                            input_text = input_item.text().replace(" ", "")

                        args[label_text] = input_text
        return args

    @pyqtSlot()
    def export_to_png(self):
        """
        Adds the options to export to png
        """
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog

        save_dialog = QFileDialog()
        save_dialog.setDefaultSuffix('png')
        file_name, _ = save_dialog.getSaveFileName(self, "Choose file location", "", "png (*.png)", options=options)

        if file_name:
            if file_name[-4:] != ".png":
                file_name = file_name + ".png"

            plots.plot_layout_widget.export("png", file_name)

    @pyqtSlot()
    def export_to_svg(self):
        """
        Adds the option to export to svg
        """
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog

        save_dialog = QFileDialog()
        save_dialog.setDefaultSuffix('svg')
        file_name, _ = save_dialog.getSaveFileName(self, "Choose file location", "",
                                                   "svg (*.svg)", options=options)

        if file_name:
            if file_name[-4:] != ".svg":
                file_name = file_name + ".svg"

            plots.plot_layout_widget.export("svg", file_name)

    @pyqtSlot()
    def export_to_mpl(self):
        """
        Adds the option to export to mpl
        """
        plots.plot_layout_widget.export("mpl")

    @pyqtSlot()
    def export_to_csv(self):
        """
        Exports current dataframe to csv
        """
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog

        save_dialog = QFileDialog()
        save_dialog.setDefaultSuffix('csv')
        file_name, _ = save_dialog.getSaveFileName(self, "Choose file location", "", "csv (*.csv)", options=options)

        if file_name:
            if file_name[-4:] != ".csv":
                file_name = file_name + ".csv"

            df = data.get_full_dataframe()
            if isinstance(df, pd.DataFrame):
                df.to_csv(path_or_buf=file_name, index=False)

    @pyqtSlot()
    def export_to_csv_w_mask(self):
        """
        Exports current dataframe and mask to csv
        """
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog

        save_dialog = QFileDialog()
        save_dialog.setDefaultSuffix('csv')
        file_name, _ = save_dialog.getSaveFileName(self, "Choose file location", "", "csv (*.csv)", options=options)

        if file_name:
            if file_name[-4:] != ".csv":
                file_name = file_name + ".csv"

            df = data.get_full_dataframe()
            if isinstance(df, pd.DataFrame):
                df.to_csv(path_or_buf=file_name, index=False)
            mask = data.get_mask_dataframe()
            if isinstance(mask, pd.DataFrame):
                mask.to_csv(path_or_buf=(file_name[:-4]+"_mask"+file_name[-4:]), index=False)

    @pyqtSlot()
    def change_ssh_settings(self):
        self.sshwidget = SSHSettingsWindow(self)
        self.sshwidget.show()

    @pyqtSlot()
    def change_database_settings(self):
        self.databasewidget = DatabaseSettingsWindow(self)
        self.databasewidget.show()

    @pyqtSlot()
    def toggleTooltipStatus(self, isChecked):
        plots.toggleTooltipFlag(isChecked)
        if isChecked:
            self.showPointToolTips = True
        else:
            self.showPointToolTips = False

    @pyqtSlot()
    def add_new_plot(self, name, x_data, y_data, x_lbl, y_lbl):
        """
        Adds a new plot to the plot selector
        Parameters
        ----------
        name
        x_data
        y_data
        x_lbl
        y_lbl
        """
        plots.register_plot(name, x_data, y_data, x_lbl, y_lbl)
        annotations.register_plot_annotation(name)

        self.update_plot_selector()
        plots.update_plot_view()

    @pyqtSlot()
    def change_algorithm_slot(self):
        """
        Changes the currently loaded algorithm implementation
        """
        current_selection = self.active_analysis_algorithm.currentText()
        algorithms.set_specific_algorithm_instance(current_selection, data.get_dataframe())

    @pyqtSlot()
    def run_detection(self):
        """
        Starts the detection
        """
        selected_algorithm = self.active_analysis_algorithm.currentText()
        if not selected_algorithm:
            if self._confirm_error("Error", "No algorithm selected."):
                return

        additional_args = self.get_param_list()

        algorithms.set_specific_algorithm_instance(selected_algorithm, data.get_dataframe(), additional_args)
        algorithms.get_instance().signals.status_signal.connect(lambda val: self.progress_bar_update_slot(val))
        algorithms.get_instance().signals.error_signal.connect(lambda val: self.error_msg_slot(val))

        algorithms.get_instance().signals.add_plot_signal.connect(
            lambda name, x_data, y_data, x_label, y_label: self.add_new_plot(name, x_data, y_data, x_label, y_label))
        algorithms.get_instance().signals.add_line_signal.connect(
            lambda plt_name, name, x_data, y_data: plots.add_line(plt_name, name, x_data, y_data))
        algorithms.get_instance().signals.add_infinite_line_signal.connect(
            lambda plt_name, name, y: plots.add_infinite_line(plt_name, name, y))

        algorithms.get_instance().signals.result_signal.connect(lambda val: self.set_algorithm_result(val))

        if not algorithms.get_instance().are_required_arguments_satisfied(**additional_args):
            if self._confirm_error("Error", "Missing required additional parameters to run this algorithm."):
                return

        if data.get_dataframe() is None or data.get_dataframe().empty:
            if self._confirm_error("Error", "No data imported."):
                return

        self.analysis_run_btn.setDisabled(True)
        self.thread_pool.start(algorithms.get_instance())

        self.status_bar.clearMessage()
        self.progress_bar.show()

    @pyqtSlot()
    def remove_detected_points(self):
        columns_with_novelties = [col for col in algorithms.get_plots_with_detected_novelties() if col in plots.get_registered_plot_keys()]
        df = data.get_full_dataframe()
        time_column = data.get_dataframe_index_column()
        for column in columns_with_novelties:
            novelties = algorithms.get_detected_novelties(column)
            list_of_novelty_keys = [k for k, v in novelties.items() if v == 1]
            df.loc[:, column][df[time_column].isin(list_of_novelty_keys)] = np.nan
            for k in list_of_novelty_keys:
                novelties[k] = -9
            algorithms.set_detected_novelties(column, novelties)
            plots.add_plot_novelties(column, algorithms.get_detected_novelties(column))
        self.update_values_in_current_dataset(df)
        plots.update_plot_view()
        self.update_statistics()

    def update_values_in_current_dataset(self, df):
        data.set_dataframe(df, [])
        plots.update_available_plots()
        self.tab_datainspector.set_data(data.get_dataframe())
        # self.set_updated_novelties(algorithms.get_detected_novelties(plots.get_active_plot()[0]), plots.get_active_plot()[0])
        datamedicalimputationwidget.DataMedicalImputationWidget.on_import_data(self.tab_datamedimputation)

    def update_added_values_novelty_color(self, mask_df):
        columns = plots.get_registered_plot_keys()
        time_column = data.get_dataframe_index_column()
        for col in columns:
            list_nan_timestamps = mask_df[time_column][mask_df[col].values]
            col_novelties = algorithms.get_detected_novelties(col)
            for time in list_nan_timestamps:
                if time not in col_novelties:
                    col_novelties[time] = -8
            algorithms.set_detected_novelties(col, col_novelties)
            plots.add_plot_novelties(col, algorithms.get_detected_novelties(col))
        plots.update_plot_view()
        self.update_statistics()


    @pyqtSlot()
    def set_algorithm_result(self, val):
        """
        Sets the results of algorithms for visualization
        Parameters
        ----------
        val
        """
        self.analysis_run_btn.setDisabled(False)

        for plot_name, plot_primary_novelties in val.items():
            algorithms.set_detected_novelties(plot_name, plot_primary_novelties)
            plots.add_plot_novelties(plot_name, algorithms.get_detected_novelties(plot_name))

        plots.update_plot_view()
        self.update_statistics()

    @pyqtSlot()
    def update_statistics(self):
        """
        Reloads the statistics in the sidebar based on plot selection
        """
        _, active_plot = plots.get_active_plot()
        if active_plot:
            self.stats_number_of_data_points.setText(stats.get_number_data_points(active_plot))
            self.stats_num_primary_novelties.setText(stats.get_number_novelties(active_plot, 1))
            self.stats_num_secondary_novelties.setText(stats.get_number_novelties(active_plot, 2))
            self.novelty_selection_number_active.setText(stats.get_number_novelties(active_plot, 'all'))
            self.stats_data_range.setText(stats.get_range_dp(active_plot))
            self.stats_mean.setText(stats.get_mean(active_plot))
            self.stats_std.setText(stats.get_std(active_plot))
            self.stats_mad.setText(stats.get_median_absolute_deviation(active_plot))

    def on_progress_bar_timeout(self):
        """
        Removes the progress bar visibility after reaching 100%
        """
        if hasattr(self, 'timer'):
            self.timer.stop()
            self.timer.deleteLater()
            del self.timer
        self.progress_bar.hide()
        self.status_bar.showMessage("Ready")
        return

    @pyqtSlot()
    def fm_open_csv_action(self):
        """
        File selector to import csv files
        """
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Choose CSV File", "",
                                                   "csv Files (*.csv)", options=options)
        if file_name:
            data.set_instance("CSVImporter", file_name)
            data.get_instance().signals.result_signal.connect(
                lambda result_data, labels: self.data_import_result_slot(result_data, labels))
            data.get_instance().signals.status_signal.connect(lambda status: self.progress_bar_update_slot(status))
            data.get_instance().signals.error_signal.connect(lambda s: self.error_msg_slot(s))
            self.thread_pool.start(data.get_instance())
            logging.info("Loaded CSV-File '"+file_name+"'")

    @pyqtSlot()
    def data_import_result_slot(self, df, labels):
        """
        Show the imported data in the plot
        Parameters
        ----------
        df
        labels
        """
        data.set_dataframe(df, labels)
        data.reset_imputed_dataframe()
        data.reset_mask_dataframe()
        algorithms.reset_detected_novelties()
        plots.register_available_plots()
        annotations.register_plot_annotation()

        self.tab_datainspector.set_data(data.get_dataframe())
        self.set_data_selection(data.data_slider_start, data.data_slider_end)

        if data.get_dataframe_length() > data.truncate_size:
            self._no_confirm_info("Dataframe length exceeding size", "Dataframe length is exceeding " + str(
                data.truncate_size) + ". The dataframe set is truncated to " + str(
                data.truncate_size) + " data points and can be enlarged in the slicer settings.")

        self.data_selection_slider.setRangeLimit(data.data_slider_start, data.get_dataframe_length())
        self.data_selection_slider.valueChanged.connect(
            lambda start_value, end_value: self.update_data_selection_text(start_value, end_value))
        self.data_selection_start.textChanged.connect(lambda: self.update_data_selection_slider())
        self.data_selection_end.textChanged.connect(lambda: self.update_data_selection_slider())
        self.update_plot_selector()
        self.update_data_selection_slider()

        """
        Switch to the annotation tab if not in imputation tab
        """
        if self.main_widget.currentIndex() != 1:
            self.main_widget.setCurrentIndex(0)
        """
        Notify Imputation Tab about new data
        Notify Imputation Tab about new data
        """
        datamedicalimputationwidget.DataMedicalImputationWidget.on_import_data(self.tab_datamedimputation)

    @pyqtSlot()
    def update_annotation_options(self, string):
        """
        Sets Annotation-UI depending on selected annotation label
        """
        self.annotation_active_label_sensor_label.hide()
        self.annotation_active_label_sensor.hide()
        self.annotation_active_label_sensor.setCurrentIndex(0)
        self.annotation_corrected_value_label.hide()
        self.annotation_corrected_value.hide()
        self.annotation_corrected_value.setText("")
        self.annotation_detailed_comment.setText("")

        if string == "Sensor":
            self.annotation_active_label_sensor_label.show()
            self.annotation_active_label_sensor.show()
            self.annotation_corrected_value.show()
            self.annotation_corrected_value_label.show()
            self.annotation_detailed_comment_label.setText("Comment:")
        elif string == "Condition":
            self.annotation_detailed_comment_label.setText("Probable Condition:")
        else:
            self.annotation_corrected_value.show()
            self.annotation_corrected_value_label.show()
            self.annotation_detailed_comment_label.setText("Description:")

    @pyqtSlot()
    def update_data_selection_text(self, start_value, end_value):
        """
        Changes the selected data subset text based on slider settings
        Parameters
        ----------
        start_value
        end_value
        """
        self.data_selection_start.setText(str(start_value))
        self.data_selection_end.setText(str(end_value))

    @pyqtSlot()
    def update_data_selection_slider(self):
        """
        Updates the slider widget based on selected data subset
        """
        input_start = self.data_selection_start.text()
        input_end = self.data_selection_end.text()

        if not input_start:
            input_start = 0

        if not input_end:
            input_end = data.get_dataframe_length()

        input_start = int(input_start)
        input_end = int(input_end)

        if input_start > input_end:
            return False

        if input_start < 0:
            return False

        if input_end > data.get_dataframe_length():
            return False

        self.data_selection_slider.setRange(input_start, input_end)

    def set_data_selection(self, start, end):
        """
        Sets a specific data subset selection in the GUI
        Parameters
        ----------
        start
        end
        """
        self.data_selection_start.setText(str(start))
        self.data_selection_end.setText(str(end))
        self.data_selection_start.setEnabled(True)
        self.data_selection_end.setEnabled(True)
        self.data_selection_slider.setEnabled(True)

    @pyqtSlot()
    def slice_data_slot(self):
        """
        Slices the data
        """
        if not self.data_selection_start.text():
            start = 0
        else:
            start = int(self.data_selection_start.text())

        if not self.data_selection_end.text():
            end = data.get_dataframe_length()
        else:
            end = int(self.data_selection_end.text())

        data.set_slice(start, end)
        plots.register_available_plots()
        plots.update_plot_view()
        self.tab_datainspector.set_data(data.get_dataframe())

    @pyqtSlot()
    def reset_data_slice_slot(self):
        """
        Reset the slicing to show the complete data
        """
        start = 0
        end = data.get_dataframe_length()
        data.set_slice(start, end)
        self.data_selection_start.setText(str(start))
        self.data_selection_end.setText(str(end))
        self.data_selection_slider.setRange(start, end)

        plots.register_available_plots()
        plots.update_plot_view()

    @pyqtSlot()
    def fm_open_wfm_wave_action(self):
        """
        File selector to import waveform files
        """
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        file_names, _ = QFileDialog.getOpenFileNames(self, "Choose Waveform Files", "",
                                                     "wfm files (*.hea)", options=options)
        if file_names:
            data.set_instance("WFMImporter", file_names)
            data.get_instance().signals.result_signal.connect(
                lambda result_data, labels: self.data_import_result_slot(result_data, labels))
            data.get_instance().signals.status_signal.connect(lambda status: self.progress_bar_update_slot(status))
            data.get_instance().signals.error_signal.connect(lambda s: self.error_msg_slot(s))
            self.thread_pool.start(data.get_instance())
            logging.info("Loaded Waveform-Files '"+file_names+"'")

    @pyqtSlot()
    def fm_open_wfm_numeric_action(self):
        """
        File selector to import numeric waveform files
        """
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        file_names, _ = QFileDialog.getOpenFileNames(self, "Choose Waveform Files", "",
                                                     "wfm files (*.hea)", options=options)
        if file_names:
            data.set_instance("WFMNumericImporter", file_names)
            data.get_instance().signals.result_signal.connect(
                lambda result_data, labels: self.data_import_result_slot(result_data, labels))
            data.get_instance().signals.status_signal.connect(lambda status: self.progress_bar_update_slot(status))
            data.get_instance().signals.error_signal.connect(lambda s: self.error_msg_slot(s))
            self.thread_pool.start(data.get_instance())
            logging.info("Loaded Numeric Waveform-Files '"+file_names+"'")


    @pyqtSlot()
    def fm_import_patient_action(self):
        """
        Calls the interface to select patients directly from a database
        """

        if not os.path.exists(os.getcwd() + "\\ndas\\local_data\\sshSettings.json"):
            self._confirm_error("Error", "Please configure your ssh authentification data first.")
        elif not os.path.exists(os.getcwd() + "\\ndas\\local_data\\db_asic_scheme.json"):
            self._confirm_error("Error", "Please configure your database authentification data first.")
        else:
            self.importdatabase = ImportDatabaseWindow(self)
            self.importdatabase.show()

    def update_plot_selector(self):
        """
        Updates the available plots in the plot selector
        """
        self.plot_selector.clear()
        self.tab_statistics.plot_selector.clear()

        if not plots.registered_plots:
            self.plot_selector.setDisabled(True)
            self.tab_statistics.plot_selector.setDisabled(True)
        else:
            self.plot_selector.setDisabled(False)
            self.tab_statistics.plot_selector.setDisabled(False)
            for k, v in plots.registered_plots.items():
                self.plot_selector.addItem(plots.registered_plots[k].plot_name  + " (" +str(len(v.main_dot_plot.x_data)) + ")")
                self.tab_statistics.plot_selector.addItem(plots.registered_plots[k].plot_name)
        self.tab_benchmark.data_generator_settings_widget.update_plot_selection()

    def _confirm_error(self, title, text):
        """
        Displays a error message with confirm checkbox
        Parameters
        ----------
        title
        text
        """
        reply = QMessageBox.critical(self, title, text, QMessageBox.Ok)
        if reply != QMessageBox.Ok:
            self.analysis_run_btn.setDisabled(False)
            return False
        return True

    def _no_confirm_info(self, title, text):
        """
        Displays an error message without confirming
        Parameters
        ----------
        title
        text
        """
        QMessageBox.information(self, title, text, QMessageBox.Ok)

    def _confirm_info(self, title, text):
        """
        Displays an info message
        Parameters
        ----------
        title
        text
        """
        reply = QMessageBox.information(self, title, text, QMessageBox.Ok)
        if reply != QMessageBox.Yes:
            return False
        return True

    def _confirm_quit(self):
        """
        Asks for confirmation to quit the program
        """
        reply = QMessageBox.question(self, 'Message', "Are you sure you want to exit the program?", QMessageBox.Yes,
                                     QMessageBox.No)
        if reply != QMessageBox.Yes:
            return False
        return True

    def open_about_window(self):
        """
        Displays the about window with licensing information.
        """
        QMessageBox.about(self, "About this Software",
                          "This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.\n\nThis program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.\n\nYou should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.")

    def closeEvent(self, e):
        """
        Event that triggers the confirmation to quit the program
        Parameters
        ----------
        e
        """
        if not self._confirm_quit():
            e.ignore()
        else:
            e.accept()

    @pyqtSlot()
    def close(self):
        """
        Close function to quit the program
        """
        super().close()

    @pyqtSlot()
    def fps_updated_slot(self, fps: float):
        """
        Slot to update the number of generated fps
        Parameters
        ----------
        fps
        """
        if fps > 999:
            fps = 999

        self.fps_label.setText("Generating %i fps " % fps)

    @pyqtSlot()
    def mouse_moved_slot(self, x: float, y: float):
        """
        Slot that updates the X and Y coordinates in the statusbar if the mouse pointer is moved over the plot
        Parameters
        ----------
        x : float
            the X coordinate
        y : float
            the Y coordinate
        """
        self.x_label.setText("x=%0.03f" % x)
        self.y_label.setText("y=%0.03f" % y)

    @pyqtSlot()
    def point_selection_changed_slot(self, number_selected: int):
        """
        Updates the number of selected points
        Parameters
        ----------
        number_selected
        """
        self.annotation_number_selected.setText("%s" % number_selected)
        self.novelty_selection_number_selected.setText("%s" % number_selected)

    @pyqtSlot()
    def point_labeling_changed_slot(self, number_labeled: int):
        """
        Updates the number of labeled points
        Parameters
        ----------
        number_labeled
        """
        self.annotation_number_active.setText("%s" % number_labeled)

    def on_mark_selected_clicked(self):
        print("test")
        current_plot = plots.get_active_plot()[0]
        current_novelties = algorithms.get_detected_novelties(current_plot)
        selected_points = annotations.get_all_selection_points()
        for point in selected_points:
            current_novelties[point.x] = 1
        self.set_updated_novelties(current_novelties, current_plot)

    def on_unmark_selected_clicked(self):
        current_plot = plots.get_active_plot()[0]
        current_novelties = algorithms.get_detected_novelties(current_plot)
        selected_points = annotations.get_all_selection_points()
        for point in selected_points:
            current_novelties[point.x] = 0
        self.set_updated_novelties(current_novelties, current_plot)

    def on_invert_marking_selected_clicked(self):
        current_plot = plots.get_active_plot()[0]
        current_novelties = algorithms.get_detected_novelties(current_plot)
        selected_points = annotations.get_all_selection_points()
        for point in selected_points:
            if point.x in current_novelties:
                current_novelties[point.x] = 1 - sorted((0, current_novelties[point.x], 1))[1]
            else:
                current_novelties[point.x] = 1
        self.set_updated_novelties(current_novelties, current_plot)

    def set_updated_novelties(self, plot_primary_novelties, plot_name):
        algorithms.set_detected_novelties(plot_name, plot_primary_novelties)
        plots.add_plot_novelties(plot_name, algorithms.get_detected_novelties(plot_name))

        plots.update_plot_view()
        self.update_statistics()
