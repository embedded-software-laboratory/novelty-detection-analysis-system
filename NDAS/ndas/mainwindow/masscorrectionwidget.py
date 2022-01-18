import sys
import traceback
from io import StringIO

import numpy as np

import pandas as pd
import time
import random
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QRunnable
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from ndas.extensions import algorithms, data, plots, physiologicallimits
from ndas.dataimputationalgorithms import base_imputation
from ndas.misc import parameter
import humanfriendly
from datetime import timedelta


class MassCorrectionWidget(QWidget):
    """
    Widget to overview the imported data and apply imputation algorithms
    """

    def __init__(self, threadpool_obj):
        super().__init__()

        self.listSelectedFiles = []
        self.SelectedFolder = ""
        self.additional_parameters_layout_list = []
        self.running_threads = {}
        self.progress = 0
        self.start_time = 0
        self.running_thread_progress_ui = {}

        self.thread_pool = threadpool_obj
        self.Baseimputator = base_imputation.BaseImputation()
        self.stacked_widget = QStackedWidget()
        self.n_button = QPushButton("Next")
        self.p_button = QPushButton("Previous")
        self.e_button = QPushButton("Early Stop")
        self.button_layout = QHBoxLayout()

        self.n_button.clicked.connect(self.__next_page)
        self.p_button.clicked.connect(self.__prev_page)
        self.e_button.clicked.connect(self.__early_stop)

        self.layout = QVBoxLayout()
        # self.layout.addStretch(2)
        self.layout.addWidget(self.stacked_widget)
        self.layout.addStretch(2)
        self.button_layout.addWidget(self.p_button)
        self.button_layout.addWidget(self.e_button)
        self.button_layout.addWidget(self.n_button)
        self.layout.addLayout(self.button_layout)
        self.setLayout(self.layout)

        self.OptionWidget = QWidget()
        self.OptionWidget_layout = QVBoxLayout()
        self.OptionWidget.setLayout(self.OptionWidget_layout)
        self.stacked_widget.addWidget(self.OptionWidget)

        self.sizepolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.sizepolicy.setHorizontalStretch(0)

        """
        Build I/O Widget
        """
        self.IOWidget = QWidget()
        self.IOWidget_layout = QHBoxLayout()
        self.IOWidget.setLayout(self.IOWidget_layout)

        self.INFiles = QGroupBox("Input-Files Settings")
        self.INFiles_layout = QVBoxLayout()

        self.INFiles_choice_layout = QHBoxLayout()
        self.INFiles_layout.addLayout(self.INFiles_choice_layout)
        self.selectedFilesString = QLabel("Selected:")
        self.selectedFilesString.setSizePolicy(self.sizepolicy)
        self.INFiles_choice_layout.addWidget(self.selectedFilesString)
        self.selectFilesButton = QPushButton("Select Files")
        self.selectFilesButton.clicked.connect(self._on_click_open_files)
        self.selectFilesButton.setSizePolicy(self.sizepolicy)
        self.INFiles_choice_layout.addWidget(self.selectFilesButton)

        self.INFiles.setLayout(self.INFiles_layout)

        self.OUTFiles = QGroupBox("Output-Files Settings")
        self.OUTFiles_layout = QVBoxLayout()

        self.OUTFiles_choice_layout = QHBoxLayout()
        self.OUTFiles_layout.addLayout(self.OUTFiles_choice_layout)
        self.selectedFolderString = QLabel("Selected Folder:")
        self.selectedFolderString.setSizePolicy(self.sizepolicy)
        self.OUTFiles_choice_layout.addWidget(self.selectedFolderString)
        self.selectFolderButton = QPushButton("Select Folder")
        self.selectFolderButton.clicked.connect(self._on_click_select_folder)
        self.selectFolderButton.setSizePolicy(self.sizepolicy)
        self.OUTFiles_choice_layout.addWidget(self.selectFolderButton)

        self.OUTFiles_ext_layout = QHBoxLayout()
        self.OUTFiles_layout.addLayout(self.OUTFiles_ext_layout)
        self.selectedExtensionString = QLabel("Select a suffix (Example: )")
        self.selectedExtensionString.setSizePolicy(self.sizepolicy)
        self.OUTFiles_ext_layout.addWidget(self.selectedExtensionString)
        self.selectExtension = QLineEdit()
        self.selectExtension.textChanged.connect(lambda ext: self._on_extension_changed(ext))
        self.selectExtension.setText("_corrected")
        self.selectExtension.setSizePolicy(self.sizepolicy)
        self.OUTFiles_ext_layout.addWidget(self.selectExtension)

        self.OUTMask_layout = QHBoxLayout()
        self.OUTFiles_layout.addLayout(self.OUTMask_layout)
        self.selectedMaskString = QLabel("Also export mask of imputed values?")
        self.selectedMaskString.setSizePolicy(self.sizepolicy)
        self.OUTMask_layout.addWidget(self.selectedMaskString)
        self.selectMask = QCheckBox("")
        self.selectMask.setSizePolicy(self.sizepolicy)
        self.OUTMask_layout.addWidget(self.selectMask)

        self.OUTFiles.setLayout(self.OUTFiles_layout)

        # self.IOWidget_layout.addStretch(0)
        self.IOWidget_layout.addWidget(self.INFiles)
        # self.IOWidget_layout.addStretch(0)
        self.IOWidget_layout.addWidget(self.OUTFiles)
        # self.IOWidget_layout.addStretch(0)
        self.OptionWidget_layout.addWidget(self.IOWidget)

        """
        Build Algorithms Widget
        """
        self.AlgWidget = QWidget()
        self.AlgWidget_layout = QHBoxLayout()
        self.AlgWidget.setLayout(self.AlgWidget_layout)

        self.DetAlgs = QGroupBox("Detection-Algorithms Settings")
        self.DetAlgs_layout = QVBoxLayout()
        self.DetAlgs.setLayout(self.DetAlgs_layout)

        self.DetAlgs_choice_layout = QHBoxLayout()
        self.DetAlgs_layout.addLayout(self.DetAlgs_choice_layout)
        self.selectedDAlgString = QLabel("Select Detection Algorithm:")
        self.selectedDAlgString.setSizePolicy(self.sizepolicy)
        self.DetAlgs_choice_layout.addWidget(self.selectedDAlgString)

        self.DetAlgs_settings = QGroupBox("Algorithm Settings")
        self.DetAlgs_settings_layout = QVBoxLayout()
        self.DetAlgs_settings.setLayout(self.DetAlgs_settings_layout)

        self.selectDAlgBox = QComboBox()
        self.selectDAlgBox.currentTextChanged.connect(lambda selection: self._update_options(selection))
        for algorithm in algorithms.get_available_algorithms():
            self.selectDAlgBox.addItem(algorithm)
        self.selectDAlgBox.setSizePolicy(self.sizepolicy)
        self.DetAlgs_choice_layout.addWidget(self.selectDAlgBox)

        self.DetAlgs_layout.addWidget(self.DetAlgs_settings)

        self.ImpAlgs = QGroupBox("Imputation-Algorithms Settings")
        self.ImpAlgs_layout = QVBoxLayout()
        self.ImpAlgs.setLayout(self.ImpAlgs_layout)

        self.ImpAlgs_choice_layout = QHBoxLayout()
        self.ImpAlgs_layout.addLayout(self.ImpAlgs_choice_layout)
        self.selectedIAlgString = QLabel("Select Imputation Algorithm:")
        self.selectedIAlgString.setSizePolicy(self.sizepolicy)
        self.ImpAlgs_choice_layout.addWidget(self.selectedIAlgString)
        self.selectIAlgBox = QComboBox()
        for key in self.Baseimputator.Methods.keys():
            self.selectIAlgBox.addItem(key)
        self.selectIAlgBox.setSizePolicy(self.sizepolicy)
        self.ImpAlgs_choice_layout.addWidget(self.selectIAlgBox)

        # self.AlgWidget_layout.addStretch(0)
        self.AlgWidget_layout.addWidget(self.DetAlgs)
        # self.AlgWidget_layout.addStretch(0)
        self.AlgWidget_layout.addWidget(self.ImpAlgs)
        # self.AlgWidget_layout.addStretch(0)
        self.OptionWidget_layout.addWidget(self.AlgWidget)

        """
        Build Progress Widget
        """
        self.number_to_be_executed = 0
        self.number_executed = 0
        self.ProgWidget = QWidget()
        self.ProgWidget_layout = QVBoxLayout()
        self.ProgWidget.setLayout(self.ProgWidget_layout)
        self.Progress_label = QLabel("Processed Files:")
        self.Progress_bar = QProgressBar()
        self.Estimated_time_rem = QLabel("Time Remaining: Not started yet")

        self.Individual_Progress_Box = QGroupBox("Individual Progress")
        self.Individual_Progress_Box_layout = QVBoxLayout()
        self.Individual_Progress_Box.setLayout(self.Individual_Progress_Box_layout)
        self.Individual_Progress_Box_internal = QWidget()
        self.Individual_Progress_Box_internal_layout = QGridLayout()
        self.Individual_Progress_Box_internal.setLayout(self.Individual_Progress_Box_internal_layout)
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.Individual_Progress_Box_internal)
        self.scroll.setWidgetResizable(True)
        #self.scroll.setFixedHeight(200)
        self.Individual_Progress_Box_layout.addWidget(self.scroll)

        self.Errors_label = QLabel("Errors:")
        self.ProgWidget_layout.addWidget(self.Progress_label)
        self.ProgWidget_layout.addWidget(self.Progress_bar)
        self.ProgWidget_layout.addWidget(self.Estimated_time_rem)
        self.ProgWidget_layout.addWidget(self.Individual_Progress_Box)
        self.ProgWidget_layout.addWidget(self.Errors_label)
        self.stacked_widget.addWidget(self.ProgWidget)

        self.__set_button_text(self.stacked_widget.currentIndex())

    def __next_page(self):
        idx = self.stacked_widget.currentIndex()
        if idx < self.stacked_widget.count() - 1:
            self.stacked_widget.setCurrentIndex(idx + 1)
        else:
            self.stacked_widget.setCurrentIndex(0)
        self.__set_button_text(self.stacked_widget.currentIndex())

    def __prev_page(self):
        idx = self.stacked_widget.currentIndex()
        if idx > 0:
            self.stacked_widget.setCurrentIndex(idx - 1)
        else:
            self.stacked_widget.setCurrentIndex(self.stacked_widget.count() - 1)
        self.__set_button_text(self.stacked_widget.currentIndex())

    def __early_stop(self):
        self.number_to_be_executed = self.number_executed + self.thread_pool.activeThreadCount()
        self.thread_pool.clear()
        self.update_done_label()

    def __set_button_text(self, idx):
        self.p_button.setDisabled(False)
        self.n_button.setDisabled(False)
        self.e_button.setDisabled(True)
        self.n_button.setText("Next")
        if idx == 0:
            self.p_button.setDisabled(True)
        if idx == (self.stacked_widget.count() - 1):
            self.p_button.setDisabled(True)
            self.n_button.setDisabled(True)
            self.e_button.setDisabled(False)
            self.n_button.setText("Running...")
            self.number_executed = 0
            self.number_to_be_executed = len(self.listSelectedFiles)
            self.running_threads = {}
            self.running_thread_progress_ui = {}
            self.progress = 0
            self.update_done_label()
            self.Errors_label.setText("Errors: ")
            self.Estimated_time_rem.setText("Time Remaining: calculating...")
            self.start_time = time.time()
            for filename in self.listSelectedFiles:
                self.running_threads[filename] = 0
                out_filename = self.SelectedFolder + "/" + (filename.split("/")[-1]).split(".")[0] + self.selectExtension.text().strip() + ".csv"
                worker = ExecuteOneDatasetCorrection(filename, out_filename, self.selectMask.isChecked(), self.selectIAlgBox.currentText(), self.selectDAlgBox.currentText(), self._get_param_list())
                worker.signals.done.connect(self.done_fn)
                worker.signals.error.connect(self.error_fn)
                worker.signals.progress.connect(self.progress_fn)
                self.thread_pool.start(worker)
        if idx == (self.stacked_widget.count() - 2):
            self.n_button.setText("Run (May take a while)")

    def _on_click_open_files(self):
        filenames, _ = QFileDialog.getOpenFileNames(caption="Select one or more files to open", filter="Dataset (*.csv *.hea)")
        self.listSelectedFiles = filenames
        filestringlabel = "Selected " + str(len(filenames)) + " file(s): "
        if filenames:
            filestringlabel = filestringlabel + filenames[0].split("/")[-1]
            if len(filenames) > 1:
                filestringlabel = filestringlabel + ", ..."
        self.selectedFilesString.setText(filestringlabel)
        self._on_extension_changed(self.selectExtension.text())

    def _on_click_select_folder(self):
        folderpath = QFileDialog.getExistingDirectory(caption="Select Export Directory")
        self.SelectedFolder = folderpath

        self.selectedFolderString.setText("Selected Folder: " + folderpath)

    def _on_extension_changed(self, extension):
        filename = "Examplefile.csv"
        if self.listSelectedFiles:
            filename = self.listSelectedFiles[0].split("/")[-1]
        extended_filename = filename.split(".")[0] + extension.strip() + ".csv"
        self.selectedExtensionString.setText("Select a suffix (Example: " + extended_filename + ")")

    def _update_options(self, choice_string):
        for param in self.additional_parameters_layout_list:
            for i in range(self.DetAlgs_settings_layout.count()):
                layout_item = self.DetAlgs_settings_layout.itemAt(i)
                if layout_item.layout() == param:
                    self.delete_items_of_layout(layout_item.layout())
                    self.DetAlgs_settings_layout.removeItem(layout_item)
                    break

        additional_parameters = algorithms.get_algorithm_required_arguments(choice_string)
        q_layout = QGridLayout()

        row = 0
        for arg in additional_parameters:
            q_label = QLabel(arg.argument_name + ": ")

            if arg.type == parameter.ArgumentType.INTEGER:
                q_label.setText("Value for " + q_label.text())
                q_input = QSpinBox()
                q_input.setMinimum(arg.minimum)
                q_input.setMaximum(arg.maximum)
                q_input.setValue(arg.default)
            elif arg.type == parameter.ArgumentType.FLOAT:
                q_label.setText("Value for " + q_label.text())
                q_input = QDoubleSpinBox()
                q_input.setDecimals(3)
                q_input.setValue(arg.default)
                q_input.setSingleStep(0.01)
                q_input.setMinimum(arg.minimum)
                q_input.setMaximum(arg.maximum)
            elif arg.type == parameter.ArgumentType.BOOL:
                q_label.setText("Use option " + q_label.text())
                q_input = QCheckBox()
                q_input.setChecked(arg.default)
            else:
                q_input = QLineEdit(arg.default)

            if arg.tooltip is not None:
                q_label.setToolTip(arg.tooltip)
                q_input.setToolTip(arg.tooltip)

            q_label.setSizePolicy(self.sizepolicy)
            q_input.setSizePolicy(self.sizepolicy)
            q_layout.addWidget(q_label, row, 0)
            q_layout.addWidget(q_input, row, 1)
            row += 1

        self.additional_parameters_layout_list.append(q_layout)
        self.DetAlgs_settings_layout.addLayout(q_layout)

    def _get_param_list(self):
        """
        Returns the parameter list for additional parameters of algorithms
        """
        args = {}
        for param in self.additional_parameters_layout_list:
            for i in range(self.DetAlgs_settings_layout.count()):
                layout_item = self.DetAlgs_settings_layout.itemAt(i)
                if layout_item.layout() == param:
                    for x in range(0, layout_item.layout().count(), 2):
                        label_item = layout_item.layout().itemAt(x).widget()
                        input_item = layout_item.layout().itemAt(x + 1).widget()

                        label_text = label_item.text().replace(": ", "")
                        label_text = label_text.replace("Use option ", "")
                        label_text = label_text.replace("Value for ", "")

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

    def progress_fn(self, prog_tuple):
        if self.running_threads[prog_tuple[1]] == 0:
            self.running_thread_progress_ui[prog_tuple[1]] = (QLabel(prog_tuple[1].split("/")[-1]), QProgressBar(), QLabel("importing file"))
            row = self.Individual_Progress_Box_internal_layout.rowCount()
            self.Individual_Progress_Box_internal_layout.addWidget(self.running_thread_progress_ui[prog_tuple[1]][0], row, 0)
            self.Individual_Progress_Box_internal_layout.addWidget(self.running_thread_progress_ui[prog_tuple[1]][1], row, 1)
            self.Individual_Progress_Box_internal_layout.addWidget(self.running_thread_progress_ui[prog_tuple[1]][1], row, 1)
            self.Individual_Progress_Box_internal_layout.addWidget(self.running_thread_progress_ui[prog_tuple[1]][2], row, 2)
        self.running_thread_progress_ui[prog_tuple[1]][1].setValue(prog_tuple[0])

        if prog_tuple[0] >= 97:
            self.running_thread_progress_ui[prog_tuple[1]][2].setText("exporting file..")
        elif prog_tuple[0] >= 95:
            if self.selectMask.isChecked():
                self.running_thread_progress_ui[prog_tuple[1]][2].setText("exporting mask..")
            else:
                self.running_thread_progress_ui[prog_tuple[1]][2].setText("exporting file..")
        elif prog_tuple[0] >= 85:
            self.running_thread_progress_ui[prog_tuple[1]][2].setText("imputing missing values..")
        elif prog_tuple[0] >= 75:
            self.running_thread_progress_ui[prog_tuple[1]][2].setText("removing outliers from data..")
        elif prog_tuple[0] >= 5:
            self.running_thread_progress_ui[prog_tuple[1]][2].setText("detecting outliers..")
        else:
            self.running_thread_progress_ui[prog_tuple[1]][2].setText("importing file..")

        self.progress = self.progress + (prog_tuple[0]-self.running_threads[prog_tuple[1]])/self.number_to_be_executed
        self.running_threads[prog_tuple[1]] = prog_tuple[0]
        self.Progress_bar.setValue(self.progress)
        estimated_time_left = ((100-self.progress)/self.progress)*(time.time()-self.start_time)
        if estimated_time_left > 0:
            self.Estimated_time_rem.setText("Time Remaining: about "+humanfriendly.format_timespan(timedelta(seconds=5*int(estimated_time_left/5)), max_units=2))

    def error_fn(self, err_tuple):
        self.Errors_label.setText(self.Errors_label.text()+"\n"+err_tuple[1].split("/")[-1]+" has not been processed - "+err_tuple[0])

    def done_fn(self, name):
        self.number_executed += 1
        self.update_done_label()
        self.progress = self.progress + (100-self.running_threads[name])/self.number_to_be_executed
        self.running_threads[name] = 100
        self.Progress_bar.setValue(self.progress)
        self.running_thread_progress_ui[name][0].deleteLater()
        self.running_thread_progress_ui[name][1].deleteLater()
        self.running_thread_progress_ui[name][2].deleteLater()
        if self.number_executed == self.number_to_be_executed:
            self.Progress_bar.setValue(100)
            self.Estimated_time_rem.setText("Time Remaining: All Files have been processed")
            self.n_button.setDisabled(False)
            self.n_button.setText("Done")

    def update_done_label(self):
        self.Progress_label.setText("Processed Files: "+str(self.number_executed)+"/"+str(self.number_to_be_executed))

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


class ExecuteOneDatasetCorrection(QRunnable):
    def __init__(self, filename, out_filename, export_mask, imp_algorithm_name, det_algorithm_name, additional_args, *args, **kwargs):
        super().__init__()
        self.signals = WorkerSignals()
        self.filename = filename
        self.out_filename = out_filename
        self.export_mask = export_mask
        self.imp_algorithm_name = imp_algorithm_name
        self.det_algorithm_name = det_algorithm_name
        self.additional_args = additional_args
        self.det_result = None
        self.data = None
        self.corr_data = None
        self.Baseimputator = base_imputation.BaseImputation()

    def run(self):
        """Long-running task."""
        try:
            with open(self.filename, "r") as raw_file:
                data = raw_file.read().replace('"', '').replace(";", ",")

            data = pd.read_csv(StringIO(data), dtype=np.float32, delimiter=',', na_values=['.', ''], skip_blank_lines=True)
            self.data = data.sort_values(data.columns[0])
            self.signals.progress.emit((5, self.filename))
            inc = algorithms.get_specific_algorithm_instance(self.det_algorithm_name, self.data, self.additional_args)
            inc.signals.result_signal.connect(self.set_result)
            inc.signals.error_signal.connect(self.det_error)
            inc.signals.status_signal.connect(self.det_progress)
            inc.run()
            self.corr_data = self.data
            for column in self.det_result.keys():
                novelties = self.det_result[column]
                list_of_novelty_keys = [k for k, v in novelties.items() if v == 1]
                self.corr_data.loc[:, column][self.corr_data[data.columns[0]].isin(list_of_novelty_keys)] = np.nan
            self.signals.progress.emit((85, self.filename))
            self.corr_data = self.Baseimputator.base_imputation(self.corr_data, method_string=self.imp_algorithm_name)
            self.signals.progress.emit((95, self.filename))
            if self.export_mask:
                expanded_data = self.data.set_index(data.columns[0]).reindex(index=self.corr_data[data.columns[0]], columns=[v for v in self.corr_data.columns if v != data.columns[0]]).reset_index()
                mask = ((self.corr_data == expanded_data) | ((self.corr_data != self.corr_data) & (expanded_data != expanded_data))) * -1 + 1
                if isinstance(mask, pd.DataFrame):
                    mask.to_csv(path_or_buf=(self.out_filename[:-4] + "_mask" + self.out_filename[-4:]), index=False)
                self.signals.progress.emit((97, self.filename))
            if isinstance(self.corr_data, pd.DataFrame):
                self.corr_data.to_csv(path_or_buf=self.out_filename, index=False)
            self.signals.progress.emit((100, self.filename))
        except:
            self.signals.error.emit((traceback.format_exc().splitlines()[-1], self.filename))
        finally:
            self.signals.done.emit(self.filename)

    def set_result(self, result):
        self.det_result = result

    def det_error(self, error_string):
        self.signals.error.emit((error_string, self.filename))

    def det_progress(self, det_progress):
        self.signals.progress.emit((5+det_progress*0.7, self.filename))


class WorkerSignals(QObject):
    progress = pyqtSignal(tuple)
    error = pyqtSignal(tuple)
    done = pyqtSignal(str)
