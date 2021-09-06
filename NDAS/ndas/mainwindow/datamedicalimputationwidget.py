import logging
import pandas as pd
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from ndas.misc import rangeslider, loggerwidget
from ndas.extensions import data
import pyqtgraph as pg
import qtwidgets
from ndas.dataimputationalgorithms import base_imputation


class DataMedicalImputationWidget(QWidget):
    """
    Widget to overview the imported data and apply imputation algorithms
    """

    def __init__(self, main_window):
        super().__init__()
        self.Baseimputator = base_imputation.BaseImputation()
        self.main_window = main_window
        self.Dataframe = pd.DataFrame()
        self.layout = QHBoxLayout(self)
        self.layout_left = QGridLayout()
        self.layout_left.setColumnStretch(1, 3)
        self.layout_left.addWidget((QLabel('Timeline of diagnoses:\n(hover for details)')), 0, 0, alignment=(Qt.AlignCenter))
        self.layout_right = QVBoxLayout()
        self.layout.addLayout((self.layout_left), stretch=6)
        self.layout.addLayout(self.layout_right)
        self.bar_plot = pg.PlotWidget()
        self.bar_plot.getAxis('left').setPen(pg.mkPen(None))
        self.bar_plot.getAxis('left').setTextPen(pg.mkPen(None))
        self.bar_plot.setToolTip('Hover Bars for more Information about diagnoses')
        self.bar_plot.setXRange(0, 1)
        self.bar_plot.setMouseEnabled(x=False, y=False)
        self.bar_v_line = pg.InfiniteLine(pos=(-1), angle=90, pen=(pg.mkPen(self.palette().color(QPalette.Highlight))))
        self.bar_plot.addItem(self.bar_v_line)
        self.bar_plot.scene().sigMouseMoved.connect(lambda evt: self.on_mouse_moved_over_graph(evt, -1))
        self.layout_left.addWidget(self.bar_plot, 0, 1)
        self.layout_left.setRowStretch(0, 1)
        self.Titles = [
         'Temperature (°C)', 'Heartrate (bpm)', 'Respiratory Rate (bpm)', 'Systolic Blood Pressure (mmHg)', 'Diastolic Blood Pressure (mmHg)', 'Mean Blood Pressure (mmHg)', 'Oxygen Saturation (%)', 'End Tidal Carbon Dioxide (mmHg)', 'Central Venous Pressure (mmHg)']
        self.Titles_Short = ['Temperature', 'Heartrate', 'Resp. Rate', 'pSys Blood', 'pDia Blood', 'pMean Blood', 'Oxygen Sat.', 'End Tidal CO2', 'CVP']
        self.Column_IDs = ['TempC', 'HR', 'RR', 'pSy', 'pDi', 'pMe', 'SaO2', 'etCO2', 'CVP']
        self.Graphs = []
        self.Graph_Plots = []
        self.Graph_Infos = []
        self.Graph_Info_Layouts = []
        self.GI_num_points_layouts = []
        self.GI_num_points = []
        self.GI_ranges_layouts = []
        self.GI_ranges = []
        self.GI_means_layouts = []
        self.GI_means = []
        self.GI_quartiles_layouts = []
        self.GI_quartiles = []
        self.v_lines = []
        for i in range(len(self.Titles)):
            self.Graphs.append(pg.PlotWidget())
            self.Graphs[i].setBackground('w')
            self.Graphs[i].setTitle(self.Titles[i])
            self.Graph_Plots.append(self.Graphs[i].plot([], [], pen=pg.mkPen(color=(128,
                                                                                    128,
                                                                                    128), width=1), symbol='o', symbolSize=2, symbolBrush='k'))
            self.Graphs[i].setXRange(0, 1)
            self.Graphs[i].setMouseEnabled(x=False)
            self.Graphs[i].scene().sigMouseMoved.connect(lambda evt, bound_i=i: self.on_mouse_moved_over_graph(evt, bound_i))
            self.v_lines.append(pg.InfiniteLine(pos=(-1), angle=90, pen=(pg.mkPen(self.palette().color(QPalette.Highlight)))))
        else:
            for i in range(len(self.Titles)):
                self.Graphs[i].addItem(self.v_lines[i])
                self.layout_left.addWidget(self.Graphs[i], i + 1, 1)
                self.Graph_Infos.append(QGroupBox('Stats - ' + self.Titles_Short[i]))
                self.Graph_Info_Layouts.append(QVBoxLayout())
                self.Graph_Infos[i].setLayout(self.Graph_Info_Layouts[i])
                self.GI_num_points_layouts.append(QHBoxLayout())
                self.GI_num_points.append(QLabel('-'))
                self.GI_num_points_layouts[i].addWidget((QLabel('Number of points:')), alignment=(Qt.AlignLeft))
                self.GI_num_points_layouts[i].addWidget((self.GI_num_points[i]), alignment=(Qt.AlignRight))
                self.Graph_Info_Layouts[i].addLayout(self.GI_num_points_layouts[i])
                self.GI_ranges_layouts.append(QHBoxLayout())
                self.GI_ranges.append(QLabel('-'))
                self.GI_ranges_layouts[i].addWidget((QLabel('Range of points:')), alignment=(Qt.AlignLeft))
                self.GI_ranges_layouts[i].addWidget((self.GI_ranges[i]), alignment=(Qt.AlignRight))
                self.Graph_Info_Layouts[i].addLayout(self.GI_ranges_layouts[i])
                self.GI_means_layouts.append(QHBoxLayout())
                self.GI_means.append(QLabel('-'))
                self.GI_means_layouts[i].addWidget((QLabel('Mean Value:')), alignment=(Qt.AlignLeft))
                self.GI_means_layouts[i].addWidget((self.GI_means[i]), alignment=(Qt.AlignRight))
                self.Graph_Info_Layouts[i].addLayout(self.GI_means_layouts[i])
                self.GI_quartiles_layouts.append(QHBoxLayout())
                self.GI_quartiles.append(QLabel('-'))
                self.GI_quartiles_layouts[i].addWidget((QLabel('Quartiles:')), alignment=(Qt.AlignLeft))
                self.GI_quartiles_layouts[i].addWidget((self.GI_quartiles[i]), alignment=(Qt.AlignRight))
                self.Graph_Info_Layouts[i].addLayout(self.GI_quartiles_layouts[i])
                self.layout_left.addWidget(self.Graph_Infos[i], i + 1, 0)
                self.layout_left.setRowStretch(i + 1, 1)
            else:
                self.Imputation_Settings = QGroupBox('Data Imputation Settings')
                self.Imputation_Settings_Layout = QVBoxLayout()
                self.Imputation_Settings.setLayout(self.Imputation_Settings_Layout)
                self.Imputation_Method_Layout = QGridLayout()
                self.Imputation_Method_Selector = QComboBox()
                self.Imputation_Method_Selector.addItem('')
                for key in self.Baseimputator.Methods.keys():
                    self.Imputation_Method_Selector.addItem(key)
                else:
                    self.Current_Point_Spacing = QLineEdit('5')
                    self.Current_Point_Spacing.setDisabled(True)
                    self.Current_Point_Spacing.setValidator(QIntValidator(1, 1000))
                    self.Desired_Density_Multiplier = QLineEdit('1')
                    self.Desired_Density_Multiplier.setValidator(QIntValidator(1, 30))
                    self.Imputation_Method_Layout.addWidget(QLabel('Select Method:'), 0, 0)
                    self.Imputation_Method_Layout.addWidget(self.Imputation_Method_Selector, 0, 1)
                    self.Imputation_Method_Layout.addWidget(QLabel('Current Point Spacing:'), 1, 0)
                    self.Imputation_Method_Layout.addWidget(self.Current_Point_Spacing, 1, 1)
                    self.Imputation_Method_Layout.addWidget(QLabel('Desired Density Multiplier:'), 2, 0)
                    self.Imputation_Method_Layout.addWidget(self.Desired_Density_Multiplier, 2, 1)
                    self.Imputation_Button = QPushButton('Impute')
                    self.Imputation_Button.clicked.connect(lambda: self.on_click_imputation())
                    self.Imputation_Settings_Layout.addLayout(self.Imputation_Method_Layout)
                    self.Imputation_Settings_Layout.addWidget((self.Imputation_Button), alignment=(Qt.AlignTop))
                    self.layout_right.addWidget((self.Imputation_Settings), alignment=(Qt.AlignTop))
                    self.Data_Visualization_Settings = QGroupBox('Data Visualization Settings')
                    self.Data_Visualization_Settings_Layout = QVBoxLayout()
                    self.Data_Visualization_Settings.setLayout(self.Data_Visualization_Settings_Layout)
                    self.Data_Visualization_Settings_Layout.addWidget(QLabel('Show Graph of:'))
                    self.Show_Graph_Layout = QGridLayout()
                    self.Show_Graph_Checkboxes = []
                    for i in range(len(self.Titles)):
                        widget = QCheckBox(self.Titles_Short[i])
                        widget.stateChanged.connect(lambda state, bound_i=i: self.on_toggle_show_graph(state, bound_i))
                        self.Show_Graph_Checkboxes.append(widget)
                        self.Show_Graph_Checkboxes[i].setChecked(True)
                    else:
                        for i in range(len(self.Titles)):
                            self.Show_Graph_Layout.addWidget(self.Show_Graph_Checkboxes[i], i // 3, i % 3)
                        else:
                            self.Data_Visualization_Settings_Layout.addLayout(self.Show_Graph_Layout)
                            self.Data_Visualization_Settings_Layout.addSpacerItem(QSpacerItem(0, 10))
                            self.data_selection_start = QLineEdit()
                            self.data_selection_end = QLineEdit()
                            self.data_selection_start.setValidator(QIntValidator(0, 999999999))
                            self.data_selection_end.setValidator(QIntValidator(0, 999999999))
                            self.data_selection_start.setText('0')
                            self.data_selection_end.setText('1')
                            self.data_selection_start.setMaximumWidth(80)
                            self.data_selection_end.setMaximumWidth(80)
                            self.data_selection_start.setDisabled(True)
                            self.data_selection_end.setDisabled(True)
                            self.data_selection_btn = QPushButton('Set Range')
                            self.data_selection_btn.clicked.connect(lambda: self.on_click_set_range())
                            self.data_selection_reset_btn = QPushButton('Reset Range')
                            self.data_selection_reset_btn.clicked.connect(lambda: self.on_click_reset_range())
                            self.data_selection_start_label = QLabel('X-Range:')
                            self.data_selection_separation_label = QLabel('  -')
                            self.data_selection_range_layout = QHBoxLayout()
                            self.data_selection_range_layout.addWidget((self.data_selection_start_label), alignment=(Qt.AlignLeft))
                            self.data_selection_range_layout.addWidget((self.data_selection_start), alignment=(Qt.AlignLeft))
                            self.data_selection_range_layout.addWidget((self.data_selection_separation_label), alignment=(Qt.AlignCenter))
                            self.data_selection_range_layout.addWidget((self.data_selection_end), alignment=(Qt.AlignRight))
                            self.data_selection_slider = rangeslider.RangeSlider()
                            self.data_selection_slider.setDisabled(True)
                            self.data_selection_btn_layout = QHBoxLayout()
                            self.data_selection_btn_layout.addWidget(self.data_selection_btn)
                            self.data_selection_btn_layout.addWidget(self.data_selection_reset_btn)
                            self.Data_Visualization_Settings_Layout.addWidget((self.data_selection_slider), alignment=(Qt.AlignTop))
                            self.Data_Visualization_Settings_Layout.addLayout(self.data_selection_range_layout)
                            self.Data_Visualization_Settings_Layout.addLayout(self.data_selection_btn_layout)
                            self.DataToggle = qtwidgets.Toggle(handle_color=(Qt.gray), bar_color=(Qt.lightGray), checked_color=(self.palette().color(QPalette.Highlight)))
                            self.DataToggle.setDisabled(True)
                            self.DataToggle.stateChanged.connect(lambda state: self.on_toggle_show_imputed_data(state))
                            self.DataToggle_layout = QHBoxLayout()
                            self.DataToggle_layout = QHBoxLayout()
                            self.DataToggle_layout.addWidget((QLabel('Show original Data')), alignment=(Qt.AlignCenter))
                            self.DataToggle_layout.addWidget((self.DataToggle), alignment=(Qt.AlignCenter))
                            self.DataToggle_label2 = QLabel('Show imputed Data', alignment=(Qt.AlignCenter))
                            self.DataToggle_label2.setStyleSheet('color: lightGray')
                            self.DataToggle_layout.addWidget(self.DataToggle_label2)
                            self.Data_Visualization_Settings_Layout.addLayout(self.DataToggle_layout)
                            self.layout_right.addWidget((self.Data_Visualization_Settings), alignment=(Qt.AlignTop))
                            self.Patient_Information = QGroupBox('Patient Information')
                            self.Patient_Information_Layout = QVBoxLayout()
                            self.Patient_Information.setLayout(self.Patient_Information_Layout)
                            self.Patient_Information_General = QGridLayout()
                            self.Patient_ID = QLabel('-')
                            self.Patient_Gender = QLabel('-')
                            self.Patient_Age = QLabel('-')
                            self.Patient_Ethnicity = QLabel('-')
                            self.Patient_Height = QLabel('-')
                            self.Patient_Weight = QLabel('-')
                            self.Patient_Information_General.addWidget(QLabel('ID:'), 0, 0)
                            self.Patient_Information_General.addWidget(self.Patient_ID, 0, 1)
                            self.Patient_Information_General.addWidget(QLabel('Gender:'), 1, 0)
                            self.Patient_Information_General.addWidget(self.Patient_Gender, 1, 1)
                            self.Patient_Information_General.addWidget(QLabel('Age:'), 2, 0)
                            self.Patient_Information_General.addWidget(self.Patient_Age, 2, 1)
                            self.Patient_Information_General.addWidget(QLabel('Ethnicity:'), 3, 0)
                            self.Patient_Information_General.addWidget(self.Patient_Ethnicity, 3, 1)
                            self.Patient_Information_General.addWidget(QLabel('Height (cm):'), 4, 0)
                            self.Patient_Information_General.addWidget(self.Patient_Height, 4, 1)
                            self.Patient_Information_General.addWidget(QLabel('Weight (kg):'), 5, 0)
                            self.Patient_Information_General.addWidget(self.Patient_Weight, 5, 1)
                            self.ICD_Layout = QGridLayout()
                            self.ICD_Labels = []
                            self.ICD_Strings = ['280-285', '286-287', '288-289', '390-392', '393-398', '401-405', '410-414', '415-417', '420-429', '430-438', '440-449', '451-459', '460-466', '470-478', '480-488', '490-496', '500-508', '510-519', '800-804', '805-809', '810-819', '820-829', '830-839', '840-848', '850-854', '860-869', '870-879', '880-887', '890-897', '900-904', '905-909', '910-919', '920-924', '925-929', '930-939', '940-949', '950-957', '958-959', '960-979', '980-989', '990-995', '996-999']
                            self.ICD_Descriptions = ['Anemia', 'Coagulation/hemorrhagic', 'Other diseases of the blood and blood forming organs', 'Acute rheumatic fever', 'Chronic rheumatic heart disease', 'Hypertensive disease', 'Ischemic heart disease', 'Diseases of pulmonary circulation', 'Other forms of heart disease', 'Cerebrovascular disease', 'Diseases of arteries, arterioles, and capillaries', 'Diseases of veins and lymphatics, and other diseases of circulatory system', 'Acute respiratory infections', 'Other diseases of the upper respiratory tract', 'Pneumonia and influenza', 'Chronic obstructive pulmonary disease and allied conditions', 'Pneumoconioses and other lung diseases due to external agents', 'Other diseases of respiratory system', 'Fracture of skull', 'Fracture of neck and trunk', 'Fracture of upper limb', 'Fracture of lower limb', 'Dislocation', 'Sprains and strains of joints and adjacent muscles', 'Intracranial injury, excluding those with skull fracture', 'Internal injury of thorax, abdomen, and pelvis', 'Open wound of head, neck, and trunk', 'Open wound of upper limb', 'Open wound of lower limb', 'Injury of blood vessels', 'Late effects of injuries, poisonings, toxic effects, and other external causes', 'Superficial injury', 'Contusion with intact skin surface', 'Crushing injury', 'Effects of foreign body entering through Body orifice', 'Burns', 'Injury of nerves and spinal cord', 'Certain traumatic complications and unspecified injuries', 'Poisoning by drugs, medicinal and biological substances', 'Toxic effects of substances chiefly nonmedicinal as to source', 'Other and unspecified effects of external causes', 'Complications of surgical and medical care, not elsewhere classified']
                            for i in range(len(self.ICD_Strings)):
                                self.ICD_Labels.append(QLabel(self.ICD_Strings[i]))
                                self.ICD_Labels[i].setStyleSheet('color: lightGray')
                                self.ICD_Labels[i].setToolTip(self.ICD_Descriptions[i])
                                self.ICD_Layout.addWidget(self.ICD_Labels[i], i // 5, i % 5)
                            else:
                                self.Patient_Information_Layout.addLayout(self.Patient_Information_General)
                                self.Patient_Information_Layout.addSpacerItem(QSpacerItem(0, 10))
                                self.Patient_Information_Layout.addWidget(QLabel('Diagnosis ICD-Code-Ranges: (hover for details)'))
                                self.Patient_Information_Layout.addSpacerItem(QSpacerItem(0, 2))
                                self.Patient_Information_Layout.addLayout(self.ICD_Layout)
                                self.layout_right.addWidget(self.Patient_Information)
                                self.logging = QGroupBox('Logging')
                                self.logging_layout = QVBoxLayout()
                                self.logging.setLayout(self.logging_layout)
                                self.logger_widget = loggerwidget.QPlainTextEditLogger()
                                self.logging_layout.addWidget(self.logger_widget)
                                self.layout_right.addWidget(self.logging)

    def on_click_imputation(self):
        """
        Calls Imputation Method and enables some UI elements
        """
        if not self.Imputation_Method_Selector.currentText() == '':
            self.DataToggle.setChecked(False)
            self.Dataframe = self.Baseimputator.base_imputation(data.get_full_dataframe(), int(self.Desired_Density_Multiplier.text()), int(self.Current_Point_Spacing.text()), self.Imputation_Method_Selector.currentText())
            data.set_imputed_dataframe(self.Dataframe)
            self.DataToggle.setDisabled(False)
            self.DataToggle.setChecked(True)
            self.DataToggle_label2.setStyleSheet('color: black')
        else:
            logging.info('Please select an imputation Method.')

    def on_toggle_show_graph(self, state, index):
        """
        Toggles Visibility of the Graphs
        """
        self.layout_left.setRowStretch(index + 1, int(state))
        self.Graph_Infos[index].setVisible(state)
        self.Graphs[index].setVisible(state)
        self.layout_left.update()

    def on_click_set_range(self):
        """
        Sets the Range for Data to be Visualized and calls Data Visualization
        """
        lower_limit = int(float(self.data_selection_start.text()))
        upper_limit = int(float(self.data_selection_end.text()))
        if self.Dataframe['tOffset'].empty:
            self.visualize_data(self.Dataframe)
        else:
            if lower_limit >= self.Dataframe['tOffset'].dropna().max():
                lower_index = -1
            else:
                lower_index = self.Dataframe['tOffset'].ge(lower_limit).idxmax()
            if upper_limit >= self.Dataframe['tOffset'].dropna().max():
                upper_index = -1
            else:
                upper_index = self.Dataframe['tOffset'].le(upper_limit).iloc[::-1].idxmax()
            self.visualize_data(self.Dataframe.iloc[lower_index:upper_index])
        self.bar_plot.setXRange(lower_limit, upper_limit)
        for Plot in self.Graphs:
            Plot.setXRange(lower_limit, upper_limit)

    def on_click_reset_range(self):
        """
        Resets the Range for Data to be Visualized and calls Data Visualization
        """
        self.visualize_data(self.Dataframe)
        self.data_selection_start.setText(str(self.Dataframe['tOffset'].dropna().min()))
        self.data_selection_end.setText(str(self.Dataframe['tOffset'].dropna().max()))
        self.update_data_selection_slider()
        self.bar_plot.setXRange(self.Dataframe['tOffset'].dropna().min(), self.Dataframe['tOffset'].dropna().max())
        for Plot in self.Graphs:
            Plot.autoRange()
            Plot.setXRange(self.Dataframe['tOffset'].dropna().min(), self.Dataframe['tOffset'].dropna().max())

    def on_toggle_show_imputed_data(self, state):
        """
        Toggles which Data should be displayed and calls Data Visualization
        """
        if state:
            self.Dataframe = data.get_imputed_dataframe()
        else:
            self.Dataframe = data.get_full_dataframe()
        self.on_click_set_range()

    def on_import_data(self):
        """
        Activates some UI elements and calls Data Visualization
        """
        some_data = data.get_full_dataframe()
        if not some_data.empty:
            if 'tOffset' in some_data.columns:
                self.Dataframe = some_data
                self.DataToggle.setChecked(False)
                self.DataToggle.setDisabled(True)
                self.DataToggle_label2.setStyleSheet('color: lightGray')
                self.data_selection_start.setDisabled(False)
                self.data_selection_end.setDisabled(False)
                self.data_selection_slider.setDisabled(False)
                self.data_selection_slider.setRangeLimit(self.Dataframe['tOffset'].dropna().min(), self.Dataframe['tOffset'].dropna().max())
                self.on_click_reset_range()
                self.data_selection_slider.valueChanged.connect(lambda start_value, end_value: self.update_data_selection_text(start_value, end_value))
                self.data_selection_start.textChanged.connect(lambda: self.update_data_selection_slider())
                self.data_selection_end.textChanged.connect(lambda: self.update_data_selection_slider())
                data.set_imputed_dataframe(pd.DataFrame())

    def on_mouse_moved_over_graph(self, pos, i):
        if i == -1:
            point = self.bar_plot.plotItem.vb.mapSceneToView(pos)
        else:
            point = self.Graphs[i].plotItem.vb.mapSceneToView(pos)
        self.bar_v_line.setPos(point.x())
        for v_line in self.v_lines:
            v_line.setPos(point.x())
        else:
            self.main_window.x_label.setText('y=%0.01f' % point.x())
            self.main_window.y_label.setText('')

    def visualize_data(self, update_dataframe):
        """
        Visualizes the Graphs and Patient Information

        Only Show non-empty graphs
        """
        for i in range(len(self.Column_IDs)):
            self.Show_Graph_Checkboxes[i].setChecked(not update_dataframe[self.Column_IDs[i]].isnull().all())
        else:
            length_of_data = update_dataframe['tOffset'].iloc[(-1)] - update_dataframe['tOffset'].iloc[0]
            self.bar_plot.clear()
            self.bar_plot.addItem(self.bar_v_line)
            icd_changes = {}
            for i in range(len(self.ICD_Strings)):
                local_icd_series = update_dataframe.iloc[:, 16 + i]
                if local_icd_series.sum() > 0:
                    self.ICD_Labels[i].setStyleSheet('color: black')
                    list_of_value_change_indices = local_icd_series[(local_icd_series.diff() != 0)].index.tolist()
                    if local_icd_series[list_of_value_change_indices[0]] == 0:
                        list_of_value_change_indices.remove(list_of_value_change_indices[0])
                    else:
                        for index in list_of_value_change_indices:
                            x_value = update_dataframe['tOffset'][(index - 1 + int(float(local_icd_series[index])))]
                            if str(x_value) not in icd_changes:
                                icd_changes[str(x_value)] = []
                            else:
                                icd_changes[str(x_value)] = icd_changes[str(x_value)] + [(2 * local_icd_series[index] - 1) * (i + 1)]

                else:
                    self.ICD_Labels[i].setStyleSheet('color: lightGray')
            else:
                for key in sorted(icd_changes.keys()):
                    bar = pg.BarGraphItem(x0=[int(float(key)) - length_of_data / 120], x1=[int(float(key)) + length_of_data / 120], y=[0], height=0.6, pen=(pg.mkPen('k')), brush=(self.palette().color(QPalette.Highlight)))
                    bar.setToolTip(self.list_of_icd_indices_to_tooltip_string(key, icd_changes[key]))
                    self.bar_plot.addItem(bar)
                else:
                    self.Patient_ID.setText(str(int(float(update_dataframe['ID'].iloc[0]))))
                    Gender = int(update_dataframe['gender(n m f)'].iloc[0])
                    if Gender == 1:
                        self.Patient_Gender.setText('Male')
                    elif Gender == 2:
                        self.Patient_Gender.setText('Female')
                    else:
                        self.Patient_Gender.setText('Not Specified')
                    Age = update_dataframe['age(90= >89)'].iloc[0]
                    Age_String = 'Not Specified'
                    if Age == 90:
                        Age_String = 'Above 89'
                    elif Age:
                        Age_String = str(int(float(Age)))
                    self.Patient_Age.setText(Age_String)
                    Eth = update_dataframe['ethnicity(n cauc asia hisp afram natam)'].iloc[0]
                    if Eth == 1:
                        self.Patient_Ethnicity.setText('Caucasian')
                    elif Eth == 2:
                        self.Patient_Ethnicity.setText('Asian')
                    elif Eth == 3:
                        self.Patient_Ethnicity.setText('Hispanic')
                    elif Eth == 4:
                        self.Patient_Ethnicity.setText('African American')
                    elif Eth == 5:
                        self.Patient_Ethnicity.setText('Native American')
                    else:
                        self.Patient_Ethnicity.setText('Not Specified / Other')
                    self.Patient_Height.setText('%.1f' % update_dataframe['height(cm)'].iloc[0])
                    self.Patient_Weight.setText('%.1f' % update_dataframe['weight(kg)'].iloc[0])
                    for c_id in range(len(self.Column_IDs)):
                        x_y_values = update_dataframe[['tOffset', self.Column_IDs[c_id]]].dropna()
                        self.Graph_Plots[c_id].setData(x_y_values['tOffset'].tolist(), x_y_values[self.Column_IDs[c_id]].tolist())
                        self.GI_num_points[c_id].setText(str(x_y_values[self.Column_IDs[c_id]].count()))
                        self.GI_ranges[c_id].setText('[{:.1f} - {:.1f}]'.format(x_y_values[self.Column_IDs[c_id]].min(), x_y_values[self.Column_IDs[c_id]].max()))
                        self.GI_means[c_id].setText('{:.2f}'.format(x_y_values[self.Column_IDs[c_id]].mean()))
                        self.GI_quartiles[c_id].setText('({:.1f}, {:.1f}, {:.1f})'.format(x_y_values[self.Column_IDs[c_id]].quantile(q=0.25), x_y_values[self.Column_IDs[c_id]].quantile(), x_y_values[self.Column_IDs[c_id]].quantile(q=0.75)))

    def list_of_icd_indices_to_tooltip_string(self, time, list_of_icd_indices):
        """
        Takes a list of indices(+1) for self.ICD_Texts (pos. = Diagnosis added, neg. = Diagnosis removed) and converts them into a tooltip text.
        """
        return_string = 'Changes at time ' + str(int(float(time))) + ':\n\n'
        sorted_list_of_icd_indices = sorted(list_of_icd_indices, key=abs)
        for i in sorted_list_of_icd_indices:
            index = int(float(i))
            if index < 0:
                return_string = return_string + 'removed Diagnosis for ' + self.ICD_Strings[(-index - 1)] + '  (' + self.ICD_Descriptions[(-index - 1)] + ')\n'
            else:
                return_string = return_string + 'added Diagnosis for ' + self.ICD_Strings[(index - 1)] + '  (' + self.ICD_Descriptions[(index - 1)] + ')\n'
        else:
            return return_string

    @pyqtSlot()
    def update_data_selection_text(self, start_value, end_value):
        """
        Changes the selected data subset text based on slider settings

        Parameters
        ----------
        start_value
        end_value

        Returns
        -------

        """
        self.data_selection_start.setText(str(start_value))
        self.data_selection_end.setText(str(end_value))

    @pyqtSlot()
    def update_data_selection_slider(self):
        """
        Updates the slider widget based on selected data subset

        Returns
        -------

        """
        input_start = self.data_selection_start.text()
        input_end = self.data_selection_end.text()
        if not input_start:
            input_start = self.Dataframe['tOffset'].dropna().min()
        if not input_end:
            input_end = self.Dataframe['tOffset'].dropna().max()
        input_start = int(float(input_start))
        input_end = int(float(input_end))
        if input_start > input_end:
            return False
        if input_start < self.Dataframe['tOffset'].dropna().min():
            return False
        if input_end > self.Dataframe['tOffset'].dropna().max():
            return False
        self.data_selection_slider.setRange(input_start, input_end)