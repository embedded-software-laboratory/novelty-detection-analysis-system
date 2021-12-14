import logging
import pandas as pd
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from ndas.misc import rangeslider, loggerwidget
from ndas.extensions import data, plots, physiologicallimits
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
        self.setLayout(self.layout)
        self.layout_left = QGridLayout()
        self.layout_left.setColumnStretch(1, 3)
        self.bar_plot_label = QLabel('Timeline of diagnoses:\n(hover for details)')
        self.layout_left.addWidget(self.bar_plot_label, 0, 0, alignment=Qt.AlignCenter)
        self.layout_right = QVBoxLayout()
        self.layout.addLayout(self.layout_left, stretch=6)
        self.layout.addLayout(self.layout_right)

        """
        Create Canvas for Bar Plot
        """
        self.bar_plot = pg.PlotWidget()
        self.bar_plot.getAxis('left').setPen(pg.mkPen(None))
        self.bar_plot.getAxis('left').setTextPen(pg.mkPen(None))
        self.bar_plot.setToolTip('Hover Bars for more Information about diagnoses')
        self.bar_plot.setXRange(0, 1)
        self.bar_plot.setMouseEnabled(x=False, y=False)
        self.bar_v_line = pg.InfiniteLine(pos=(-1), angle=90, pen=(pg.mkPen(self.palette().color(QPalette.Highlight))))
        self.bar_plot.addItem(self.bar_v_line, ignoreBounds=True)
        self.bar_plot.scene().sigMouseMoved.connect(lambda evt: self.on_mouse_moved_over_graph(evt, -1))
        self.layout_left.addWidget(self.bar_plot, 0, 1)
        self.layout_left.setRowStretch(0, 1)

        """
        Create Data Graphs
        """
        self.Titles = ['Temperature (°C)', 'Heartrate (bpm)', 'Respiratory Rate (bpm)', 'Systolic Blood Pressure (mmHg)', 'Diastolic Blood Pressure (mmHg)', 'Mean Blood Pressure (mmHg)', 'Oxygen Saturation (%)', 'End Tidal Carbon Dioxide (mmHg)', 'Central Venous Pressure (mmHg)']
        self.Titles_Short = ['Temperature', 'Heartrate', 'Resp. Rate', 'pSys Blood', 'pDia Blood', 'pMean Blood', 'Oxygen Sat.', 'End Tidal CO2', 'CVP']
        self.Column_IDs = ['TempC', 'HR', 'RR', 'pSy', 'pDi', 'pMe', 'SaO2', 'etCO2', 'CVP']
        self.Graphs = []
        self.Graph_Infos = []
        self.Show_Graph_Checkboxes = []

        """
        Create Box "Data Imputation Settings"
        """
        self.Imputation_Settings = QGroupBox('Data Imputation Settings')
        self.Imputation_Settings_Layout = QVBoxLayout()
        self.Imputation_Settings.setLayout(self.Imputation_Settings_Layout)
        self.Imputation_Method_Layout = QGridLayout()
        self.Imputation_Method_Selector = QComboBox()
        self.Imputation_Method_Selector.addItem('')
        for key in self.Baseimputator.Methods.keys():
            self.Imputation_Method_Selector.addItem(key)

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
        self.Imputation_Settings_Layout.addWidget(self.Imputation_Button, alignment=Qt.AlignTop)
        self.layout_right.addWidget(self.Imputation_Settings, alignment=Qt.AlignTop)

        """
        Create Box "Data Visualization Settings"
        """
        self.Data_Visualization_Settings = QGroupBox('Data Visualization Settings')
        self.Data_Visualization_Settings_Layout = QVBoxLayout()
        self.Data_Visualization_Settings.setLayout(self.Data_Visualization_Settings_Layout)
        self.Data_Visualization_Settings_Layout.addWidget(QLabel('Show Graph of:'))
        self.Show_Graph_Layout = QGridLayout()

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
        self.data_selection_range_layout.addWidget(self.data_selection_start_label, alignment=Qt.AlignLeft)
        self.data_selection_range_layout.addWidget(self.data_selection_start, alignment=Qt.AlignLeft)
        self.data_selection_range_layout.addWidget(self.data_selection_separation_label, alignment=Qt.AlignCenter)
        self.data_selection_range_layout.addWidget(self.data_selection_end, alignment=Qt.AlignRight)
        self.data_selection_slider = rangeslider.RangeSlider()
        self.data_selection_slider.setDisabled(True)
        self.data_selection_btn_layout = QHBoxLayout()
        self.data_selection_btn_layout.addWidget(self.data_selection_btn)
        self.data_selection_btn_layout.addWidget(self.data_selection_reset_btn)
        self.Data_Visualization_Settings_Layout.addWidget(self.data_selection_slider, alignment=Qt.AlignTop)
        self.Data_Visualization_Settings_Layout.addLayout(self.data_selection_range_layout)
        self.Data_Visualization_Settings_Layout.addLayout(self.data_selection_btn_layout)

        self.DataToggle = qtwidgets.Toggle(handle_color=Qt.gray, bar_color=Qt.lightGray, checked_color=(self.palette().color(QPalette.Highlight)))
        self.DataToggle.setDisabled(True)
        self.DataToggle.stateChanged.connect(lambda state: self.on_toggle_show_imputed_data(state))
        self.DataToggle_layout = QHBoxLayout()
        self.DataToggle_layout = QHBoxLayout()
        self.DataToggle_layout.addWidget(QLabel('Show original Data'), alignment=Qt.AlignCenter)
        self.DataToggle_layout.addWidget(self.DataToggle, alignment=Qt.AlignCenter)
        self.DataToggle_label2 = QLabel('Show imputed Data')
        self.DataToggle_label2.setStyleSheet('color: lightGray')
        self.DataToggle_layout.addWidget(self.DataToggle_label2, alignment=Qt.AlignCenter)
        self.Data_Visualization_Settings_Layout.addLayout(self.DataToggle_layout)

        self.apply_results_button = QPushButton('Apply imputation results onto loaded dataset')
        self.apply_results_button.clicked.connect(lambda: self.on_click_apply_results())
        self.apply_results_button.setDisabled(True)
        self.Data_Visualization_Settings_Layout.addWidget(self.apply_results_button)
        self.layout_right.addWidget(self.Data_Visualization_Settings, alignment=Qt.AlignTop)

        """
        Create Box "Patient Information"
        """
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
        self.Patient_BMI = QLabel('-')

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
        self.Patient_Information_General.addWidget(QLabel('BMI (calculated):'), 6, 0)
        self.Patient_Information_General.addWidget(self.Patient_BMI, 6, 1)

        self.ICD_Layout = QGridLayout()
        self.ICD_Labels = []
        self.ICD_Strings = ['280-285', '286-287', '288-289', '390-392', '393-398', '401-405', '410-414', '415-417', '420-429', '430-438', '440-449', '451-459', '460-466', '470-478', '480-488', '490-496', '500-508', '510-519', '800-804', '805-809', '810-819', '820-829', '830-839', '840-848', '850-854', '860-869', '870-879', '880-887', '890-897', '900-904', '905-909', '910-919', '920-924', '925-929', '930-939', '940-949', '950-957', '958-959', '960-979', '980-989', '990-995', '996-999']
        self.ICD_Descriptions = ['Anemia', 'Coagulation/hemorrhagic', 'Other diseases of the blood and blood forming organs', 'Acute rheumatic fever', 'Chronic rheumatic heart disease', 'Hypertensive disease', 'Ischemic heart disease', 'Diseases of pulmonary circulation', 'Other forms of heart disease', 'Cerebrovascular disease', 'Diseases of arteries, arterioles, and capillaries', 'Diseases of veins and lymphatics, and other diseases of circulatory system', 'Acute respiratory infections', 'Other diseases of the upper respiratory tract', 'Pneumonia and influenza', 'Chronic obstructive pulmonary disease and allied conditions', 'Pneumoconioses and other lung diseases due to external agents', 'Other diseases of respiratory system', 'Fracture of skull', 'Fracture of neck and trunk', 'Fracture of upper limb', 'Fracture of lower limb', 'Dislocation', 'Sprains and strains of joints and adjacent muscles', 'Intracranial injury, excluding those with skull fracture', 'Internal injury of thorax, abdomen, and pelvis', 'Open wound of head, neck, and trunk', 'Open wound of upper limb', 'Open wound of lower limb', 'Injury of blood vessels', 'Late effects of injuries, poisonings, toxic effects, and other external causes', 'Superficial injury', 'Contusion with intact skin surface', 'Crushing injury', 'Effects of foreign body entering through Body orifice', 'Burns', 'Injury of nerves and spinal cord', 'Certain traumatic complications and unspecified injuries', 'Poisoning by drugs, medicinal and biological substances', 'Toxic effects of substances chiefly nonmedicinal as to source', 'Other and unspecified effects of external causes', 'Complications of surgical and medical care, not elsewhere classified']
        self.ICD_Details = ['', '', 'Diseases of white blood cells, Polycythemia, Lymphadenitis, Hypersplenism, Other diseases of Spleen, Methemoglobinemia, Hypercoagulable state, Myelofibrosis, Heparin-induced thrombocytopenia', '', '', '', '', 'Pulmonary Heart Disease, Arteriovenous fistula of pulmonary vessels, Aneurysm of pulmonary artery', 'Pericarditis, Endocarditis, Myocarditis, Cardiomyopathy, Conduction disorders, Cardiac dysrhythmias, Heart failure, Cardiomegaly, Myocardial rupture, Hyperkinetic heart disease, Takotsubo syndrome', '', '', 'Phlebitis, Vein Embolism and Thrombosis, Varicose Veins, Hemorrhoids, Hypotension', '', 'Deviated nasal septum, Polyp, Chronic Pharyngitis, Chronic Sinusitis, Chronic disease of tonsils and adenoids, Peritonsilar abscess, Chronic laryngitis, Allergic Rhinitis', '', 'Bronchitis, Emphysema, Asthma, Bronchiectasis, COPD', '', 'Empyema, Pleurisy, Pneumothorax, Pulmonary congestion, ARDS, Respiratory failure', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'Early complications of Physical Trauma, Other Injuries', '', '', 'Effects of Radiation, Coldness, Heat, Air Presure, Lightning, Drowning, Neoplasms, Thirst, Exhaustion, Motion sickness, Asphyxiation, Electrocution; Adverse effects of Anaphylaxis, Anginoeurotic edema, Medication, Allergy, Anesthesia, Child abuse, Anaphylactic shock', '']
        for i in range(len(self.ICD_Strings)):
            self.ICD_Labels.append(QLabel(self.ICD_Strings[i]))
            self.ICD_Labels[i].setStyleSheet('color: lightGray')
            if not self.ICD_Details[i] == '':
                self.ICD_Labels[i].setToolTip(self.ICD_Descriptions[i]+ " (This includes: "+self.ICD_Details[i]+")")
            else:
                self.ICD_Labels[i].setToolTip(self.ICD_Descriptions[i])
            self.ICD_Layout.addWidget(self.ICD_Labels[i], i // 5, i % 5)

        self.Patient_Information_Layout.addLayout(self.Patient_Information_General)
        self.Patient_Information_Layout.addSpacerItem(QSpacerItem(0, 10))
        self.Patient_Information_ICD_Label = QLabel('Diagnosis ICD-Code-Ranges: (hover for details)')
        self.Patient_Information_Layout.addWidget(self.Patient_Information_ICD_Label)
        self.ICD_Grid_Widget = QWidget()
        self.ICD_Grid_Widget.setLayout(self.ICD_Layout)
        self.Patient_Information_Layout.addWidget(self.ICD_Grid_Widget)
        self.layout_right.addWidget(self.Patient_Information)

        """
        Create Box "Logging"
        """
        self.logging = QGroupBox('Logging')
        self.logging_layout = QVBoxLayout()
        self.logging.setLayout(self.logging_layout)
        self.logger_widget = loggerwidget.QPlainTextEditLogger()
        self.logging_layout.addWidget(self.logger_widget)
        self.layout_right.addWidget(self.logging)

        self.reset_graphs()

    def reset_graphs(self):
        self.bar_plot_label.hide()
        self.bar_plot.hide()
        self.Patient_Information_ICD_Label.hide()
        self.ICD_Grid_Widget.hide()
        for graph in self.Graphs:
            self.layout_left.removeWidget(graph)
            graph.deleteLater()
        for graph_info in self.Graph_Infos:
            self.layout_left.removeWidget(graph_info)
            graph_info.deleteLater()
        for checkbox in self.Show_Graph_Checkboxes:
            self.Show_Graph_Layout.removeWidget(checkbox)
            checkbox.deleteLater()
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
        self.h_lines = []
        self.Show_Graph_Checkboxes = []
        for row_num in range(self.layout_left.rowCount()):
            self.layout_left.setRowStretch(row_num, 0)

    def create_graph_for_each_series(self, list_of_columns):
        self.reset_graphs()
        for i, column in enumerate(list_of_columns):
            phys_dt = physiologicallimits.get_physical_dt(column)
            if phys_dt:
                title = phys_dt.id + " ("+phys_dt.unit+")"
            else:
                title = column
            self.Graphs.append(pg.PlotWidget())
            self.Graphs[i].setBackground('w')
            self.Graphs[i].setTitle(title)
            self.Graph_Plots.append(self.Graphs[i].plot([], [], pen=pg.mkPen(color=(128, 128, 128), width=1), symbol='o', symbolSize=2, symbolBrush='k'))
            self.Graphs[i].setXRange(0, 1)
            self.Graphs[i].setYRange(0, 1)
            self.Graphs[i].setMouseEnabled(x=False, y=True)
            self.Graphs[i].scene().sigMouseMoved.connect(lambda evt, bound_i=i: self.on_mouse_moved_over_graph(evt, bound_i))
            self.v_lines.append(pg.InfiniteLine(pos=(-1), angle=90, pen=(pg.mkPen(self.palette().color(QPalette.Highlight)))))
            self.h_lines.append(pg.InfiniteLine(pos=(-1000), angle=0, pen=(pg.mkPen(self.palette().color(QPalette.Highlight)))))
            self.Graph_Infos.append(QGroupBox('Stats - ' + title))
            self.Graph_Info_Layouts.append(QVBoxLayout())
            self.Graph_Infos[i].setLayout(self.Graph_Info_Layouts[i])
            self.GI_num_points_layouts.append(QHBoxLayout())
            self.GI_num_points.append(QLabel('-'))
            self.GI_num_points_layouts[i].addWidget((QLabel('Number of points:')), alignment=Qt.AlignLeft)
            self.GI_num_points_layouts[i].addWidget((self.GI_num_points[i]), alignment=Qt.AlignRight)
            self.Graph_Info_Layouts[i].addLayout(self.GI_num_points_layouts[i])
            self.GI_ranges_layouts.append(QHBoxLayout())
            self.GI_ranges.append(QLabel('-'))
            self.GI_ranges_layouts[i].addWidget((QLabel('Range of points:')), alignment=Qt.AlignLeft)
            self.GI_ranges_layouts[i].addWidget((self.GI_ranges[i]), alignment=Qt.AlignRight)
            self.Graph_Info_Layouts[i].addLayout(self.GI_ranges_layouts[i])
            self.GI_means_layouts.append(QHBoxLayout())
            self.GI_means.append(QLabel('-'))
            self.GI_means_layouts[i].addWidget((QLabel('Mean Value:')), alignment=Qt.AlignLeft)
            self.GI_means_layouts[i].addWidget((self.GI_means[i]), alignment=Qt.AlignRight)
            self.Graph_Info_Layouts[i].addLayout(self.GI_means_layouts[i])
            self.GI_quartiles_layouts.append(QHBoxLayout())
            self.GI_quartiles.append(QLabel('-'))
            self.GI_quartiles_layouts[i].addWidget((QLabel('Quartiles:')), alignment=Qt.AlignLeft)
            self.GI_quartiles_layouts[i].addWidget((self.GI_quartiles[i]), alignment=Qt.AlignRight)
            self.Graph_Info_Layouts[i].addLayout(self.GI_quartiles_layouts[i])

        for i, column in enumerate(list_of_columns):
            phys_dt = physiologicallimits.get_physical_dt(column)
            if phys_dt:
                title = phys_dt.id
            else:
                title = column
            self.Graphs[i].addItem(self.v_lines[i], ignoreBounds=True)
            self.Graphs[i].addItem(self.h_lines[i], ignoreBounds=True)
            self.layout_left.addWidget(self.Graph_Infos[i], i + 1, 0)
            self.layout_left.addWidget(self.Graphs[i], i + 1, 1)
            self.layout_left.setRowStretch(i + 1, 1)
            widget = QCheckBox(title)
            widget.stateChanged.connect(lambda state, bound_i=i: self.on_toggle_show_graph(state, bound_i))
            self.Show_Graph_Checkboxes.append(widget)
            self.Show_Graph_Checkboxes[i].setChecked(True)

        for i, checkbox in enumerate(self.Show_Graph_Checkboxes):
            self.Show_Graph_Layout.addWidget(checkbox, i // 3, i % 3)
        self.layout_left.update()

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
            self.apply_results_button.setDisabled(False)
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

    def on_click_set_range(self, redraw_graphs=False):
        """
        Sets the Range for Data to be Visualized and calls Data Visualization
        """
        lower_limit = int(float(self.data_selection_start.text()))
        upper_limit = int(float(self.data_selection_end.text()))
        time_column = self.Dataframe.columns[0]
        if self.Dataframe[time_column].empty:
            if redraw_graphs:
                self.visualize_data_and_redraw_graphs(self.Dataframe)
            else:
                self.visualize_data(self.Dataframe)
        else:
            if lower_limit >= self.Dataframe[time_column].dropna().max():
                lower_index = -1
            else:
                lower_index = self.Dataframe[time_column].ge(lower_limit).idxmax()
            if upper_limit >= self.Dataframe[time_column].dropna().max():
                upper_index = -1
            else:
                upper_index = self.Dataframe[time_column].le(upper_limit).iloc[::-1].idxmax()
            if redraw_graphs:
                self.visualize_data_and_redraw_graphs(self.Dataframe)
            else:
                self.visualize_data(self.Dataframe)

        self.bar_plot.setXRange(lower_limit, upper_limit)
        for Plot in self.Graphs:
            Plot.autoRange()
            Plot.setXRange(lower_limit, upper_limit)

    def on_click_reset_range(self, redraw_graphs=False):
        """
        Resets the Range for Data to be Visualized and calls Data Visualization
        """
        time_column = self.Dataframe.columns[0]
        if redraw_graphs:
            self.visualize_data_and_redraw_graphs(self.Dataframe)
        else:
            self.visualize_data(self.Dataframe)
        self.data_selection_start.setText(str(self.Dataframe[time_column].dropna().min()))
        self.data_selection_end.setText(str(self.Dataframe[time_column].dropna().max()))
        self.update_data_selection_slider()

        self.bar_plot.setXRange(self.Dataframe[time_column].dropna().min(), self.Dataframe[time_column].dropna().max())
        for Plot in self.Graphs:
            Plot.autoRange()
            Plot.setXRange(self.Dataframe[time_column].dropna().min(), self.Dataframe[time_column].dropna().max())

    def on_click_apply_results(self):
        time_col_name = data.get_dataframe_index_column()
        imputed_data = data.get_imputed_dataframe()
        expanded_data = data.get_full_dataframe().set_index(time_col_name).reindex(index=imputed_data[time_col_name], columns=[v for v in imputed_data.columns if v != time_col_name]).reset_index()
        mask = ((imputed_data == expanded_data) | ((imputed_data != imputed_data) & (expanded_data != expanded_data)))*-1+1
        if isinstance(data.get_mask_dataframe(), pd.DataFrame):
            mask = (mask+data.get_mask_dataframe()).clip(upper=1)
        data.set_mask_dataframe(mask)
        self.main_window.update_values_in_current_dataset(data.get_imputed_dataframe())

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
            time_column = some_data.columns[0]
            self.Dataframe = some_data
            self.DataToggle.setChecked(False)
            self.DataToggle.setDisabled(True)
            self.DataToggle_label2.setStyleSheet('color: lightGray')
            self.apply_results_button.setDisabled(True)
            self.data_selection_start.setDisabled(False)
            self.data_selection_end.setDisabled(False)
            self.data_selection_slider.setDisabled(False)
            self.data_selection_slider.setRangeLimit(self.Dataframe[time_column].dropna().min(), self.Dataframe[time_column].dropna().max())

            self.on_click_reset_range(redraw_graphs=True)
            self.data_selection_slider.valueChanged.connect(lambda start_value, end_value: self.update_data_selection_text(start_value, end_value))
            self.data_selection_start.textChanged.connect(lambda: self.update_data_selection_slider())
            self.data_selection_end.textChanged.connect(lambda: self.update_data_selection_slider())
        if isinstance(data.get_imputed_dataframe(), pd.DataFrame) and not data.get_imputed_dataframe().empty:
            self.DataToggle.setDisabled(False)
            self.apply_results_button.setDisabled(False)
            self.DataToggle_label2.setStyleSheet('color: black')

    def on_mouse_moved_over_graph(self, pos, i):
        """
        Update Vertical Lines and Y Value Label in bottom-right of main-window
        """
        if i == -1:
            point = self.bar_plot.plotItem.vb.mapSceneToView(pos)
        else:
            point = self.Graphs[i].plotItem.vb.mapSceneToView(pos)
        for h_line_ind in range(len(self.h_lines)):
            if i == h_line_ind:
                self.h_lines[h_line_ind].setPos(point.y())
            else:
                self.h_lines[h_line_ind].setPos(-1000)

        self.bar_v_line.setPos(point.x())
        for v_line in self.v_lines:
            v_line.setPos(point.x())

        self.main_window.x_label.setText('x=%0.01f' % point.x())
        self.main_window.y_label.setText('y=%0.01f' % point.y())

    def visualize_data_and_redraw_graphs(self, update_dataframe):
        used_columns = [col for col in update_dataframe.columns if col in plots.get_registered_plot_keys()]
        self.create_graph_for_each_series(used_columns)
        self.visualize_data(update_dataframe)

    def visualize_data(self, update_dataframe):
        """
        Visualizes the Graphs and Patient Information

        Only Show non-empty graphs
        """
        time_column = update_dataframe.columns[0]
        used_columns = [col for col in update_dataframe.columns if col in plots.get_registered_plot_keys()]

        for i, col in enumerate(used_columns):
            self.Show_Graph_Checkboxes[i].setChecked(not update_dataframe[col].isnull().all())
        length_of_data = update_dataframe[time_column].iloc[(-1)] - update_dataframe[time_column].iloc[0]
        self.bar_plot.clear()
        self.bar_plot.addItem(self.bar_v_line)

        """
        Update ICD Information and add diagnosis-bars to the Bar-Plot canvas
        """
        icd_changes = {}
        list_icd_columns = [col for col in update_dataframe.columns if "ICD" in col]
        for i, icd_string in enumerate(self.ICD_Strings):
            if not self.ICD_Details[i] == '':
                self.ICD_Labels[i].setToolTip(self.ICD_Descriptions[i]+ " (This includes: "+self.ICD_Details[i]+")")
            else:
                self.ICD_Labels[i].setToolTip(self.ICD_Descriptions[i])
            icd_col_name = ("ICD"+icd_string).replace('-', '')
            if icd_col_name in list_icd_columns and update_dataframe[icd_col_name].sum() > 0:
                local_icd_series = update_dataframe[icd_col_name]
                self.bar_plot_label.show()
                self.bar_plot.show()
                self.Patient_Information_ICD_Label.show()
                self.ICD_Grid_Widget.show()
                self.layout_left.setRowStretch(0, 1)
                self.ICD_Labels[i].setToolTip(self.ICD_Labels[i].toolTip()+"\n")
                self.ICD_Labels[i].setStyleSheet('color: black')
                list_of_value_change_indices = local_icd_series[(local_icd_series.diff() != 0)].index.tolist()

                if local_icd_series[list_of_value_change_indices[0]] == 0:
                    list_of_value_change_indices.remove(list_of_value_change_indices[0])

                for index in list_of_value_change_indices:
                    x_value = update_dataframe[time_column][(index - 1 + int(float(local_icd_series[index])))]
                    if str(x_value) not in icd_changes:
                        icd_changes[str(x_value)] = []
                    icd_changes[str(x_value)] = icd_changes[str(x_value)] + [(2 * local_icd_series[index] - 1) * (i + 1)]
                    if(2 * local_icd_series[index] - 1) >0:
                        self.ICD_Labels[i].setToolTip(self.ICD_Labels[i].toolTip() + "\nAdded at time: "+str(int(float(x_value))))
                    else:
                        self.ICD_Labels[i].setToolTip(self.ICD_Labels[i].toolTip() + "\nRemoved at time: " + str(int(float(x_value))))
            else:
                self.ICD_Labels[i].setStyleSheet('color: lightGray')

        for key in sorted(icd_changes.keys()):
            bar = pg.BarGraphItem(x0=[int(float(key)) - length_of_data / 120], x1=[int(float(key)) + length_of_data / 120], y=[0], height=0.6, pen=(pg.mkPen('k')), brush=(self.palette().color(QPalette.Highlight)))
            bar.setToolTip(self.list_of_icd_indices_to_tooltip_string(key, icd_changes[key]))
            self.bar_plot.addItem(bar)

        """
        Set general patient information
        """
        if "ID" in update_dataframe.columns:
            self.Patient_ID.setText(str(int(float(update_dataframe['ID'].iloc[0]))))
        else:
            self.Patient_ID.setText("No Specified")

        self.Patient_Gender.setText('Not Specified')
        if "gender(n m f)" in update_dataframe.columns:
            gender = int(update_dataframe['gender(n m f)'].iloc[0])
            if gender == 1:
                self.Patient_Gender.setText('Male')
            elif gender == 2:
                self.Patient_Gender.setText('Female')

        self.Patient_Age.setText('Not Specified')
        if "age(90= >89)" in update_dataframe.columns:
            age = update_dataframe['age(90= >89)'].iloc[0]
            if age == 90:
                self.Patient_Age.setText('Above 89')
            elif age:
                self.Patient_Age.setText(str(int(float(age))))

        self.Patient_Ethnicity.setText('Not Specified / Other')
        if "ethnicity(n cauc asia hisp afram natam)" in update_dataframe.columns:
            eth = update_dataframe['ethnicity(n cauc asia hisp afram natam)'].iloc[0]
            if eth == 1:
                self.Patient_Ethnicity.setText('Caucasian')
            elif eth == 2:
                self.Patient_Ethnicity.setText('Asian')
            elif eth == 3:
                self.Patient_Ethnicity.setText('Hispanic')
            elif eth == 4:
                self.Patient_Ethnicity.setText('African American')
            elif eth == 5:
                self.Patient_Ethnicity.setText('Native American')

        self.Patient_Height.setText('-')
        self.Patient_Weight.setText('-')
        self.Patient_BMI.setText('-')
        height = None
        if "height(cm)" in update_dataframe.columns:
            height = float(update_dataframe['height(cm)'].iloc[0])
            self.Patient_Height.setText('%.1f' % height)

        if "weight(kg)" in update_dataframe.columns:
            weight = float(update_dataframe['weight(kg)'].iloc[0])
            self.Patient_Weight.setText('%.1f' % weight)
            if height:
                self.Patient_BMI.setText('%.1f' % (weight * 10000 / height / height))
        """
        Add Data to Graphs and Graph Statistics
        """
        for c_id, col in enumerate(used_columns):
            x_y_values = update_dataframe[[time_column, col]].dropna()
            self.Graph_Plots[c_id].setData(x_y_values[time_column].tolist(), x_y_values[col].tolist())
            self.GI_num_points[c_id].setText(str(x_y_values[col].count()))
            self.GI_ranges[c_id].setText('[{:.1f} - {:.1f}]'.format(x_y_values[col].min(), x_y_values[col].max()))
            self.GI_means[c_id].setText('{:.2f}'.format(x_y_values[col].mean()))
            self.GI_quartiles[c_id].setText('({:.1f}, {:.1f}, {:.1f})'.format(x_y_values[col].quantile(q=0.25), x_y_values[col].quantile(), x_y_values[col].quantile(q=0.75)))

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
        time_column = self.Dataframe.columns[0]
        input_start = self.data_selection_start.text()
        input_end = self.data_selection_end.text()

        if not input_start:
            input_start = self.Dataframe[time_column].dropna().min()
        if not input_end:
            input_end = self.Dataframe[time_column].dropna().max()

        input_start = int(float(input_start))
        input_end = int(float(input_end))

        if input_start > input_end:
            return False
        if input_start < self.Dataframe[time_column].dropna().min():
            return False
        if input_end > self.Dataframe[time_column].dropna().max():
            return False

        self.data_selection_slider.setRange(input_start, input_end)
