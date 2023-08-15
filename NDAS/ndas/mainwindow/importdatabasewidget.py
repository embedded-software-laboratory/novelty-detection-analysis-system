# This is the window which is used to load several information and patient data from the tables stored in the SMITH_ASIC_SCHEME.

from tkinter import OFF
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QRegExpValidator
import os
from ndas.extensions import data
from ndas.database_interface import interface
from ndas.mainwindow.selectparameterswidget import SelectParametersWindow
import random


class ImportDatabaseWindow(QMainWindow):
    def __init__(self, parent=None):
        self.parent = parent
        super().__init__(parent)
        self.errorFlag = False
        self.setCentralWidget(DatabaseSettingsWidget(self))
        self.setWindowTitle("Import data from database")

    def getParent(self):
        return self.parent
        
    def closeEvent(self, event):
        # when the window is closed, set this flag of the main window class to false (it is set to true when this window is opened to prevent that this window is opened multiple times)
        self.parent.importwindowopened = False
        event.accept()


class DatabaseSettingsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parameters=""
        self.allPatientIDs = [1,10,100,1000,10000,100000,1000000]
        reg_exp_number = QRegExp("[0-9]+") #Regular expression which expresses an integer (this is used to ensure that the user only enteres numbers in some fields where this is necessary)

        database = QComboBox()
        database.addItems({"asic_data_mimic", "asic_data_sepsis", "uka_data", "smith_omop"}) # dropdown list with all available databases
        database.currentIndexChanged.connect(lambda: self.switchDatabase(database.currentText())) # when the user selects another table, load the patient ids which occur in this table (they are needed by the patient id-slider)

        self.data_source_flag = False

        self.selectLabel = QLabel()
        self.selectLabel.setText("Select patient by patient id")
        self.patientId = QLineEdit()
        self.patientId.setValidator(QRegExpValidator(reg_exp_number)) # ensure that only numbers can be entered into this field
        self.patiendIDSlider = QSlider(Qt.Horizontal)
        self.patiendIDSlider.setMinimum(1)
        self.patiendIDSlider.setMaximum(1000000)
        self.patiendIDSlider.setSingleStep(1)
        self.patiendIDSlider.sliderReleased.connect(lambda: self.sliderChanged(self.patiendIDSlider.value()))
        confirm = QPushButton("Confirm")
        confirm.clicked.connect(lambda: self.loadPatient(parent, self.patientId.text(), database.currentText()))
        select_random = QPushButton("Select Random Patient")
        select_random.clicked.connect(lambda: self.load_randomPatient(parent, database.currentText()))

        self.patientEntriesLabel = QLabel()
        self.patientEntriesLabel.setText("Show the patients who has the most entries in total in the database:")
        self.numberOfPatients = QLineEdit()
        self.numberOfPatients.setValidator(QRegExpValidator(reg_exp_number)) # ensure that only numbers can be entered into this field
        self.patientIdsScrollbar = QScrollArea()
        self.patientIdsScrollbar.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.patientIdsScrollbar.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.patientIdsScrollbar.setWidgetResizable(True)
        self.patiendidsLabel = QLabel()
        self.patientIdsScrollbar.setWidget(self.patiendidsLabel)
        self.showPatientsBtn = QPushButton("Show patient ids")
        self.showPatientsBtn.clicked.connect(lambda: self.showPatients(parent, self.numberOfPatients.text(), database.currentText()))

        self.parameterEntriesLabel = QLabel()
        self.parameterEntriesLabel.setText("Show the patients who has the most entries for a specific parameter:")
        self.numberOfPatients2 = QLineEdit()
        self.numberOfPatients2.setValidator(QRegExpValidator(reg_exp_number))
        self.parameter = QPushButton("Choose parameters...")
        self.parameter.clicked.connect(lambda: self.chooseParameters())
        self.selectedParameters = QLabel()
        self.patientIdsScrollbar2 = QScrollArea()
        self.patientIdsScrollbar2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.patientIdsScrollbar2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.patientIdsScrollbar2.setWidgetResizable(True)
        self.patiendidsLabel2 = QLabel()
        self.patientIdsScrollbar2.setWidget(self.patiendidsLabel2)
        self.showPatientsBtn2 = QPushButton("Show patient ids")
        self.showPatientsBtn2.clicked.connect(lambda: self.showPatientsWithParameter(parent, self.numberOfPatients2.text(), database.currentText()))


        self.layout = QFormLayout()
        self.layout.addRow("Select database: ", database)
        self.layout.addRow(self.selectLabel)
        self.layout.addRow(self.patientId)
        self.layout.addRow(self.patiendIDSlider)
        self.layout.addRow(confirm)
        self.layout.addRow(select_random)
        self.layout.addRow(self.patientEntriesLabel)
        self.layout.addRow("Enter number of patients:", self.numberOfPatients)
        self.layout.addRow(self.patientIdsScrollbar)
        self.layout.addRow(self.showPatientsBtn)
        self.layout.addRow(self.parameterEntriesLabel)
        self.layout.addRow("Enter number of patients:", self.numberOfPatients2)
        self.layout.addRow(self.parameter)
        self.layout.addRow("Selected parameters: ", self.selectedParameters)
        self.layout.addRow(self.patientIdsScrollbar2)
        self.layout.addRow(self.showPatientsBtn2)

        self.setLayout(self.layout)
        result = self.switchDatabase(database.currentText()) # load the patient ids which occur in the current selected database so that the range of the patient id slider can be set accordingly
        
        #if the connection to the database failed, set an according error flag - the main window closes this import window then
        if result == -1:
            self.parent().errorFlag = True

    def loadPatient(self, parent, patientid, tableName):
        """
        loads the specified patient of the specified table / database into the NDAS
        """
        filename = os.getcwd()+"\\ndas\\local_data\\imported_patients\\{}_patient_{}.csv".format(tableName, str(patientid))
        result = 0
        parent.getParent().overlay.show()
        if tableName == "smith_omop":
            if not os.path.exists(filename): # check if there already exists a local copy of the data
                result = interface.loadPatientData(tableName, str(patientid), self.omop_data_source.currentText()) # if not, load the data directly from the database (they are stored into a csv file at the above path)
            
        # if something went wrong during loading the data, present a respective error message
        if result == -1:
            parent.getParent().overlay.hide()
            QMessageBox.critical(self, "Error", "Patient not found.", QMessageBox.Ok)
        elif result in [-3,-4,-5,-6]:
            self.showConnectionErrorMessage(result)
            parent.getParent().overlay.hide()
        else:
            # if everything is ok, load the data into the NDAS and close this window
            data.set_instance("CSVImporter", filename)
            data.get_instance().signals.result_signal.connect(
                lambda result_data, labels: parent.getParent().data_import_result_slot(result_data, labels))
            data.get_instance().signals.status_signal.connect(lambda status: parent.getParent().progress_bar_update_slot(status))
            data.get_instance().signals.error_signal.connect(lambda s: parent.getParent().error_msg_slot(s))
            parent.getParent().thread_pool.start(data.get_instance())
            parent.getParent().toggleTooltipStatus(parent.getParent().toggle_tooltip_btn, True)
            parent.getParent().toggle_additional_labels.setChecked(False)
            parent.getParent().toggle_additional_labels.setEnabled(False)
            parent.getParent().currentPatientInformation = tableName + str(patientid)
            parent.getParent().setWindowTitle("NDAS (Patient Nr. " + str(patientid) + " aus " + tableName + " geladen)")
            parent.close()

    def load_randomPatient(self, parent, tableName):
        identifier = int(random.choice(self.allPatientIDs))
        self.loadPatient(parent, identifier, tableName)

    def showPatients(self, parent, numberOfPatients, tableName):
        """
        Loads the patients who has the most entries in the specified table and shows the result in the patientidsLabel
        
        Paramters:
            - numberOfPatients (int) - the number of patients that should be depicted
            - tableName (str) - the name of the table (there must be a table with this name in the SMITH_ASIC_SCHEME)
        """
        
        #determine the right postfix first, because the data are loaded from the look up table named asic_lookup_<source_db>
        db = ""
        if tableName == "asic_data_mimic":
            db = "mimic"
        elif tableName == "asic_data_sepsis":
            db = "sepsis"
            
        #retrieve the results    
        result = interface.selectBestPatients(db, numberOfPatients)
        patientids = ["Patient-ID | Number of entries\n", "----------------\n"]
        
        #if something went wrong, depict an according error message
        if result == []:
            patientids = ["No result found"]
        elif result in [-3,-4,-5,-6]:
            self.showConnectionErrorMessage(result)
        else:
        # print the result to the user interface
            for patient in result:
                patientSplit = patient.split("\t")
                patientids.append(patientSplit[0] + " | " + patientSplit[1])
        self.patiendidsLabel.setText(''.join(patientids))

    def showPatientsWithParameter(self, parent, numberOfPatients, database):
        """
        Loads the patient ids and numbers of entries per parameter which have the most entries for the specified parameters in the given table and shows the result in the patientidsLabel
        The parameters are selected by the user in a seperate window and stored in the global variable self.parameters
        
        Paramters:
           - numberOfPatients (int) - the number of patients that should be depicted
           - tableName (str) - the name of the table (there must be a table with this name in the SMITH_ASIC_SCHEME)
        
        """
        
        #determine the right postfix first, because the data are loaded from the look up table named asic_lookup_<source_db>
        db = ""
        if database == "asic_data_mimic":
            db = "mimic"
        elif database == "asic_data_sepsis":
            db = "sepsis"
            
        #retrieve the result
        result = interface.selectBestPatientsWithParameters(db, numberOfPatients, self.parameters)
        patientids = []
        
        #if something went wrong, depict an according error message
        if result == -1:
            patientids = ["No result found"]
        elif result == -2:
            patientids = ["An error occured, please enter a valid number and valid parameters."]
        elif result in [-3,-4,-5,-6]:
            self.showConnectionErrorMessage(result)
        else:
        # print the result to the user interface
            for patient in result:
                patientSplit = patient.split("\t")
                temp = patientSplit[0]
                for i in range(len(patientSplit)-1):
                    temp = temp + " | " + patientSplit[i+1]
                patientids.append(temp)
        self.patiendidsLabel2.setText(''.join(patientids))

    def chooseParameters(self):
        # opens a window in which the user can select a couple of parameters (these are used as the input for the showPatientsWithParameter-function)
        self.selectParameters = SelectParametersWindow(self)
        self.selectParameters.show()

    def setSelectedParameters(self, parameters, label):
        # sets the selected parameters which the user selected in the SelectParametersWindow (see chooseParameters-function). This function is called by the selectparameterswidget.
        self.selectedParameters.setText(label)
        self.parameters = parameters
            
    def sliderChanged(self, newValue): 
        # the slider step range is set to 1, but usually not all patient ids from the min to the max value exists in the asic-scheme tables. So if the user changes the slider, this function determines the nearest actual existing patient id for the selected value and chooses this als the selected patient id
        threshold = float("inf")
        nearestValue = -1
        for i in self.allPatientIDs:
            i = int(i)
            if abs(i-newValue) < threshold:
                threshold = abs(i-newValue)
                nearestValue = i
                if threshold == 0:
                    break
        self.patientId.setText(str(nearestValue))

    def loadPatientIDs(self, table):
        # load all existing patient ids from the given table / database. These are used to set the range of the patient id slider correctly and by the sliderChanged function
        txtFileName = ""
        if table == "smith_omop":
            txtFileName = table + self.omop_data_source.currentText()
        else:
            txtFileName = table

        if not os.path.exists(os.getcwd()+"\\ndas\\local_data\\{}_patient_ids.txt".format(txtFileName)): # check if there already exist a local copy for this table / database
            if table == "smith_omop":
                result = interface.loadPatientIds(table, self.omop_data_source.currentText()) # if not, load the data directly from the datababase
            else:
                result = interface.loadPatientIds(table) # if not, load the data directly from the datababase
            
            # error messages for the different possible failures
            if result in [-3,-4,-5,-6]:
                self.showConnectionErrorMessage(result)
                return -1
                
            # write the result to a text file. This reduces the loading time for further uses
            file = open(os.getcwd()+"\\ndas\\local_data\\{}_patient_ids.txt".format(txtFileName), "w")
            for id in result:
                file.write(id)
            file.close()
        else:
            # if there already exists the local text file, load the ids from there
            # but do a connection test first
            
            result = interface.testDatabaseConnection()
            if not(result == 1): # connection failed, don't need to procced
                self.showConnectionErrorMessage(result)
                return -1 
            file = open(os.getcwd()+"\\ndas\\local_data\\{}_patient_ids.txt".format(txtFileName), "r")
            result = file.readlines()
            
        # find the smallest and biggest ids and set the range of the slider accordingly
        self.allPatientIDs = result
        maxValue = -1
        minValue = float("inf")
        for i in result:
            i = int(i)
            if i > maxValue: 
                maxValue = i
            if i < minValue:
                minValue = i
        self.patiendIDSlider.setMinimum(minValue)
        self.patiendIDSlider.setMaximum(maxValue)

    def switchDatabase(self, table):
        if table == "smith_omop" or table == "uka_data": 
            self.showPatientsBtn.setEnabled(False)
            self.showPatientsBtn2.setEnabled(False)
            self.parameter.setEnabled(False)
        else:
            self.showPatientsBtn.setEnabled(True)
            self.showPatientsBtn2.setEnabled(True)
            self.parameter.setEnabled(True)
        if table == "smith_omop" and self.data_source_flag == False: 
            self.omop_data_source = QComboBox()
            self.omop_data_source.addItems({"MIMIC3", "MIMICIV", "HIRID"})
            self.omop_data_source.currentIndexChanged.connect(lambda: self.loadPatientIDs("smith_omop"))
            self.layout.insertRow(1, "Select data source: ", self.omop_data_source)
            self.data_source_flag = True
        else:
            if self.data_source_flag == True:
                self.layout.removeRow(1)
                self.data_source_flag = False
        
        self.loadPatientIDs(table)        
        
    def showConnectionErrorMessage(self, errorCode):
        if errorCode == -6:
            QMessageBox.critical(self, "Error", "Database authentication failed, please make sure that you entered correct authentication data", QMessageBox.Ok)
        elif errorCode == -3:
            QMessageBox.critical(self, "Error", "Connection to the database failed, make sure that you are connected to the i11-VPN", QMessageBox.Ok)
        elif errorCode == -4:
            QMessageBox.critical(self, "Error", "Could not establish a connection to the database (connection timed out)", QMessageBox.Ok)
        elif errorCode == -5:
            QMessageBox.critical(self, "Error", "SSH authentication failed, please make sure that you entered correct authentication data", QMessageBox.Ok)