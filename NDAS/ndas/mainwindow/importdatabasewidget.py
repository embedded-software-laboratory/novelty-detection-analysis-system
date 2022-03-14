from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QRegExpValidator
import os
from ndas.extensions import data
from ndas.database_interface import interface
from ndas.mainwindow.selectparameterswidget import SelectParametersWindow


class ImportDatabaseWindow(QMainWindow):
    def __init__(self, parent=None):
        self.parent = parent
        super().__init__(parent)
        self.setCentralWidget(DatabaseSettingsWidget(self))
        self.setWindowTitle("Import data from database")

    def getParent(self):
        return self.parent


class DatabaseSettingsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parameters=""
        self.allPatientIDs = [1,10,100,1000,10000,100000,1000000]
        reg_exp_number = QRegExp("[0-9]+")

        database = QComboBox()
        database.addItems({"asic_data_mimic", "asic_data_sepsis"})
        database.currentIndexChanged.connect(lambda: self.loadPatientIds(database.currentText()))
        self.selectLabel = QLabel()
        self.selectLabel.setText("Select patient by patient id")
        self.patientId = QLineEdit()
        self.patientId.setValidator(QRegExpValidator(reg_exp_number))
        self.patiendIDSlider = QSlider(Qt.Horizontal)
        self.patiendIDSlider.setMinimum(1)
        self.patiendIDSlider.setMaximum(1000000)
        self.patiendIDSlider.setSingleStep(1)
        self.patiendIDSlider.sliderReleased.connect(lambda: self.sliderChanged(self.patiendIDSlider.value()))
        confirm = QPushButton("Confirm")
        confirm.clicked.connect(lambda: self.loadPatient(parent, self.patientId.text(), database.currentText()))

        self.patientEntriesLabel = QLabel()
        self.patientEntriesLabel.setText("Show the patients who has the most entries in total in the database:")
        self.numberOfPatients = QLineEdit()
        self.numberOfPatients.setValidator(QRegExpValidator(reg_exp_number))
        self.patientIdsScrollbar = QScrollArea()
        self.patientIdsScrollbar.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.patientIdsScrollbar.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.patientIdsScrollbar.setWidgetResizable(True)
        self.patiendidsLabel = QLabel()
        self.patientIdsScrollbar.setWidget(self.patiendidsLabel)
        showPatients = QPushButton("Show patient ids")
        showPatients.clicked.connect(lambda: self.showPatients(parent, self.numberOfPatients.text(), database.currentText()))

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
        showPatients2 = QPushButton("Show patient ids")
        showPatients2.clicked.connect(lambda: self.showPatientsWithParameter(parent, self.numberOfPatients2.text(), database.currentText()))


        layout = QFormLayout()
        layout.addRow("Select database: ", database)
        layout.addRow(self.selectLabel)
        layout.addRow(self.patientId)
        layout.addRow(self.patiendIDSlider)
        layout.addRow(confirm)
        layout.addRow(self.patientEntriesLabel)
        layout.addRow("Enter number of patients:", self.numberOfPatients)
        layout.addRow(self.patientIdsScrollbar)
        layout.addRow(showPatients)
        layout.addRow(self.parameterEntriesLabel)
        layout.addRow("Enter number of patients:", self.numberOfPatients2)
        layout.addRow(self.parameter)
        layout.addRow("Selected parameters: ", self.selectedParameters)
        layout.addRow(self.patientIdsScrollbar2)
        layout.addRow(showPatients2)

        self.setLayout(layout)
        self.loadPatientIds(database.currentText())


    def loadPatient(self, parent, patientid, tableName):
        filename = os.getcwd()+"\\ndas\\local_data\\imported_patients\\{}_patient_{}.csv".format(tableName, str(patientid))
        result = 0
        if not os.path.exists(filename):
            result = interface.loadPatientData(tableName, str(patientid))
        if result == -1:
            QMessageBox.critical(self, "Error", "Patient not found.", QMessageBox.Ok)
        elif result == -3:
            QMessageBox.critical(self, "Error", "Connection to the database failed, make sure that you are connected to the i11-VPN", QMessageBox.Ok)
        elif result == -4:
            QMessageBox.critical(self, "Error", "Could not establish a connection to the database (connection timed out)", QMessageBox.Ok)
        elif result == -5:
            QMessageBox.critical(self, "Error", "SSH authentication failed, please make sure that you entered correct authentication data", QMessageBox.Ok)
        elif result == -6:
            QMessageBox.critical(self, "Error", "Database authentication failed, please make sure that you entered correct authentication data", QMessageBox.Ok)
        else:
            data.set_instance("CSVImporter", filename)
            data.get_instance().signals.result_signal.connect(
                lambda result_data, labels: parent.getParent().data_import_result_slot(result_data, labels))
            data.get_instance().signals.status_signal.connect(lambda status: parent.getParent().progress_bar_update_slot(status))
            data.get_instance().signals.error_signal.connect(lambda s: parent.getParent().error_msg_slot(s))
            parent.getParent().thread_pool.start(data.get_instance())
            parent.getParent().toggleTooltipStatus(parent.getParent().toggle_tooltip_btn, True)
            parent.close()

    def showPatients(self, parent, numberOfPatients, tableName):
        db = ""
        if tableName == "asic_data_mimic":
            db = "mimic"
        elif tableName == "asic_data_sepsis":
            db = "sepsis"
        result = interface.selectBestPatients(db, numberOfPatients)
        patientids = ["Patient-ID | Number of entries\n", "----------------\n"]
        if result == []:
            patientids = ["No result found"]
        elif result == -3:
            QMessageBox.critical(self, "Error", "Connection to the database failed, make sure that you are connected to the i11-VPN", QMessageBox.Ok)
        elif result == -4:
            QMessageBox.critical(self, "Error", "Could not establish a connection to the database (connection timed out)", QMessageBox.Ok)
        elif result == -5:
            QMessageBox.critical(self, "Error", "SSH authentication failed, please make sure that you entered correct authentication data", QMessageBox.Ok)
        elif result == -6:
            QMessageBox.critical(self, "Error", "Database authentication failed, please make sure that you entered correct authentication data", QMessageBox.Ok)
        else:
            for patient in result:
                patientSplit = patient.split("\t")
                patientids.append(patientSplit[0] + " | " + patientSplit[1])
        self.patiendidsLabel.setText(''.join(patientids))

    def showPatientsWithParameter(self, parent, numberOfPatients, database):
        db = ""
        if database == "asic_data_mimic":
            db = "mimic"
        elif database == "asic_data_sepsis":
            db = "sepsis"
        result = interface.selectBestPatientsWithParameters(db, numberOfPatients, self.parameters)
        patientids = []
        if result == -1:
            patientids = ["No result found"]
        elif result == -2:
            patientids = ["An error occured, please enter a valid number and valid parameters."]
        elif result == -3:
            QMessageBox.critical(self, "Error", "Connection to the database failed, make sure that you are connected to the i11-VPN", QMessageBox.Ok)
        elif result == -4:
            QMessageBox.critical(self, "Error", "Could not establish a connection to the database (connection timed out)", QMessageBox.Ok)
        elif result == -5:
            QMessageBox.critical(self, "Error", "SSH authentication failed, please make sure that you entered correct authentication data", QMessageBox.Ok)
        elif result == -6:
            QMessageBox.critical(self, "Error", "Database authentication failed, please make sure that you entered correct authentication data", QMessageBox.Ok)
        else:
            for patient in result:
                patientSplit = patient.split("\t")
                temp = patientSplit[0]
                for i in range(len(patientSplit)-1):
                    temp = temp + " | " + patientSplit[i+1]
                patientids.append(temp)
        self.patiendidsLabel2.setText(''.join(patientids))

    def chooseParameters(self):
        self.selectParameters = SelectParametersWindow(self)
        self.selectParameters.show()

    def setSelectedParameters(self, parameters, label):
        self.selectedParameters.setText(label)
        self.parameters = parameters
        print(self.parameters)
            
    def sliderChanged(self, newValue): 
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
        
    def loadPatientIds(self, table):
        if not os.path.exists(os.getcwd()+"\\ndas\\local_data\\{}_patient_ids.txt".format(table)):
            result = interface.loadPatientIds(table)
            if result == -6:
                QMessageBox.critical(self, "Error", "Database authentication failed, please make sure that you entered correct authentication data", QMessageBox.Ok)
                return
            file = open(os.getcwd()+"\\ndas\\local_data\\{}_patient_ids.txt".format(table), "w")
            for id in result:
                file.write(id)
            file.close()
        else:
            file = open(os.getcwd()+"\\ndas\\local_data\\{}_patient_ids.txt".format(table), "r")
            result = file.readlines()
            
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