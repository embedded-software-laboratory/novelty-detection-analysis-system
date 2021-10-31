from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import os
from ndas.extensions import data
from ndas.database_interface import interface

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
		database = QComboBox()
		database.addItems({"asic_data_mimic", "asic_data_sepsis"})
		self.selectLabel = QLabel()
		self.selectLabel.setText("Select patient by patient id")
		self.patientId = QLineEdit()

		self.patientEntriesLabel = QLabel()
		self.patientEntriesLabel.setText("Show the patients who has the most entries in total in the database:")
		self.numberOfPatients = QLineEdit()
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
		self.parameter = QLineEdit()
		self.patientIdsScrollbar2 = QScrollArea()
		self.patientIdsScrollbar2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
		self.patientIdsScrollbar2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
		self.patientIdsScrollbar2.setWidgetResizable(True)
		self.patiendidsLabel2 = QLabel()
		self.patientIdsScrollbar2.setWidget(self.patiendidsLabel2)
		showPatients2 = QPushButton("Show patient ids")
		showPatients2.clicked.connect(lambda: self.showPatientsWithParameter(parent, self.numberOfPatients2.text(), database.currentText(), self.parameter.text()))

		confirm = QPushButton("Confirm")
		confirm.clicked.connect(lambda: self.loadPatient(parent, self.patientId.text(), database.currentText()))


		layout = QFormLayout()
		layout.addRow("Select database: ", database)
		layout.addRow(self.selectLabel)
		layout.addRow(self.patientId)
		layout.addRow(confirm)
		layout.addRow(self.patientEntriesLabel)
		layout.addRow("Enter number of patients:", self.numberOfPatients)
		layout.addRow(self.patientIdsScrollbar)
		layout.addRow(showPatients)
		layout.addRow(self.parameterEntriesLabel)
		layout.addRow("Enter number of patients:", self.numberOfPatients2)
		layout.addRow("Enter parameters: ", self.parameter)
		layout.addRow(self.patientIdsScrollbar2)
		layout.addRow(showPatients2)

		self.setLayout(layout)


	def loadPatient(self, parent, patientid, database):
		filename = os.getcwd()+"\\ndas\\local_data\\imported_patients\\{}_patient_{}.csv".format(database, str(patientid))
		result = 0
		if not os.path.exists(filename):
			result = interface.startInterface(["interface", "db_asic_scheme.json", "selectPatient", str(patientid), database])
		if result == -1:
			QMessageBox.critical(self, "Error", "Patient not found.", QMessageBox.Ok)
		else:
			data.set_instance("CSVImporter", filename)
			data.get_instance().signals.result_signal.connect(
				lambda result_data, labels: parent.getParent().data_import_result_slot(result_data, labels))
			data.get_instance().signals.status_signal.connect(lambda status: parent.getParent().progress_bar_update_slot(status))
			data.get_instance().signals.error_signal.connect(lambda s: parent.getParent().error_msg_slot(s))
			parent.getParent().thread_pool.start(data.get_instance())
			parent.close()

	def showPatients(self, parent, numberOfPatients, database):
		db = ""
		if database == "asic_data_mimic":
			db = "mimic"
		elif database == "asic_data_sepsis":
			db = "sepsis"
		result = interface.startInterface(["interface", "db_asic_scheme.json", "dataDensity", "bestPatients", "entriesTotal", numberOfPatients, db])
		patientids = ["Patient-ID | Number of entries\n", "----------------\n"]
		for patient in result:
			patientSplit = patient.split("\t")
			patientids.append(patientSplit[0] + " | " + patientSplit[1])

		self.patiendidsLabel.setText(''.join(patientids))

	def showPatientsWithParameter(self, parent, numberOfPatients, database, parameters):
		db = ""
		if database == "asic_data_mimic":
			db = "mimic"
		elif database == "asic_data_sepsis":
			db = "sepsis"
		result = interface.startInterface(["interface", "db_asic_scheme.json", "dataDensity", "bestPatients", parameters.replace(",", "+"), numberOfPatients, db])
		patientids = ["Patient-ID | Number of relevant entries\n", "----------------\n"]
		for patient in result:
			patientSplit = patient.split("\t")
			patientids.append(patientSplit[0] + " | " + patientSplit[1])
		self.patiendidsLabel2.setText(''.join(patientids))