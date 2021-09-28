from PyQt5.QtWidgets import *
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
		self.radio1 = QRadioButton("Select patient by patient id")
		self.radio1.setChecked(True)
		self.radio1.toggled.connect(lambda:self.btnstate(self.radio1))

		self.patientId = QLineEdit()
		self.patientIdFrame = QFrame()
		self.patientIdLayout = QFormLayout()
		self.patientIdLayout.addRow("Enter patient id: ", self.patientId)
		self.patientIdFrame.setLayout(self.patientIdLayout)

		self.radio2 = QRadioButton("Select the patient who has the most entries in the database")
		self.radio2.toggled.connect(lambda:self.btnstate(self.radio2))

		self.radio3 = QRadioButton("Select the patient who has the most entries for a specific parameter")
		self.radio3.toggled.connect(lambda:self.btnstate(self.radio3))

		self.parameter = QLineEdit()
		self.parameterFrame = QFrame()
		self.parameterLayout = QFormLayout()
		self.parameterLayout.addRow("Enter parameter: ", self.parameter)
		self.parameterFrame.setLayout(self.parameterLayout)

		confirm = QPushButton("Confirm")
		confirm.clicked.connect(lambda: self.getPatientData(parent, self.patientId.text(), database.currentText()))


		layout = QFormLayout()
		layout.addRow("Select database: ", database)
		layout.addRow(self.radio1)
		layout.addRow(self.patientIdFrame)
		layout.addRow(self.radio2)
		layout.addRow(self.radio3)
		layout.addRow(self.parameterFrame)
		layout.addRow(confirm)

		self.setLayout(layout)

		self.mode = 0
		self.parameterFrame.hide()

	def btnstate(self, button):
		if button == self.radio1:
			self.patientIdFrame.show()
			self.parameterFrame.hide()
			self.mode = 0
		if button == self.radio2:
			self.patientIdFrame.hide()
			self.parameterFrame.hide()
			self.mode = 1
		if button == self.radio3:
			self.patientIdFrame.hide()
			self.parameterFrame.show()
			self.mode = 2

	def getPatientData(self, parent, patientid, database):
		if self.mode == 0:
			self.loadPatient(parent, patientid, database)
		if self.mode == 1:
			patientid = interface.startInterface(["interface", "db_asic_scheme.json", "dataDensity", "bestPatients", "entriesTotal", 1, database])
			self.loadPatient(parent, patientid, database)
		if self.mode == 2:
			parameters = self.parameter.text().replace(" ", "").split(",")
			patientid = interface.startInterface(["interface", "db_asic_scheme.json", "dataDensity", "bestPatients"] + parameters + [1, database])
			self.loadPatient(parent, patientid, database)
		parent.close()

	def loadPatient(self, parent, patientid, database):
		filename = os.getcwd()+"\\ndas\\local_data\\imported_patients\\{}_patient_{}.csv".format(database, str(patientid))
		if not os.path.exists(filename):
			interface.startInterface(["interface", "db_asic_scheme.json", "selectPatient", str(patientid), database])
		data.set_instance("CSVImporter", filename)
		data.get_instance().signals.result_signal.connect(
			lambda result_data, labels: parent.getParent().data_import_result_slot(result_data, labels))
		data.get_instance().signals.status_signal.connect(lambda status: parent.getParent().progress_bar_update_slot(status))
		data.get_instance().signals.error_signal.connect(lambda s: parent.getParent().error_msg_slot(s))
		parent.getParent().thread_pool.start(data.get_instance())