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
		patientId = QLineEdit()
		confirm = QPushButton("Confirm")
		confirm.clicked.connect(lambda: self.getPatientData(parent, patientId.text(), database.currentText()))


		layout = QFormLayout()
		layout.addRow("Select database: ", database)
		layout.addRow("Enter patient id: ", patientId)
		layout.addRow(confirm)

		self.setLayout(layout)

	def getPatientData(self, parent, patientid, database):
		filename = os.getcwd()+"\\ndas\\local_data\\imported_patients\\{}_patient_{}.csv".format(database, str(patientid))
		if not os.path.exists(filename):
			interface.startInterface(["interface", "db_asic_scheme.json", "selectPatient", str(patientid), database])
		data.set_instance("CSVImporter", filename)
		data.get_instance().signals.result_signal.connect(
			lambda result_data, labels: parent.getParent().data_import_result_slot(result_data, labels))
		data.get_instance().signals.status_signal.connect(lambda status: parent.getParent().progress_bar_update_slot(status))
		data.get_instance().signals.error_signal.connect(lambda s: parent.getParent().error_msg_slot(s))
		parent.getParent().thread_pool.start(data.get_instance())
		parent.close()