# A window which can be used by the user to select multiple parameters from a check box list (used by the importdatabasewidget)

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import os 
import csv

class LabeledButton():
	def __init__(self, button, parameter_name):
		self.button = button
		self.parameter_name = parameter_name

class SelectParametersWindow(QMainWindow):
	def __init__(self, parent=None):
		self.parent = parent
		super().__init__(parent)
		self.setCentralWidget(DatabaseSettingsWidget(self))
		self.setWindowTitle("Select parameters")

	def getParent(self):
		return self.parent

class DatabaseSettingsWidget(QWidget):
	def __init__(self, parent=None):
		super().__init__(parent)
		layout = QFormLayout()
		self.widget = QWidget()

		self.parameters = []
        # the chosable parameters are loaded from a csv file: TODO: They rather should be loaded from the table directly, because it can't be guaranteed that all asic-scheme tables use the same columns
		with open(os.getcwd() + "\\ndas\\database_interface\\Interface_parameter.csv") as parameter_csv:
			csv_reader_object = csv.reader(parameter_csv, delimiter=";")
			firstLineFlag = True
			for row in csv_reader_object:
				if firstLineFlag: # the first row in the table are the titles of the columns, so do not use it here
					firstLineFlag = False
					continue
				if row[1]=="asic": # add a labeled checkbox for every asic parameter
					self.parameters.append(LabeledButton(QCheckBox(row[0], self), row[3]))
					layout.addRow(self.parameters[-1].button)

		confirm = QPushButton("Confirm")
		confirm.clicked.connect(lambda: self.confirm(parent))

		layout.addRow(confirm)

		self.widget.setLayout(layout)

		self.parameterScrollbar = QScrollArea()
		self.parameterScrollbar.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
		self.parameterScrollbar.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
		self.parameterScrollbar.setWidgetResizable(True)
		self.parameterScrollbar.setWidget(self.widget)

		layout2 = QFormLayout()
		layout2.addRow(self.parameterScrollbar)
		self.setLayout(layout2)

	def confirm(self, parent):
        # the parameters selected by the users are sent to the importdatabasewidget
        
		result = "" # this will be directly integrated into the according sql query
		label = "" # this is the label that will be integrated into the user interface (so that the user can see what he has selected)
		for parameter in self.parameters:
			if parameter.button.isChecked():
				result = result + parameter.parameter_name + ", "
				label = label + parameter.button.text() + ", "
		result = result[:-2] # remove the comma from the end of the string
		label = label[:-2]
		parent.getParent().setSelectedParameters(result, label)
		parent.close()