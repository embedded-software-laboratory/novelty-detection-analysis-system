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
		with open(os.getcwd() + "\\ndas\\database_interface\\Interface_parameter.csv") as parameter_csv:
			csv_reader_object = csv.reader(parameter_csv, delimiter=";")
			firstLineFlag = True
			for row in csv_reader_object:
				if firstLineFlag:
					firstLineFlag = False
					continue
				if row[1]=="asic":
					self.parameters.append(LabeledButton(QCheckBox(row[0], self), row[3]))
					layout.addRow(self.parameters[-1].button)


		#self.parameter1 = QCheckBox("Atemfrequenz (gemessen)", self)
		#self.parameters.append(self.parameter1)
		#self.parameter2 = QCheckBox("Herzfrequenz", self)
		#self.parameters.append(self.parameter2)

		confirm = QPushButton("Confirm")
		confirm.clicked.connect(lambda: self.confirm(parent))

		#layout.addRow(self.parameter1)
		#layout.addRow(self.parameter2)
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
		result = ""
		label = ""
		for parameter in self.parameters:
			if parameter.button.isChecked():
				result = result + parameter.parameter_name + ", "
				label = label + parameter.button.text() + ", "
		result = result[:-2]
		label = label[:-2]
		parent.getParent().setSelectedParameters(result, label)
		parent.close()