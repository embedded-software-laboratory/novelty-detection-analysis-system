from PyQt5.QtWidgets import *
import os

class SSHSettingsWindow(QMainWindow):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.setCentralWidget(SSHSettingsWidget(self))
		self.setWindowTitle("Configure ssh authentification data")

class SSHSettingsWidget(QWidget):
	def __init__(self, parent=None):
		super().__init__(parent)
		label = QLabel()
		label.setText("For some functionalities, this software needs to establish a ssh connection to the smith-interface-server of the I11 laboratory. In order to use this functionalities, you need to configure your authentification data accordingly:")
		label.setWordWrap(True)
		username = QLineEdit()
		password = QLineEdit()
		password.setEchoMode(QLineEdit.Password)
		confirm = QPushButton("Confirm")
		confirm.clicked.connect(lambda: self.setAuthenticationData(parent, username.text(), password.text()))


		layout = QFormLayout()
		layout.addRow(label)
		layout.addRow("Enter username: ", username)
		layout.addRow("Enter password: ", password)
		layout.addRow(confirm)

		self.setLayout(layout)

	def setAuthenticationData(self, parent, username, password):
		if not os.path.exists(os.getcwd() + "\\ndas\\local_data"):
			os.makedirs(os.getcwd() + "\\ndas\\local_data")
		file = open(os.getcwd()+"\\ndas\\local_data\\sshSettings.json", "w")
		file.write("{\n	\"username\": \"" + username + "\",")
		file.write("\n	")
		file.write("\"password\": \"" + password + "\"\n}")
		parent.close()