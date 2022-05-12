# This is a window which allows the user to enter the login data for the smith-interface server

from PyQt5.QtWidgets import *
import os
import json

class SSHSettingsWindow(QMainWindow):
    def __init__(self, parent=None):
        self.parent = parent
        super().__init__(parent)
        self.setCentralWidget(SSHSettingsWidget(self))
        self.setWindowTitle("Configure ssh authentification data")
        
    def closeEvent(self, event):
        # when the window is closed, set this flag of the main window class to false (it is set to true when this window is opened to prevent that this window is opened multiple times)
        self.parent.sshsettingsopened = False
        event.accept()

class SSHSettingsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        label = QLabel()
        label.setText("For some functionalities, this software needs to establish a ssh connection to the smith-interface-server of the I11 laboratory. In order to use this functionalities, you need to configure your authentification data accordingly:")
        label.setWordWrap(True)
        
        un = ""
        pwd = ""
        
        # if the user already entered login data earlier, load them so that the user can see what he entered
        if os.path.exists(os.getcwd() + "\\ndas\\local_data\\sshSettings.json"):
            file = open(os.getcwd()+"\\ndas\\local_data\\sshSettings.json", "r")
            data = json.load(file)
            un = data["username"]
            pwd = data["password"]
            file.close()
        
        self.hidePassword = True;
        username = QLineEdit(un)
        self.password = QLineEdit(pwd)
        self.password.setEchoMode(QLineEdit.Password)
        confirm = QPushButton("Confirm")
        confirm.clicked.connect(lambda: self.setAuthenticationData(parent, username.text(), self.password.text()))
        self.show_password = QPushButton("Show Password")
        self.show_password.clicked.connect(lambda: self.togglePasswordField())


        layout = QFormLayout()
        layout.addRow(label)
        layout.addRow("Enter username: ", username)
        layout.addRow("Enter password: ", self.password)
        layout.addRow(confirm)
        layout.addRow(self.show_password)

        self.setLayout(layout)

    def setAuthenticationData(self, parent, username, password):
        # stores the entered login data into a json file in the local_data folder
        if not os.path.exists(os.getcwd() + "\\ndas\\local_data"):
            os.makedirs(os.getcwd() + "\\ndas\\local_data")
        file = open(os.getcwd()+"\\ndas\\local_data\\sshSettings.json", "w")
        file.write("{\n    \"username\": \"" + username + "\",")
        file.write("\n    ")
        file.write("\"password\": \"" + password + "\"\n}")
        parent.close()
        
    def togglePasswordField(self):
        # toggles the password field between password mode and normal text field (default when the window is opened is password mode)
        if self.hidePassword:
            self.password.setEchoMode(QLineEdit.Normal)
            self.show_password.setText("Hide password")
        else:
            self.password.setEchoMode(QLineEdit.Password)
            self.show_password.setText("Show password")
        self.hidePassword = not self.hidePassword