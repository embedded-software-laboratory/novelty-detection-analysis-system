# install_dependencies.bat
This script can be executed if you want to do development work or run the ndas python script directly. It checks wether... 
1. ...you have a compatible python version istalled,
2. ...all necessary libraries are installed and if not, installes the missing ones. 
So this simplifies the setup process as you don't need to install all packages by hand. 
You don't need to run this script if you just want to use the compiled version (ndas.exe).

# How to compile the NDAS
If you want to compile the software, follow these steps:
1. (if not yet done) install the python library "pyinstaller" 
2. Switch to the folder "novelty-detection-analysis-system\NDAS" and run the following command: pyinstaller --onefile ndas.py --hiddenimport PyQt5.QtQml --hiddenimport PyQt5.QtSql --hiddenimport PyQt5.QtOpenGL --hiddenimport wfdb --hiddenimport seaborn
3. The resulting exe-file is located in a folder named "dist". Copy it into the parent directory "novelty-detection-analysis-system\NDAS". From there, it should be possible to execute it. The folders "dist" and "build" can be deleted. 
