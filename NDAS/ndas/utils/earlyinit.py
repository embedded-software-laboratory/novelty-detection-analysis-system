import datetime
import importlib
import sys
import traceback

from PyQt5.QtCore import QT_VERSION, PYQT_VERSION, PYQT_VERSION_STR
from PyQt5.QtCore import QVersionNumber, QLibraryInfo
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox

START_TIME = datetime.datetime.now()


def init_log():
    """
    Initializes the logging
    """
    from ndas.utils import logger
    logger.init_log()
    logger.init.debug("Log initialized.")


def _missing_package_error_str(name: str):
    """
    Creates message if package is missing

    Parameters
    ----------
    name
    """
    blocks = ["Fatal error: <b>{}</b> is required. "
              "Please install first!".format(name),
              "<b>The error encountered was:</b><br />%ERROR%"]
    lines = ['Hint: Use pip to install the required packages']
    blocks.append('<br />'.join(lines))
    return '<br /><br />'.join(blocks)


def _die(message, exception=None):
    """
    Quits the program if package is missing

    Parameters
    ----------
    message
    exception
    """
    app = QApplication(sys.argv)
    if exception is not None:
        message = message.replace('%ERROR%', str(exception))
    msgbox = QMessageBox(QMessageBox.Critical, "ndas: Fatal error!", message)
    msgbox.setTextFormat(Qt.RichText)
    msgbox.resize(msgbox.sizeHint())
    msgbox.exec_()
    app.quit()
    sys.exit(1)


def qt_version(q_version=None, qt_version_str=None):
    """
    Returns the QT version

    Parameters
    ----------
    q_version
    qt_version_str

    """
    if q_version is None:
        from PyQt5.QtCore import qVersion
        q_version = qVersion()

    if qt_version_str is None:
        from PyQt5.QtCore import QT_VERSION_STR
        qt_version_str = QT_VERSION_STR

    if q_version != qt_version_str:
        return '{} (compiled {})'.format(q_version, qt_version_str)
    else:
        return q_version


def check_qt_version():
    """
    Checks the QT version
    """
    try:
        qt_ver = QLibraryInfo.version().normalized()
        recent_qt_runtime = qt_ver >= QVersionNumber(5, 12)
    except (ImportError, AttributeError):
        recent_qt_runtime = False

    if QT_VERSION < 0x050C00 or PYQT_VERSION < 0x050C00 or not recent_qt_runtime:
        text = ("Fatal error: Qt >= 5.12.0 and PyQt >= 5.12.0 are required, "
                "but Qt {} / PyQt {} is installed.".format(qt_version(),
                                                           PYQT_VERSION_STR))
        _die(text)


def _check_modules(modules):
    """
    Checks if all modules are available

    Parameters
    ----------
    modules
    """
    for name, text in modules.items():
        try:
            importlib.import_module(name)
        except ImportError as e:
            _die(text, e)


def check_libraries():
    """
    Checks if all libraries are installed
    """
    modules = {
        'csv': _missing_package_error_str("csv"),
        'pickle': _missing_package_error_str("pickle"),
        'numpy': _missing_package_error_str("numpy"),
        'pyqtgraph': _missing_package_error_str("pyqtgraph"),
        'yaml': _missing_package_error_str("PyYAML"),
        'PyQt5.QtQml': _missing_package_error_str("PyQt5.QtQml"),
        'PyQt5.QtSql': _missing_package_error_str("PyQt5.QtSql"),
        'PyQt5.QtOpenGL': _missing_package_error_str("PyQt5.QtOpenGL"),
        'bs4': _missing_package_error_str("bs4"),
        'hickle': _missing_package_error_str("hickle"),
        'lxml': _missing_package_error_str("lxml"),
        'pandas': _missing_package_error_str("pandas"),
        'wfdb': _missing_package_error_str("wfdb"),
        'seaborn': _missing_package_error_str("seaborn")
    }
    _check_modules(modules)


def early_init():
    """
    Calls the early init procedure
    """
    init_log()
    check_libraries()
    check_qt_version()
