import datetime
import os
import sys
from typing import cast

import PyQt5.QtCore
from PyQt5.QtCore import QThreadPool
from PyQt5.QtGui import QPixmap, QIcon, QPalette, QColor
from PyQt5.QtWidgets import QApplication

if hasattr(PyQt5.QtCore.Qt, 'AA_EnableHighDpiScaling'):
    PyQt5.QtWidgets.QApplication.setAttribute(PyQt5.QtCore.Qt.AA_EnableHighDpiScaling, True)

if hasattr(PyQt5.QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    PyQt5.QtWidgets.QApplication.setAttribute(PyQt5.QtCore.Qt.AA_UseHighDpiPixmaps, True)

import yaml

from ndas.mainwindow import mainwindow
from ndas.extensions import data, algorithms, annotations, savestate, plots, physiologicallimits
from ndas.utils import logger

q_app = cast(QApplication, None)
q_threadpool = QThreadPool()
q_threadpool.setMaxThreadCount(int(max(q_threadpool.maxThreadCount() / 2, 1)))


def run():
    """
    Main
    """
    logger.init.debug("Main process PID: {}".format(os.getpid()))

    # Initializing config
    logger.init.debug("Initializing config...")
    config = yaml.safe_load(open("ndas/config/config.yml"))

    # Initializing application
    logger.init.debug("Initializing application...")
    global q_app, q_threadpool
    q_app = Application(sys.argv)
    q_app.setOrganizationName("ndas")
    q_app.setApplicationName("ndas")
    q_app.setStyleSheet(mainwindow.MainWindow.STYLESHEET)

    if config["plots"]["use_dark_mode"]:
        q_app.setStyle("Fusion")
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, QColor(0, 0, 0))
        palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
        palette.setColor(QPalette.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
        q_app.setPalette(palette)

    # Init modules etc
    init(config)

    # Creating mainwindow
    window = mainwindow.MainWindow(q_threadpool)
    logger.init.debug("MainWindow created")
    logger.init.debug("Multithreading with maximum %d threads" % q_threadpool.maxThreadCount())
    window.show()

    # Run
    ret = q_app.exec_()
    return ret


def init(config):
    """
    Initializes modules

    Parameters
    ----------
    config
    """
    logger.init.debug("Starting init...")
    _init_icon()
    try:
        _init_modules(config)
    except (OSError, UnicodeDecodeError) as e:
        logger.init.exception(e, "Error while initializing modules!",
                              pre_text="Error while initializing modules")
        sys.exit()

    logger.init.debug("Init done")


def _init_icon():
    """
    Loads icon
    """
    fallback_icon = QIcon()
    for size in [16, 24, 32, 48, 64, 96, 128, 256, 512]:
        filename = 'icons/Icon-{size}.png'.format(size=size)
        pixmap = QPixmap(filename)
        if pixmap.isNull():
            logger.init.error("Failed to load NDAS icon")
        else:
            fallback_icon.addPixmap(pixmap)
    icon = QIcon.fromTheme('ndas', fallback_icon)
    if icon.isNull():
        logger.init.error("Failed to load NDAS icon")
    else:
        q_app.setWindowIcon(icon)
        logger.init.debug("Icon set.")


def _init_modules(config):
    """
    Initializes all modules with config

    Parameters
    ----------
    config
    """
    # Init Annotations
    logger.init.debug("Initializing physiological info...")
    physiologicallimits.init_physiological_info(config["physiologicalinfo"])

    # Init state handler
    logger.init.debug("Initializing state handler")
    savestate.init_state_handler(config["savestate"])

    # Init Annotations
    logger.init.debug("Initializing annotations...")
    annotations.init_annotations(config["annotation"])

    # Init DataImporter
    logger.init.debug("Initializing importer...")
    data.init_data_importer(config["data"])

    # Init algorithms
    logger.init.debug("Initializing algorithms...")
    algorithms.init_algorithms(config["algorithms"])

    # Init plots
    logger.init.debug("Initializing plots...")
    plots.init_graphs(config["plots"])


class Application(QApplication):
    """
    Main app
    """

    def __init__(self, *args, **kwargs):
        """
        Main entry point

        Parameters
        ----------
        args
        kwargs
        """
        self.setAttribute(PyQt5.QtCore.Qt.AA_UseHighDpiPixmaps, True)
        self.setAttribute(PyQt5.QtCore.Qt.AA_MacDontSwapCtrlAndMeta, True)
        os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
        self.setHighDpiScaleFactorRoundingPolicy(PyQt5.QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        self.setAttribute(PyQt5.QtCore.Qt.AA_Use96Dpi)

        import ctypes
        import platform

        def make_dpi_aware():
            if int(platform.release()) >= 8:
                ctypes.windll.shcore.SetProcessDpiAwareness(True)

        make_dpi_aware()

        super(Application, self).__init__(*args, **kwargs)

        self.launch_time = datetime.datetime.now()
