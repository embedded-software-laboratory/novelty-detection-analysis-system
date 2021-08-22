import logging

LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}

LOG_NAMES = [
    'init', 'annotations', 'misc', 'signals', 'savestate', 'plots',
    'config', 'algorithms', 'mainwindow', 'importer', 'data', 'physicalinfo'
]

misc = logging.getLogger('misc')
annotations = logging.getLogger('annotations')
savestate = logging.getLogger('savestate')
algorithms = logging.getLogger('algorithms')
importer = logging.getLogger('importer')
plots = logging.getLogger('plots')
mainwindow = logging.getLogger('mainwindow')
init = logging.getLogger('init')
signals = logging.getLogger('signals')
config = logging.getLogger('config')
data = logging.getLogger('data')
physicalinfo = logging.getLogger('physicalinfo')


def init_log() -> None:
    """
    Initializes the logging
    """
    logging.captureWarnings(True)
    log_formatter = logging.Formatter("%(asctime)s [%(threadName)s] [%(name)s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.NOTSET)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
