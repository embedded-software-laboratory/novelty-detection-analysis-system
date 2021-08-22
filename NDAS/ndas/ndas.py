import sys

from .utils import earlyinit

try:
    from ndas.utils.checkpy import check_python_version
except ImportError:
    try:
        from ndas.utils.checkpy import check_python_version
    except (SystemError, ValueError):
        sys.stderr.write("Don't run this script directly.\n")
        sys.stderr.flush()
        sys.exit(100)
check_python_version()


def main():
    """
    Calls the app and the main entry point
    """
    earlyinit.early_init()

    from ndas import app
    return app.run()
