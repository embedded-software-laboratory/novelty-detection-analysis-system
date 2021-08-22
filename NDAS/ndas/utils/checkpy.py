import sys

from tkinter import Tk, messagebox

def check_python_version():
    """
    Checks if python version is correct
    """
    if sys.hexversion < 0x03060000:
        version_str = '.'.join(map(str, sys.version_info[:3]))
        text = ("At least Python 3.6 is required to run NDAS. " +
                "Currently using " + version_str + ".\n")
        if Tk and '--no-err-windows' not in sys.argv:
            root = Tk()
            root.withdraw()
            messagebox.showerror("NDAS: Fatal error!", text)
        else:
            sys.stderr.write(text)
            sys.stderr.flush()
        sys.exit(1)


if __name__ == '__main__':
    check_python_version()
