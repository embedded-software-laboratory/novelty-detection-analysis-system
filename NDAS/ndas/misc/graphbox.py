import pyqtgraph as pg
from PyQt5 import QtCore, QtGui

from ndas.misc.colors import Color


class GraphBoxItem(pg.GraphicsObject):
    """
     Item to highlight multiple novelties
    """

    def __init__(self, x1, y1, x2, y2, color=Color.RED.value, fill=None, parent=None):
        """
        Creates the box at the specified coordinates

        Parameters
        ----------
        x1
        y1
        x2
        y2
        color
        fill
        parent
        """
        super().__init__(parent)
        self._rect = QtCore.QRectF(x1, y1, x2 - x1, y2 - y1)
        self.picture = QtGui.QPicture()
        self._generate_picture(color, fill)

    @property
    def rect(self):
        """
        Returns the rectangle
        """
        return self._rect

    def _generate_picture(self, color, fill):
        """
        Generates the rectangle

        Parameters
        ----------
        color
        fill
        """
        painter = QtGui.QPainter(self.picture)
        painter.setPen(pg.mkPen(color=color, width=1))
        if fill is not None:
            painter.setBrush(pg.mkBrush(fill))
        painter.drawRect(self.rect)
        painter.end()

    def paint(self, painter, option, widget=None):
        """
        Draws the rectangle

        Parameters
        ----------
        painter
        option
        widget
        """
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        """
        Returns the rectangle dimensions
        """
        return QtCore.QRectF(self.picture.boundingRect())
