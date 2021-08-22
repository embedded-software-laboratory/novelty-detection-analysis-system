import pyqtgraph as pg
from PyQt5.QtWidgets import *

from ndas.misc.colors import Color


class BenchmarkPlotWindow(QWidget):
    """
    Widget to quickly plot benchmark results.
    """

    def __init__(self, title, time_column, data_column, novelties, *args, **kwargs):
        """
        Creation of the widget and loading of the plot

        Parameters
        ----------
        title
        time_column
        data_column
        novelties
        args
        kwargs
        """
        super(BenchmarkPlotWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle('Quick Plot: %s' % str(title))
        self.setMinimumSize(600, 400)

        self.layoutVertical = QVBoxLayout(self)

        novelty_data = []
        for index, x in time_column.iteritems():
            if x in novelties:
                novelty_data.append(novelties[x])
            else:
                novelty_data.append(0)

        color_map = {-2: Color.GREY.value, -1: Color.GREEN.value, 0: Color.BLUE.value, 1: Color.RED.value,
                     2: Color.YELLOW.value}
        brush_map = [pg.mkBrush(color_map[novelty_point]) for novelty_point in novelty_data]

        self.graph = pg.PlotWidget()
        self.scatter = pg.ScatterPlotItem(x=time_column, y=data_column, brush=brush_map, pen='w', size=10)
        self.graph.addItem(self.scatter)

        self.layoutVertical.addWidget(self.graph)
