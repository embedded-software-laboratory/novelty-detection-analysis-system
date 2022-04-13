import pyqtgraph as pg
from PyQt5.QtWidgets import *

from ndas.misc import colors


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

        color_map = {-2: colors.IGNORED, -1: colors.TRAINING, 0: colors.REGULAR, 1: colors.TIER1NOV,
                     2: colors.TIER2NOV}
        brush_map = [pg.mkBrush(color_map[novelty_point]) for novelty_point in novelty_data]

        self.graph = pg.PlotWidget()
        self.scatter = pg.ScatterPlotItem(x=time_column, y=data_column, brush=brush_map, pen='w', size=10)
        self.graph.addItem(self.scatter)

        self.layoutVertical.addWidget(self.graph)
