import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import *
from pyqtgraph.Qt import QtCore
from scipy import stats

from ndas.extensions import data, plots
from ndas.misc import graphbox
from ndas.misc.colors import Color


class StatsGraph(pg.GraphicsLayoutWidget):
    """
    Parent class of all statistics graphs
    """

    def __init__(self):
        super().__init__()


class StatsGraphWidget(QWidget):
    """
    Widget to display statistic graphs
    """

    def __init__(self):
        """
        Creates the statistic graphs and plot selector option
        """
        super().__init__()

        self.main_layout = QGridLayout(self)

        self.left_up_layout = QVBoxLayout()
        self.left_down_layout = QVBoxLayout()
        self.right_up_layout = QVBoxLayout()
        self.right_down_layout = QVBoxLayout()

        self.density_graph = DensityGraphWidget()
        self.histogram_graph = HistogramGraphWidget()
        self.correlation_graph = CorrelationGraphWidget()
        self.another_graph = BoxplotGraphWidget()

        self.left_up_layout.addWidget(self.density_graph)
        self.left_down_layout.addWidget(self.histogram_graph)
        self.right_up_layout.addWidget(self.correlation_graph)
        self.right_down_layout.addWidget(self.another_graph)

        self.main_layout.addLayout(self.left_up_layout, 0, 0)
        self.main_layout.addLayout(self.left_down_layout, 1, 0)
        self.main_layout.addLayout(self.right_up_layout, 0, 1)
        self.main_layout.addLayout(self.right_down_layout, 1, 1)

        self.plot_selector_layout = QHBoxLayout()
        self.plot_selector_label = QLabel("Active Plot:")
        self.plot_selector = QComboBox()
        self.plot_selector.setMinimumWidth(200)
        self.plot_selector.setDisabled(True)
        self.plot_selector_layout.addWidget(self.plot_selector_label)
        self.plot_selector_layout.addWidget(self.plot_selector)
        spacer = QSpacerItem(20, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.plot_selector_layout.addItem(spacer)
        self.main_layout.addLayout(self.plot_selector_layout, 2, 0, 1, 1)

        self.plot_selector.currentTextChanged.connect(lambda: self.update_statistical_plots())

    def draw_plots(self, active_plot):
        """
        Draws all four available statistic graphs

        Parameters
        ----------
        active_plot
        """
        self.density_graph.draw_plot(active_plot)
        self.histogram_graph.draw_plot(active_plot)
        self.correlation_graph.draw_plot(active_plot)
        self.another_graph.draw_plot(active_plot)

    def update_statistical_plots(self):
        """
        Updates the statistic plots if plot selection changes
        """
        active_plot = self.plot_selector.currentText()

        if (active_plot is not None and active_plot != ''
                and active_plot in data.get_dataframe().columns.tolist()):
            self.draw_plots(active_plot)


class DensityGraphWidget(StatsGraph):
    """
    Graph to visualize the number of non-nan values
    """
    custom_plot = None

    def __init__(self, *args, **kwargs):
        """
        Creates the custom plot

        Parameters
        ----------
        args
        kwargs
        """
        super(DensityGraphWidget, self).__init__(*args, **kwargs)
        self.custom_plot = self.addPlot(0, 0)
        self.legend = self.custom_plot.addLegend(colCount=1)
        self.vbs = self.addViewBox(row=1, col=0)
        self.vbs.setMaximumHeight(20)
        self.legend.setParentItem(self.vbs)
        self.legend.anchor((0, 0), (0, 0), offset=(85, -15))
        self.custom_plot.setMenuEnabled(False)
        self.custom_plot.setMouseEnabled(False, False)

    def set_custom_ticks(self, labels):
        """
        Changes the ticks to column labels

        Parameters
        ----------
        labels
        """
        ticks_y = [list(zip(range(len(labels)), labels))]
        yax = self.custom_plot.getAxis('left')
        yax.setTicks(ticks_y)

    def draw_plot(self, active_plot):
        """
        Draws the statistic plot

        Parameters
        ----------
        active_plot
        """
        self.custom_plot.clear()

        plot_data = data.get_dataframe().copy()
        time_label = data.get_dataframe_index_column()
        plot_labels = plots.get_registered_plot_keys()
        plot_data = plot_data[[time_label]+plot_labels]

        self.set_custom_ticks(plot_labels)

        if plot_data.empty:
            return

        time_stamps = len(plot_data[time_label])

        _x = []
        _active_plot_number = None
        for i, c in enumerate(plot_data):
            if i == 0:
                continue

            row_values = plot_data[c].to_numpy()
            row_values = row_values[~np.isnan(row_values)]
            row_values_length = len(row_values.tolist())
            _x.append(row_values_length)

            if active_plot == c:
                _active_plot_number = i - 1

        _y = list(range(len(plot_labels)))

        sc = pg.PlotCurveItem(y=_y, x=_x, pen=pg.mkPen(color=Color.BLUE.value, width=2))
        self.custom_plot.addItem(sc)

        color_map = [pg.mkBrush(Color.BLUE.value) if y != _active_plot_number else pg.mkBrush(Color.RED.value) for y in
                     _y]
        sc = pg.ScatterPlotItem(y=_y, x=_x, brush=color_map, name="# Samples (non-nan)")
        self.custom_plot.addItem(sc)

        self.custom_plot.setXRange(0, time_stamps)


class BoxplotGraphWidget(StatsGraph):
    """
    Graph to visualize the boxplot
    """

    custom_plot = None

    def __init__(self, *args, **kwargs):
        """
        Creates the custom plot

        Parameters
        ----------
        args
        kwargs
        """
        super(BoxplotGraphWidget, self).__init__(*args, **kwargs)
        self.custom_plot = self.addPlot(0, 0)
        self.legend = self.custom_plot.addLegend(colCount=6)
        self.vbs = self.addViewBox(row=1, col=0)
        self.vbs.setMaximumHeight(20)
        self.legend.setParentItem(self.vbs)
        self.legend.anchor((0, 0), (0, 0), offset=(55, -15))
        self.custom_plot.setMenuEnabled(False)
        self.custom_plot.setMouseEnabled(False, False)

    def draw_plot(self, active_plot):
        """
        Draws the statistic graph

        Parameters
        ----------
        active_plot
        """
        self.custom_plot.clear()

        plot_data = data.get_dataframe().copy()[active_plot]
        plot_data = plot_data.dropna(axis=0, how="any")

        if plot_data.empty:
            return

        median = np.median(plot_data)
        median_y = [3, 7]
        median_x = [median, median]

        min = np.min(plot_data)
        min_y = [3, 7]
        min_x = [min, min]

        max = np.max(plot_data)
        max_y = [3, 7]
        max_x = [max, max]

        lq = np.percentile(plot_data, 25)
        lq_y = [3, 7]
        lq_x = [lq, lq]

        uq = np.percentile(plot_data, 75)
        uq_y = [3, 7]
        uq_x = [uq, uq]

        lower_quartil_box = graphbox.GraphBoxItem(lq_x[0], lq_y[0], median_x[1], median_y[1], color=Color.BLUE.value,
                                                  fill=Color.BLUE.value)
        upper_quartil_box = graphbox.GraphBoxItem(median_x[0], median_y[0], uq_x[1], uq_y[1], color=Color.BLUE.value,
                                                  fill=Color.BLUE.value)

        connect_x = [min_x[0], max_x[0]]
        connect_y = [5, 5]

        iqr_x = [lq_x[0], uq_x[0]]
        iqr_y = [8, 8]

        median_line = pg.PlotCurveItem(x=median_x, y=median_y,
                                       pen=pg.mkPen(color=Color.RED.value, style=QtCore.Qt.DotLine, width=3),
                                       name="Median")
        min_line = pg.PlotCurveItem(x=min_x, y=min_y, pen=pg.mkPen(color=Color.BLUE.value, width=3))
        max_line = pg.PlotCurveItem(x=max_x, y=max_y, pen=pg.mkPen(color=Color.BLUE.value, width=3))

        iqr_line = pg.PlotCurveItem(x=iqr_x, y=iqr_y, pen=pg.mkPen(color=Color.GREEN.value, width=1),
                                    name="IQR")

        iqr_limit_left_x = [iqr_x[0], iqr_x[0]]
        iqr_limit_left_y = [7.75, 8.25]
        iqr_limit_left = pg.PlotCurveItem(x=iqr_limit_left_x, y=iqr_limit_left_y,
                                          pen=pg.mkPen(color=Color.GREEN.value, width=1))
        self.custom_plot.addItem(iqr_limit_left)

        iqr_limit_right_x = [iqr_x[1], iqr_x[1]]
        iqr_limit_right_y = [7.75, 8.25]
        iqr_limit_right = pg.PlotCurveItem(x=iqr_limit_right_x, y=iqr_limit_right_y,
                                           pen=pg.mkPen(color=Color.GREEN.value, width=1))
        self.custom_plot.addItem(iqr_limit_right)

        connector_line = pg.PlotCurveItem(x=connect_x, y=connect_y, pen=pg.mkPen(color=Color.BLUE.value, width=3))

        self.custom_plot.addItem(min_line)
        self.custom_plot.addItem(max_line)

        self.custom_plot.addItem(lower_quartil_box)
        self.custom_plot.addItem(upper_quartil_box)

        self.custom_plot.addItem(connector_line)

        self.custom_plot.addItem(iqr_line)

        self.custom_plot.addItem(median_line)
        self.custom_plot.hideAxis('left')

        self.custom_plot.setYRange(0, 10, padding=0)


class HistogramGraphWidget(StatsGraph):
    """
    Graph to visualize the histogram and KDE
    """
    custom_plot = None
    y_ranges = {}

    def __init__(self, *args, **kwargs):
        """
        Creates the custom plot

        Parameters
        ----------
        args
        kwargs
        """
        super(HistogramGraphWidget, self).__init__(*args, **kwargs)
        self.custom_plot = self.addPlot(0, 0)
        self.legend = self.custom_plot.addLegend(colCount=3)
        self.vbs = self.addViewBox(row=1, col=0)
        self.vbs.setMaximumHeight(20)
        self.legend.setParentItem(self.vbs)
        self.legend.anchor((0, 0), (0, 0), offset=(55, -15))
        self.custom_plot.setMenuEnabled(False)
        self.custom_plot.setMouseEnabled(False, False)

    def draw_plot(self, active_plot):
        """
        Draws the statistic graph

        Parameters
        ----------
        active_plot
        """
        self.custom_plot.clear()

        plot_data = data.get_dataframe().copy()[active_plot]
        plot_data = plot_data.dropna(axis=0, how="any")

        if plot_data.empty:
            return

        _y, _x = np.histogram(plot_data, density=True, bins=50)

        gaussian_kde = stats.gaussian_kde(plot_data)  # )bw_method=0.5)

        xs = np.arange(np.min(_x), np.max(_x), .1)
        ys = gaussian_kde(xs)

        mean = np.mean(plot_data)
        std = np.std(plot_data)

        mean_y = [0, *gaussian_kde(mean)]
        mean_x = [mean, mean]

        stdp2_x = [mean + 2 * std, mean + 2 * std]
        stdp2_y = [0, np.max(gaussian_kde(xs))]

        stdn2_x = [mean - 2 * std, mean - 2 * std]
        stdn2_y = [0, np.max(gaussian_kde(xs))]

        bargraphitem = pg.BarGraphItem(x0=_x[1:], x1=_x[:-1], height=_y, brush=Color.BLUE.value)

        mean_line = pg.PlotCurveItem(x=mean_x, y=mean_y, pen=pg.mkPen(color=Color.RED.value, width=3), name="μ")
        kde_line = pg.PlotDataItem(x=xs, y=ys,
                                   pen=pg.mkPen(color=Color.RED.value, style=QtCore.Qt.DotLine, width=2),
                                   name="KDE")

        stdp2 = pg.PlotCurveItem(x=stdp2_x, y=stdp2_y, pen=pg.mkPen(color=Color.GREEN.value, width=2),
                                 name="μ ± 2σ")
        stdn2 = pg.PlotCurveItem(x=stdn2_x, y=stdn2_y, pen=pg.mkPen(color=Color.GREEN.value, width=2))

        self.y_ranges[active_plot] = (0, np.amax([*_y, *ys]))

        self.custom_plot.addItem(bargraphitem)
        self.custom_plot.addItem(kde_line)
        self.custom_plot.addItem(mean_line)
        self.custom_plot.addItem(stdp2)
        self.custom_plot.addItem(stdn2)

        self.custom_plot.setYRange(*self.y_ranges[active_plot], padding=0)


class CorrelationGraphWidget(StatsGraph):
    """
    Graph to visualize the correlation between dimensions
    """
    custom_plot = None

    def __init__(self, *args, **kwargs):
        """
        Creates the custom plot

        Parameters
        ----------
        args
        kwargs
        """
        super(CorrelationGraphWidget, self).__init__(*args, **kwargs)
        self.custom_plot = self.addPlot(0, 0)
        self.custom_plot.setMenuEnabled(False)
        self.legend = self.custom_plot.addLegend(colCount=1)
        self.vbs = self.addViewBox(row=1, col=0)
        self.vbs.setMaximumHeight(20)
        self.legend.setParentItem(self.vbs)
        self.legend.anchor((0, 0), (0, 0), offset=(85, -15))
        self.custom_plot.setMouseEnabled(False, False)

    def set_custom_ticks(self, labels):
        """
        Sets custom ticks and replaces them with column labels

        Parameters
        ----------
        labels
        """
        ticks_y = [list(zip(range(len(labels)), labels))]
        ticks_x = [[(v, str(v)) for v in [-1, -0.7, -0.3, 0, 0.3, 0.7, 1]]]

        xax = self.custom_plot.getAxis('bottom')
        yax = self.custom_plot.getAxis('left')

        yax.setTicks(ticks_y)
        xax.setTicks(ticks_x)
        self.custom_plot.setXRange(-1, 1)

    def draw_plot(self, active_plot):
        """
        Draws the statistic graph

        Parameters
        ----------
        active_plot
        """
        self.custom_plot.clear()

        plot_data = data.get_dataframe().copy()
        time_label = data.get_dataframe_index_column()
        plot_data = plot_data[[time_label]+plots.get_registered_plot_keys()]
        plot_labels = [single_label for single_label in plot_data.columns.tolist()[1:] if single_label != active_plot]
        self.set_custom_ticks(plot_labels)

        if plot_data.empty:
            return

        if active_plot not in plot_data.columns.tolist():
            return

        corr = plot_data.corr()
        corr = corr.drop(labels=[active_plot, time_label], axis=0)

        column_data = corr[active_plot].to_numpy()
        columns = list(range(len(plot_labels)))

        _h_line = pg.InfiniteLine(pos=(0, 0), angle=90, movable=False, pen={'color': Color.BLUE.value, 'width': 2})
        self.custom_plot.addItem(_h_line, ignoreBounds=True)

        sc = pg.ScatterPlotItem(x=column_data, y=columns, brush=Color.BLUE.value,
                                name="Pearson correlation coefficient")
        self.custom_plot.addItem(sc)

        for i, v in enumerate(column_data):
            if v <= 0:
                _x = [v, 0]
            else:
                _x = [0, v]

            _y = [columns[i], columns[i]]

            line_item = pg.PlotDataItem(x=_x, y=_y, pen=pg.mkPen(color=Color.BLUE.value, width=2))
            self.custom_plot.addItem(line_item)
