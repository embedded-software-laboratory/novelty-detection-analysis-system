import numpy as np
import pandas as pd
import pyqtgraph as pg

from ndas.extensions import data, algorithms
from ndas.mainwindow import graphlayoutwidget
from ndas.misc import graphbox
from ndas.misc.colors import Color
from ndas.utils import logger, regression_analysis

registered_plots = {}
plot_layout_widget = None

draw_outlier_series_box = False
outlier_series_threshold = 5
exclusion_substrings = []


def init_graphs(config):
    """
    Initializes the plot helper

    Parameters
    ----------
    config
    """
    global plot_layout_widget, draw_outlier_series_box, outlier_series_threshold, exclusion_substrings

    if config["use_dark_mode"]:
        pg.setConfigOption('background', 'k')
        pg.setConfigOption('foreground', 'w')
    else:
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

    draw_outlier_series_box = config["draw_outlier_series_box"]
    outlier_series_threshold = config["outlier_series_threshold"]
    exclusion_substrings = [str(v) for v in config["exclusion_substrings"]]

    pg.setConfigOptions(antialias=True)
    plot_layout_widget = graphlayoutwidget.GraphLayoutWidget()


def register_plot(name: str, x_data: any, y_data: any, x_label: str, y_label: str):
    """
    Registers a new plot with data

    Parameters
    ----------
    name
    x_data
    y_data
    x_label
    y_label

    """
    global registered_plots

    if name in registered_plots:
        logger.plots.error("The plot named %s is already registered." % name)
        return False

    if not (len(x_data) == len(y_data)):
        logger.plots.error("X and Y data have not the same length. Failed to add plot %s" % name)
        return False

    if isinstance(x_data, list):
        x_data_numpy = np.ndarray(x_data, dtype=np.float32)
        y_data_numpy = np.ndarray(y_data, dtype=np.float32)
        x_data = pd.Series(x_data)
        y_data = pd.Series(y_data)
    elif isinstance(x_data, pd.Series):
        x_data_numpy = x_data.to_numpy()
        y_data_numpy = y_data.to_numpy()
    else:
        logger.plots.error("Unsupported data format (%s). Failed to add plot %s" % (str(type(x_data)), name))
        return False

    plot_scatter = pg.ScatterPlotItem(x=x_data_numpy, y=y_data_numpy, brush=pg.mkBrush(Color.BLUE.value),
                                      pen=pg.mkPen(color='w', width=0.4),
                                      size=10, tip=None)
    plot_line = pg.PlotDataItem(x=x_data_numpy, y=y_data_numpy, pen=Color.BLUE.value)

    main_dot_plot_item = SinglePointPlotItem(plot_item_name=name, x_data=x_data, y_data=y_data,
                                             plot_item=plot_scatter)
    main_line_plot_item = SingleLinePlotItem(plot_item_name=name, x_data=x_data, y_data=y_data, plot_item=plot_line)

    registered_plots[name] = MultiPlot(name, x_label, y_label, main_dot_plot_item, main_line_plot_item)


def update_plot(name: str, x_data: any, y_data: any):
    """
        Registers a new plot with data

        Parameters
        ----------
        name
        x_data
        y_data

        """
    global registered_plots

    if name not in registered_plots:
        logger.plots.error("The plot named %s isn't yet registered." % name)
        return False

    if not (len(x_data) == len(y_data)):
        logger.plots.error("X and Y data have not the same length. Failed to add plot %s" % name)
        return False

    if isinstance(x_data, list):
        x_data_numpy = np.ndarray(x_data, dtype=np.float32)
        y_data_numpy = np.ndarray(y_data, dtype=np.float32)
        x_data = pd.Series(x_data)
        y_data = pd.Series(y_data)
    elif isinstance(x_data, pd.Series):
        x_data_numpy = x_data.to_numpy()
        y_data_numpy = y_data.to_numpy()
    else:
        logger.plots.error("Unsupported data format (%s). Failed to add plot %s" % (str(type(x_data)), name))
        return False

    reg_plot = registered_plots[name]
    novelty_data = _match_novelties_to_timedata(x_data, algorithms.get_detected_novelties(name))

    reg_plot.main_dot_plot.x_data = x_data
    reg_plot.main_dot_plot.y_data = y_data
    reg_plot.main_dot_plot.plot_item.setData(x_data_numpy, y_data_numpy)
    reg_plot.main_dot_plot.plot_item.setBrush(_get_brush_map(novelty_data))
    reg_plot.main_dot_plot.plot_item.setPen(_get_pen_map(novelty_data))
    reg_plot.main_line_plot.x_data = x_data
    reg_plot.main_line_plot.y_data = y_data
    reg_plot.main_line_plot.plot_item.setData(x_data_numpy, y_data_numpy)


def add_line(plot_name, line_name, x_data, y_data, color=None, p_width=3):
    """
    Adds a line to an existing plot

    Parameters
    ----------
    plot_name
    line_name
    x_data
    y_data
    color
    p_width

    """
    global registered_plots

    if registered_plots[plot_name]:

        if color is None:
            color = Color.RED

        plot_line = pg.PlotCurveItem(x=x_data.to_numpy(), y=y_data.to_numpy(),
                                     pen=pg.mkPen(color=color.value, width=p_width))
        plot_item = SingleLinePlotItem(plot_item_name=line_name, x_data=x_data, y_data=y_data, plot_item=plot_line)
        registered_plots[plot_name].supplementary_plots.append(plot_item)
        logger.plots.debug("Added line %s to plot %s" % (line_name, plot_name))
    else:
        logger.plots.error("add_line: Argument plot_name is unknown.")


def add_infinite_line(plot_name, line_name, y, color=None, alpha=0.5):
    """
    Adds an infinite line to an existing plot

    Parameters
    ----------
    plot_name
    line_name
    y
    color
    alpha
    """
    global registered_plots

    if plot_name in registered_plots:

        if color is None:
            color = Color.RED

        inf_line = pg.InfiniteLine(angle=0, movable=False, name=line_name, label=line_name,
                                   pen={'color': color.value, 'alpha': alpha}, pos=y)
        plot_item = SingleLinePlotItem(plot_item_name=line_name, y_data=y, plot_item=inf_line)

        registered_plots[plot_name].supplementary_plots.append(plot_item)
        logger.plots.debug("Added line %s to plot %s" % (line_name, plot_name))
    else:
        logger.plots.error("add_infinite_line: Argument plot_name is unknown.")


def add_linear_regression_line():
    """
    Adds a linear regression line to the current plot
    """
    global registered_plots
    k, _ = get_active_plot()

    if not k:
        logger.plots.error("add_linear_regression_line: No active plot found.")
    else:
        x_data_np = registered_plots[k].main_dot_plot.x_data.to_numpy()
        y_data_np = registered_plots[k].main_dot_plot.y_data.to_numpy()

        regression_line_x, regression_line_y = regression_analysis.get_linear_fitting_line(x_data_np, y_data_np)
        add_line(k, "Linear Fit Line", pd.Series(regression_line_x), pd.Series(regression_line_y))
        update_plot_view()


def add_spline_regression_curve():
    """
    Adds a spline regression curve to the current plot
    """
    global registered_plots
    k, _ = get_active_plot()

    if not k:
        logger.plots.error("add_spline_regression_curve: No active plot found.")
    else:
        x_data_np = registered_plots[k].main_dot_plot.x_data.to_numpy()
        y_data_np = registered_plots[k].main_dot_plot.y_data.to_numpy()

        regression_curve_x, regression_curve_y = regression_analysis.get_spline_interpolation_curve(x_data_np,
                                                                                                    y_data_np)
        add_line(k, "Spline Fit Curve", pd.Series(regression_curve_x), pd.Series(regression_curve_y))
        update_plot_view()


def get_registered_plot_keys():
    """
    Returns the keys of registered plots excluding "LOS"
    """
    global registered_plots
    return [k for k, v in registered_plots.items() if "LOS" not in k]


def register_available_plots(current_active_plot=None):
    """
    Registers all plots for the available data

    Parameters
    ----------
    current_active_plot

    """
    global registered_plots, exclusion_substrings
    registered_plots = {}

    plot_layout_widget.clear_plots()

    df = data.get_dataframe()
    columns = data.get_dataframe_columns()
    labels = data.get_dataframe_labels()

    for idx, col in enumerate(columns[1:]):
        temp_data = df[col]
        if not temp_data.dropna().empty and not any(pattern in labels[idx + 1] for pattern in exclusion_substrings):
            temp_time = df[columns[0]]
            temp_df = pd.DataFrame({columns[0]: temp_time, col: temp_data})
            temp_df.dropna(axis=0, inplace=True)
            register_plot(col, temp_df[columns[0]], temp_df[col], labels[0], labels[idx + 1])

    if current_active_plot is not None:
        set_plot_active(current_active_plot)
    else:
        registered_plots[list(registered_plots.keys())[0]].active = True


def update_available_plots():
    df = data.get_dataframe()
    columns = data.get_dataframe_columns()
    for col in get_registered_plot_keys():
        temp_data = df[col]
        if not temp_data.dropna().empty:
            temp_time = df[columns[0]]
            temp_df = pd.DataFrame({columns[0]: temp_time, col: temp_data})
            temp_df.dropna(axis=0, inplace=True)
            update_plot(col, temp_df[columns[0]], temp_df[col])


def set_plot_active(name):
    """
    Sets a plot as active

    Parameters
    ----------
    name

    """
    global registered_plots
    for k, v in registered_plots.items():
        registered_plots[k].active = False
    if name in registered_plots:
        registered_plots[name].active = True
    update_plot_view()


def set_plot_line_status(name, status):
    """
    Enables or disables the line plot

    Parameters
    ----------
    name
    status

    """
    if name in registered_plots:
        registered_plots[name].line_plot_visible = status


def set_plot_point_status(name, status):
    """
    Enables or disables the scatter plot

    Parameters
    ----------
    name
    status

    """
    if name in registered_plots:
        registered_plots[name].dot_plot_visible = status


def update_plot_view():
    """
    Redraws the current plot
    """
    global registered_plots
    plot_name, _ = get_active_plot()

    if not plot_name:
        plot_layout_widget.clear_plots()
    else:
        logger.plots.debug("Drawing plot: %s" % registered_plots[plot_name].plot_name)
        plot_layout_widget.draw_registered_plot(registered_plots[plot_name])


def get_active_plot():
    """
    Returns the currently active plot
    """
    global registered_plots

    for k, v in registered_plots.items():
        if registered_plots[k].active:
            return k, registered_plots[k]
    return False, False


def _get_brush_map(novelty_data: list):
    """
    Gets a brush map for colorizing scatter plot points

    Parameters
    ----------
    novelty_data
    """
    color_map = {-9: Color.PURPLE.value, -8: Color.PINK.value, -2: Color.GREY.value, -1: Color.GREEN.value, 0: Color.BLUE.value, 1: Color.RED.value,
                 2: Color.YELLOW.value}
    return [pg.mkBrush(color_map[novelty_point]) for novelty_point in novelty_data]


def _get_pen_map(novelty_data: list):
    """
    Gets a pen map for colorizing the pen of scatter plot points

    Parameters
    ----------
    novelty_data
    """
    return [pg.mkPen(color='w', width=0.4) for _ in novelty_data]


def _match_novelties_to_timedata(x_data: pd.Series, novelties: dict):
    """
    Matches detection results to time data

    Parameters
    ----------
    x_data
    novelties
    """
    novelty_data = []
    for index, x in x_data.iteritems():
        if x in novelties:
            novelty_data.append(novelties[x])
        else:
            novelty_data.append(0)
    return novelty_data


def _match_timedata_to_novelties(x_data: pd.Series, novelties: dict):
    """
    Matches timedata to novelty data list

    Parameters
    ----------
    x_data
    novelties
    """
    novelty_data = {}
    for index, x in enumerate(x_data.items()):
        novelty_data[x[1]] = novelties[index]
    return novelty_data


def _get_outlier_boxes(novelty_list: list, threshold: int, x_data: np.ndarray, y_data: np.ndarray):
    """
    Boxifies novelties if certain threshold is passed

    Parameters
    ----------
    novelty_list
    threshold
    x_data
    y_data
    """
    outlier_box_items = []

    _outlier_counter = 0
    x1, x2, y1, y2 = (None, None, None, None)

    for i, v in enumerate(novelty_list):
        if v == 1:
            _outlier_counter = _outlier_counter + 1

            if x1 is None or x_data[i] < x1:
                x1 = x_data[i]

            if y1 is None or y_data[i] < y1:
                y1 = y_data[i]

            if x2 is None or x_data[i] > x2:
                x2 = x_data[i]

            if y2 is None or y_data[i] > y2:
                y2 = y_data[i]

        else:

            if _outlier_counter >= threshold:
                box_plot_item = SinglePlotItem("outlier_series")
                box_plot_item.plot_item = graphbox.GraphBoxItem(x1, y1, x2, y2)
                outlier_box_items.append(box_plot_item)

            _outlier_counter = 0
            x1, x2, y1, y2 = (None, None, None, None)

    return outlier_box_items


def add_plot_novelties(plot_name: str, novelties: dict):
    """
    Adds colorized novelties to a plot

    Parameters
    ----------
    plot_name
    novelties
    """
    global registered_plots, draw_outlier_series_box, outlier_series_threshold

    if not plot_name:
        return False

    if not novelties:
        return False

    if plot_name not in registered_plots:
        return False

    reg_plot = registered_plots[plot_name]
    novelty_data = _match_novelties_to_timedata(reg_plot.main_dot_plot.x_data, novelties)

    x_data = reg_plot.main_dot_plot.x_data
    y_data = reg_plot.main_dot_plot.y_data

    x_data_numpy = x_data.to_numpy()
    y_data_numpy = y_data.to_numpy()

    plot_scatter = pg.ScatterPlotItem(x=x_data_numpy, y=y_data_numpy, brush=_get_brush_map(novelty_data),
                                      pen=_get_pen_map(novelty_data),
                                      size=10, hoverable=True)
    plot_line = pg.PlotDataItem(x=x_data_numpy, y=y_data_numpy, pen=Color.BLUE.value)

    primary_point_plot_item = SinglePointPlotItem(plot_item_name=reg_plot.main_dot_plot.plot_item_name, x_data=x_data,
                                                  y_data=y_data, plot_item=plot_scatter, novelties=novelty_data)
    primary_line_plot_item = SingleLinePlotItem(plot_item_name=reg_plot.main_line_plot.plot_item_name, x_data=x_data,
                                                y_data=y_data, plot_item=plot_line)

    reg_plot.main_dot_plot = primary_point_plot_item
    reg_plot.main_line_plot = primary_line_plot_item

    reg_plot.supplementary_plots = [singleplotitem for singleplotitem in reg_plot.supplementary_plots if not isinstance(singleplotitem.plot_item, graphbox.GraphBoxItem)]

    if draw_outlier_series_box:
        for box_item in _get_outlier_boxes(novelty_data, outlier_series_threshold, x_data_numpy, y_data_numpy):
            logger.plots.debug("Added outlier series box")
            reg_plot.supplementary_plots.append(box_item)


def format_for_save():
    """
    Formats the plots for the save file
    """
    global registered_plots
    novelties = {}
    for k, v in registered_plots.items():
        if registered_plots[k].main_dot_plot.novelties is not None:
            novelties[registered_plots[k].main_dot_plot.plot_item_name] = registered_plots[k].main_dot_plot.novelties
    return novelties


def restore_from_save(data):
    """
    Restore the plots from a save file

    Parameters
    ----------
    data
    """
    global registered_plots
    register_available_plots()

    for key, value in data.items():
        if key in registered_plots:
            add_plot_novelties(key, _match_timedata_to_novelties(registered_plots[key].main_dot_plot.x_data, data[key]))
    update_plot_view()

def toggleTooltipFlag(flag):
    plot_layout_widget.toggleTooltipFlag(flag)

def toggleLabelFlag(flag):
    plot_layout_widget.toggleLabelFlag(flag)

class SinglePlotItem:
    """
    A plot consisting of multiple plot items
    """

    def __init__(self,
                 plot_item_name: str,
                 x_data: any = None,
                 y_data: any = None,
                 plot_item: any = None,
                 novelties: list = None,
                 color: 'Color' = None):
        super(SinglePlotItem).__init__()
        self.plot_item_name = plot_item_name
        self.x_data = x_data
        self.y_data = y_data
        self.plot_item = plot_item
        self.novelties = novelties
        self.color = color


class SingleLinePlotItem(SinglePlotItem):
    """
    A line plot item
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SinglePointPlotItem(SinglePlotItem):
    """
    A scatter plot item
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MultiPlot:
    """
    A multiplot consisting of a line and a scatter plot
    """

    def __init__(self,
                 plot_name: str,
                 x_label: str,
                 y_label: str,
                 main_dot_plot: SinglePlotItem,
                 main_line_plot: SinglePlotItem):
        super().__init__()
        self.plot_name = plot_name
        self.x_label = x_label
        self.y_label = y_label
        self.main_dot_plot = main_dot_plot
        self.main_line_plot = main_line_plot
        self.dot_plot_visible = True
        self.line_plot_visible = False
        self.supplementary_plots = []
        self.active = False

