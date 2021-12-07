import numpy as np
from scipy import stats


def get_number_data_points(g) -> str:
    """
    Returns the number of data points in a graph

    Parameters
    ----------
    g
    """
    size = g.main_dot_plot.y_data.size
    if size:
        r_value = size
    else:
        r_value = 0
    return str(r_value)


def get_number_novelties(g, klass) -> str:
    """
    Returns the number of novelties in a graph

    Parameters
    ----------
    g
    klass
    """
    if klass == 1:
        if g.main_dot_plot.novelties:
            r_value = sum(1 for e in g.main_dot_plot.novelties if e not in [-2, -1, 0, 2])
        else:
            r_value = 0
    elif klass == 2:
        if g.main_dot_plot.novelties:
            r_value = sum(1 for e in g.main_dot_plot.novelties if e not in [-2, -1, 0, 1])
        else:
            r_value = 0
    elif klass == 'all':
        if g.main_dot_plot.novelties:
            r_value = sum(1 for e in g.main_dot_plot.novelties if e not in [-2, -1, 0])
        else:
            r_value = 0
    else:
        r_value = 0

    return str(r_value)


def get_minimum_dp(g) -> str:
    """
    Returns the minimum value in graph

    Parameters
    ----------
    g
    """
    minimum = g.main_dot_plot.y_data[~np.isnan(g.main_dot_plot.y_data)].min()
    if minimum != np.nan:
        r_value = minimum
    else:
        r_value = "NaN"
    return str(r_value)


def get_maximum_dp(g) -> str:
    """
    Returns maximum number in graph

    Parameters
    ----------
    g
    """
    maximum = g.main_dot_plot.y_data[~np.isnan(g.main_dot_plot.y_data)].max()
    if maximum != np.nan:
        r_value = maximum
    else:
        r_value = "NaN"
    return str(r_value)


def get_range_dp(g) -> str:
    """
    Returns the range of the data in graph

    Parameters
    ----------
    g
    """
    _min = get_minimum_dp(g)
    _max = get_maximum_dp(g)
    return "[%s - %s]" % (_min, _max)


def get_mean(g) -> str:
    """
    Returns the mean of the data in graph

    Parameters
    ----------
    g
    """
    return str(np.round_(np.mean(g.main_dot_plot.y_data), decimals=3))


def get_std(g) -> str:
    """
    Returns the std of the data in graph

    Parameters
    ----------
    g
    """
    return str(np.round_(np.std(g.main_dot_plot.y_data), decimals=3))


def get_median_absolute_deviation(g) -> str:
    """
    Returns the MAD in data in graph

    Parameters
    ----------
    g
    """
    return str(np.round_(stats.median_abs_deviation(g.main_dot_plot.y_data, nan_policy='omit'), decimals=3))
