import numpy as np
from numpy import arange
from scipy import interpolate, ndimage
from scipy.optimize import curve_fit


def get_linear_fitting_line(x: np.ndarray, y: np.ndarray):
    """
    Calculates a linear fitting line through data

    Parameters
    ----------
    x
    y
    """
    if len(x) != len(y):
        print("fail: x and y with different length")
        return False

    not_nan_values = np.isfinite(y)
    y = y[not_nan_values]
    x = x[not_nan_values]

    y_mean = np.mean(y)
    x_mean = np.mean(x)

    # calculate slope of the line of best fit
    t_sum = 0.0
    div = 0.0

    for i, v in enumerate(y):
        t_sum = t_sum + ((x[i] - x_mean) * (y[i] - y_mean))
        div = div + ((x[i] - x_mean) * (x[i] - x_mean))

    m = t_sum / div

    # compute y-intercept of the line
    b = y_mean - (m * x_mean)

    def y_(_x, _m, _b):
        return (_m * _x) + _b

    x_line = [x[0], x[len(x) - 1]]
    y_line = [y_(x[0], m, b), y_(x[len(x) - 1], m, b)]

    return x_line, y_line


# polynomial regression
def get_polynomial_fitting_curve(x: np.ndarray, y: np.ndarray):
    """
    Calculates a polynomial fitting curve through data

    Parameters
    ----------
    x
    y
    """

    def objective(_x, _a, _b, _c, _d, _e, _f, _g, _h, _i, _j):
        return (_a * _x) + (_b * _x ** 2) + (_c * _x ** 3) + (_d * _x ** 4) + (_e * _x ** 5) + (_f * _x ** 6) + (
                _g * _x ** 7) + (
                       _h * _x ** 8) + (_i * _x ** 9) + (
                   _j)

    if len(x) != len(y):
        print("fail: x and y with different length")
        return False

    y = y.astype(float)
    x = x.astype(float)

    not_nan_values = np.isfinite(y)
    y = y[not_nan_values]
    x = x[not_nan_values]

    popt, _ = curve_fit(objective, x, y)
    a, b, c, d, e, f, g, h, i, j = popt

    x_line = arange(x[0], x[len(x) - 1], 1)
    y_line = objective(x_line, a, b, c, d, e, f, g, h, i, j)

    return x_line, y_line


def get_spline_interpolation_curve(x: np.ndarray, y: np.ndarray, smoothing_factor=2000, gaussian_sigma=10):
    """
    Calculates a spline interpolation through data

    Parameters
    ----------
    x
    y
    smoothing_factor
    gaussian_sigma
    """
    if len(x) != len(y):
        print("fail: x and y with different length")
        return False

    y = y.astype(float)
    x = x.astype(float)

    not_nan_values = np.isfinite(y)
    y = y[not_nan_values]
    x = x[not_nan_values]

    spline = interpolate.UnivariateSpline(x, ndimage.gaussian_filter1d(y, gaussian_sigma), s=smoothing_factor)

    return x, spline(x)
