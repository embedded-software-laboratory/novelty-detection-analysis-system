import pandas as pd
import numpy as np


class PandasInterpolation:
    """
    Implements the interpolation methods already found in Pandas
    """

    def simple_interpolation(self, dataframe, lim_dir='forward', lim_are='inside'):
        time_series = dataframe.iloc[:, 0:1]
        other_series = dataframe.iloc[:, 1:10]
        non_series_data = dataframe.iloc[:, 10:]
        interpolated_time_series = time_series.interpolate()
        interpolated_other_series = other_series.interpolate(limit_area=lim_are, limit_direction=lim_dir)
        interpolated_non_series_data = non_series_data.interpolate(method='nearest')

        result = pd.concat([interpolated_time_series, interpolated_other_series, interpolated_non_series_data], axis=1)
        return result

    def polynomial_interpolation(self, dataframe, lim_dir='forward', lim_are='inside'):
        time_series = dataframe.iloc[:, 0:1]
        other_series = dataframe.iloc[:, 1:10]
        non_series_data = dataframe.iloc[:, 10:]

        interpolated_time_series = time_series.interpolate()
        interpolated_other_series = other_series.interpolate(method='polynomial', limit_area= lim_are, limit_direction=lim_dir, order=3)
        interpolated_non_series_data = non_series_data.interpolate(method='nearest')

        result = pd.concat([interpolated_time_series, interpolated_other_series, interpolated_non_series_data], axis=1)
        return result

    def spline_interpolation(self, dataframe, lim_dir='forward', lim_are='inside'):
        time_series = dataframe.iloc[:, 0:1]
        other_series = dataframe.iloc[:, 1:10]
        non_series_data = dataframe.iloc[:, 10:]

        interpolated_time_series = time_series.interpolate()
        interpolated_other_series = other_series.interpolate(method='spline', limit_area= lim_are, limit_direction=lim_dir, order=1)
        interpolated_non_series_data = non_series_data.interpolate(method='nearest')

        result = pd.concat([interpolated_time_series, interpolated_other_series, interpolated_non_series_data], axis=1)
        return result
