import pandas as pd
import numpy as np
from scipy import ndimage
import hickle as hkl


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

    def neural_interpolation(self, dataframe_in, file_path="C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\", **kwargs):
        dataframe = dataframe_in.copy(True)
        mask_min = [28, 20, 5, 35, 20, 25, 60, 18, -10]
        mask_max = [44, 220, 50, 280, 180, 210, 100, 49, 50]
        time_series = dataframe.iloc[:, 0:1]
        other_series = dataframe.iloc[:, 1:10]
        non_series_data = dataframe.iloc[:, 10:]

        list_weight_name_pre=['temp', 'hr', 'rr', 'bps', 'bpd', 'bpm', 'o2', 'co2', 'cvp']

        interpolated_time_series = time_series.interpolate()
        interpolated_non_series_data = non_series_data.interpolate(method='nearest')

        transformed_other_series = self.df_to_list_of_keras_inter_inputs(dataframe)
        interpolated_other_series = []
        for i in range(9):
            if isinstance(transformed_other_series[i], np.ndarray):
                weights = hkl.load(file_path+list_weight_name_pre[i]+"_weights.hkl")
                weighted_transformed_column = transformed_other_series[i] * np.squeeze(weights[0])
                interpolated_other_series.append(np.sum(weighted_transformed_column, axis=1)+np.squeeze(weights[1]))
            else:
                interpolated_other_series.append(other_series.iloc[:, i])
        interpolated_other_series = np.column_stack(interpolated_other_series)
        interpolated_other_series = np.clip(interpolated_other_series, mask_min, mask_max)
        interpolated_other_series = pd.DataFrame(data=interpolated_other_series, index=other_series.index, columns=other_series.columns)

        result = pd.concat([interpolated_time_series, interpolated_other_series, interpolated_non_series_data], axis=1)
        return result

    def neural_interpolation_mask(self, dataframe, filepath ="C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\", **kwargs):
        mask = np.isnan(dataframe.values)
        mask_edges = np.zeros(dataframe.values.shape)
        for col in dataframe.columns:
            mask_edges[dataframe[col].first_valid_index():dataframe[col].last_valid_index(), dataframe.columns.get_loc(col)] = 1
        imp_res = self.neural_interpolation(dataframe, filepath)
        return imp_res.where(mask & (mask_edges > 0), dataframe)

    def neural_interpolation_mask_round(self, dataframe, filepath ="C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\", **kwargs):
        mask = np.isnan(dataframe.values)
        mask_edges = np.zeros(dataframe.values.shape)
        for col in dataframe.columns:
            mask_edges[dataframe[col].first_valid_index():dataframe[col].last_valid_index(), dataframe.columns.get_loc(col)] = 1
        column_names = dataframe.columns[1:10]
        imp_res = self.neural_interpolation(dataframe, filepath)
        imp_res = imp_res.round(pd.Series([1, 0, 0, 0, 0, 0, 0, 0, 0], index=column_names))
        return imp_res.where(mask & (mask_edges > 0), dataframe)

    def df_to_list_of_keras_inter_inputs(self, dataframe_in):
        mask_min = [28, 20, 5, 35, 20, 25, 60, 18, -10]
        mask_max = [44, 220, 50, 280, 180, 210, 100, 49, 50]
        dafr = dataframe_in.interpolate(axis=0, limit_direction='both', limit_area=None)
        xv = dafr.values[:, 0]
        non_empty = np.count_nonzero(~np.isnan(dafr.values[:, 1:10]), axis=0) > 10
        non_empty = non_empty & (np.count_nonzero(dafr.values[:, 1:10], axis=0) > 1)
        list_of_inputs = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        list_of_other_medians = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(9):
            if non_empty[i]:
                datum = [np.clip(dafr.values[:, i + 1], mask_min[i], mask_max[i])]
                for j in range(8):
                    datum.append(ndimage.gaussian_filter1d(datum[0], sigma=0.75 * (2 ** j), mode='nearest'))
                for k in range(3):
                    datum.append(ndimage.uniform_filter1d(datum[0], 6 * (2 ** k), mode='nearest'))
                list_of_inputs[i] = np.column_stack(datum)
                list_of_other_medians[i] = datum[4] / np.nanmedian(datum[4])
            else:
                list_of_other_medians[i] = np.ones(len(xv))
        for i in range(9):
            if non_empty[i]:
                median = np.nanmedian(list_of_inputs[i][:, 4])
                list_of_outside_info = []
                for j in range(9):
                    list_of_outside_info.append(list_of_other_medians[j] * median)
                list_of_inputs[i] = np.clip(
                    np.column_stack([list_of_inputs[i]] + list_of_outside_info),
                    mask_min[i], mask_max[i])
        return list_of_inputs
