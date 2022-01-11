import pandas as pd
import numpy as np
from scipy import ndimage
import hickle as hkl
from ndas.extensions import data, plots, physiologicallimits
from itertools import repeat


class PandasInterpolation:
    """
    Implements the interpolation methods already found in Pandas
    """
    def dataset_separation(self, dataframe):
        df_columns = dataframe.columns
        index_column = data.get_dataframe_index_column()
        reg_columns = plots.get_registered_plot_keys()
        other_series_columns = [col for col in df_columns if col in reg_columns]
        excluded_columns = [col for col in df_columns if col not in reg_columns+[index_column]]

        time_series = dataframe[index_column]
        other_series = dataframe[other_series_columns]
        non_series_data = dataframe[excluded_columns]

        interpolated_time_series = time_series.interpolate(limit_direction='both', limit_area=None)
        interpolated_non_series_data = non_series_data.interpolate(method='nearest', limit_direction='both', limit_area=None)

        return interpolated_time_series, other_series, interpolated_non_series_data

    def simple_interpolation(self, dataframe, lim_dir='forward', lim_are='inside'):
        interpolated_time_series, other_series, interpolated_non_series_data = self.dataset_separation(dataframe)
        interpolated_other_series = other_series.interpolate(limit_area=lim_are, limit_direction=lim_dir)

        result = pd.concat([interpolated_time_series, interpolated_other_series, interpolated_non_series_data], axis=1)
        return result[dataframe.columns]

    def polynomial_interpolation(self, dataframe, lim_dir='forward', lim_are='inside'):
        interpolated_time_series, other_series, interpolated_non_series_data = self.dataset_separation(dataframe)
        interpolated_other_series = other_series.interpolate(method='polynomial', order=2, limit_area=lim_are, limit_direction=lim_dir)

        result = pd.concat([interpolated_time_series, interpolated_other_series, interpolated_non_series_data], axis=1)
        return result[dataframe.columns]

    def spline_interpolation(self, dataframe, lim_dir='forward', lim_are='inside'):
        interpolated_time_series, other_series, interpolated_non_series_data = self.dataset_separation(dataframe)
        interpolated_other_series = other_series.interpolate(method='spline', order=1, limit_area=lim_are, limit_direction=lim_dir)

        result = pd.concat([interpolated_time_series, interpolated_other_series, interpolated_non_series_data], axis=1)
        return result[dataframe.columns]

    def neural_interpolation(self, dataframe_in, file_path="ndas\\dataimputationalgorithms\\NI_weights\\", **kwargs):
        # This has to be the same order as used in training
        pre_trained_columns = ["temperature", "heartrate", "respiration", "systemicsystolic", "systemicdiastolic", "systemicmean", "sao2", "end-tidal-co2", "central-venous-pressure"]

        interpolated_time_series, other_series, interpolated_non_series_data = self.dataset_separation(dataframe_in)

        transformed_other_series_dict = self.prepare_columns_for_keras_inter(other_series, pre_trained_columns)

        interpolated_other_series = []
        mask_min = []
        mask_max = []
        for col in other_series.columns:
            found_dt = physiologicallimits.get_physical_dt(col)
            if found_dt and found_dt.id in pre_trained_columns and isinstance(transformed_other_series_dict[found_dt.id], np.ndarray):
                weights = hkl.load(file_path+found_dt.id+"_weights.hkl")
                weighted_transformed_column = transformed_other_series_dict[found_dt.id] * np.squeeze(weights[0])
                interpolated_other_series.append(np.sum(weighted_transformed_column, axis=1)+np.squeeze(weights[1]))
                mask_min.append(found_dt.low)
                mask_max.append(found_dt.high)
            else:
                interpolated_other_series.append(other_series[col].interpolate(axis=0, limit_direction='both', limit_area=None).values)
                mask_min.append(-np.inf)
                mask_max.append(np.inf)
        interpolated_other_series = np.column_stack(interpolated_other_series)
        interpolated_other_series = np.clip(interpolated_other_series, mask_min, mask_max)
        interpolated_other_series = pd.DataFrame(data=interpolated_other_series, index=other_series.index, columns=other_series.columns)

        result = pd.concat([interpolated_time_series, interpolated_other_series, interpolated_non_series_data], axis=1)
        return result[dataframe_in.columns]

    def neural_interpolation_mask(self, dataframe, filepath ="ndas\\dataimputationalgorithms\\NI_weights\\", **kwargs):
        mask = np.isnan(dataframe.values)
        mask_edges = np.zeros(dataframe.values.shape)
        for col in dataframe.columns:
            mask_edges[dataframe[col].first_valid_index():dataframe[col].last_valid_index(), dataframe.columns.get_loc(col)] = 1
        imp_res = self.neural_interpolation(dataframe, filepath)
        return imp_res.where(mask & (mask_edges > 0), dataframe)

    def neural_interpolation_mask_round(self, dataframe, filepath ="ndas\\dataimputationalgorithms\\NI_weights\\", **kwargs):
        mask = np.isnan(dataframe.values)
        mask_edges = np.zeros(dataframe.values.shape)
        for col in dataframe.columns:
            mask_edges[dataframe[col].first_valid_index():dataframe[col].last_valid_index(), dataframe.columns.get_loc(col)] = 1
        imp_res = self.neural_interpolation(dataframe, filepath)
        for col in imp_res.columns:
            if col in plots.get_registered_plot_keys():
                decimals = 0
                if physiologicallimits.get_physical_dt(col) and physiologicallimits.get_physical_dt(col).id == 'temperature':
                    decimals = 1
                imp_res[col] = imp_res[col].round(decimals)
        return imp_res.where(mask & (mask_edges > 0), dataframe)

    def prepare_columns_for_keras_inter(self, dataframe_in, pre_trained_columns=None):
        if pre_trained_columns is None:
            pre_trained_columns = ["temperature", "heartrate", "respiration", "systemicsystolic", "systemicdiastolic", "systemicmean", "sao2", "end-tidal-co2", "central-venous-pressure"]
        dafr = dataframe_in.interpolate(axis=0, limit_direction='both', limit_area=None)
        length = len(dafr.index)
        dict_of_inputs = {}
        dict_of_other_medians = {}

        for pre_col in pre_trained_columns:
            found_column = next((col for col in dafr.columns if physiologicallimits.is_alias_of(col, pre_col)), False)
            if found_column and physiologicallimits.get_physical_dt(pre_col).low < 0:
                dafr[found_column] -= physiologicallimits.get_physical_dt(pre_col).low

            if found_column and all(map(any, repeat(iter((x != 0 and not np.isnan(x)) for x in dafr[found_column]), 10))):
                phys_lim = physiologicallimits.get_physical_dt(pre_col)
                datum = [dafr[found_column].clip(phys_lim.low, phys_lim.high).values]

                for j in range(8):
                    datum.append(ndimage.gaussian_filter1d(datum[0], sigma=0.75 * (2 ** j), mode='nearest'))

                for k in range(3):
                    datum.append(ndimage.uniform_filter1d(datum[0], 6 * (2 ** k), mode='nearest'))

                dict_of_inputs[pre_col] = np.column_stack(datum)
                dict_of_other_medians[pre_col] = datum[4] / np.nanmedian(datum[4])

            else:
                dict_of_inputs[pre_col] = 0
                dict_of_other_medians[pre_col] = np.ones(length)

        for k in dict_of_inputs.keys():
            if isinstance(dict_of_inputs[k], np.ndarray):
                phys_lim = physiologicallimits.get_physical_dt(k)
                median = np.nanmedian(dict_of_inputs[k][:, 4])
                list_of_outside_info = []

                for j in pre_trained_columns:
                    list_of_outside_info.append(dict_of_other_medians[j] * median)

                if phys_lim.low < 0:
                    dict_of_inputs[k] = np.clip((np.column_stack([dict_of_inputs[k]] + list_of_outside_info) + phys_lim.low), phys_lim.low, phys_lim.high)

                else:
                    dict_of_inputs[k] = np.clip(np.column_stack([dict_of_inputs[k]] + list_of_outside_info), phys_lim.low, phys_lim.high)

        return dict_of_inputs
