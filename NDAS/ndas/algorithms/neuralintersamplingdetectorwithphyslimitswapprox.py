import numpy as np
import pandas as pd
from ndas.algorithms.basedetector import BaseDetector  # Import the basedetector class
from ndas.misc.parameter import ArgumentType
from ndas.dataimputationalgorithms.base_imputation import BaseImputation
from ndas.extensions import plots, physiologicallimits
from kneed import KneeLocator
import logging


class NeuralInterSamplingDetectorWithPhysicalLimitsWithApprox(BaseDetector):
    """
    A detector for physiological limits
    """

    def __init__(self, *args, **kwargs):
        """ Init the detector and register additional arguments

        Use the register_parameter method to define additional required arguments to run this detector.

        Parameters
        ----------
        args
        kwargs
        """
        super(NeuralInterSamplingDetectorWithPhysicalLimitsWithApprox, self).__init__(*args, **kwargs)
        self.register_parameter("Aggressiveness", ArgumentType.FLOAT, 0.1, 0, 2, tooltip="Parameter for the aggressiveness of error detection.")

    def detect(self, datasets_in, **kwargs) -> dict:
        """ Return novelties in given dataset

        This method needs to be overwritten from the Basedetector class.
        It has to contain the required logic for novelty detection.

        Parameters
        ----------
        datasets_in
            The dataset
        kwargs
            Additional parameters

        Returns
        -------
        novelties : dict
            A dict of plots with novelties with observation offset as key
        """
        datasets = datasets_in.copy(True)
        time_column = datasets.columns[0]
        used_columns = [col for col in datasets.columns if col in plots.get_available_plot_keys(datasets)]
        self.s_value = 0.25 + (float(kwargs["Aggressiveness"])/4)

        thresholds = {k: ((self.get_physiological_information(k).nisd_pla_thr / (2*self.s_value)) if self.get_physiological_information(k) else (self.get_physiological_information('default').nisd_pla_thr / (2*self.s_value))) for k in used_columns}
        logging.debug("Using Thresholds: "+str(thresholds))

        current_diffs = {}
        clipped_dataset = datasets.copy(deep=True)
        for c in used_columns:
            phys_info = self.get_physiological_information(c)
            if phys_info:
                clipped_dataset[c].where(clipped_dataset[c].between(phys_info.low, phys_info.high), other=np.nan, inplace=True)

        # Iteration 1
        imputation_accumulated = pd.DataFrame(0, columns=datasets.columns, index=datasets.index)
        current_status = 5.0
        self.signal_percentage(int(current_status) % 100)
        for i in range(50):
            temp_imputation = pd.DataFrame(np.nan, columns=datasets.columns, index=datasets.index)
            mask = np.random.random((datasets.values.shape[0], datasets.values.shape[1]-1))

            imputation_res = BaseImputation().base_imputation(dataframe=clipped_dataset.where(np.concatenate((np.full((datasets.values.shape[0], 1),True), mask>=(1/3)), axis=1), other=np.nan), method_string='neural inter mask')
            loc_mask=np.concatenate((np.full((datasets.values.shape[0], 1),False), mask>=(1/3)), axis=1)
            loc_other = imputation_res[imputation_res[time_column].isin(datasets[time_column].values)]
            loc_other.reset_index(drop=True, inplace=True)
            temp_imputation = temp_imputation.where(loc_mask, other=loc_other)

            loc_mask = np.concatenate((np.full((datasets.values.shape[0], 1),True), ((mask<(1/3)) | (mask>=(2/3)))), axis=1)
            imputation_res = BaseImputation().base_imputation(dataframe=clipped_dataset.where(loc_mask, other=np.nan), method_string='neural inter mask')
            loc_other = imputation_res[imputation_res[time_column].isin(datasets[time_column].values)]
            loc_other.reset_index(drop=True, inplace=True)
            temp_imputation = temp_imputation.where(loc_mask, other=loc_other)

            loc_mask = np.concatenate((np.full((datasets.values.shape[0], 1),True), mask<(2/3)), axis=1)
            imputation_res = BaseImputation().base_imputation(dataframe=clipped_dataset.where(loc_mask, other=np.nan), method_string='neural inter mask')
            loc_other = imputation_res[imputation_res[time_column].isin(datasets[time_column].values)]
            loc_other.reset_index(drop=True, inplace=True)
            temp_imputation = temp_imputation.where(loc_mask, other=loc_other)

            imputation_accumulated = imputation_accumulated + temp_imputation
            current_status += 0.5
            self.signal_percentage(int(current_status) % 100)

        # Calculate the Neural Inter Imputation
        imputed_dataset = imputation_accumulated / 50
        result = {}
        for c in used_columns:
            novelty_data = {}
            data = clipped_dataset[[c]]
            if np.count_nonzero(~np.isnan(data.values)) > 10:
                imputed_data = imputed_dataset[[c]].iloc[data.index]
                data_diff = (data - imputed_data).abs()
                sorted_data_diff = np.sort(data_diff.values, axis=None)
                sorted_data_diff = sorted_data_diff[~np.isnan(sorted_data_diff)]
                knee_result = KneeLocator(range(len(sorted_data_diff)), sorted_data_diff, S=self.s_value, curve='convex', direction='increasing')
                threshold_value = knee_result.knee_y
                if not threshold_value:
                    threshold_value = np.inf
                if threshold_value > thresholds[c]:
                    logging.debug(c + " requires another iteration (%.2f > %.2f)" % (threshold_value, thresholds[c]))
                current_diffs[c] = threshold_value
                for index, row in data_diff.iterrows():
                    if row[c] > threshold_value:
                        novelty_data[datasets[time_column].values[index]] = 1
                    else:
                        novelty_data[datasets[time_column].values[index]] = 0
            else:
                current_diffs[c] = 0
            data_c = datasets[[time_column, c]]
            phys_info = self.get_physiological_information(c)

            if phys_info:
                self.signal_add_infinite_line(c, "physical outlier lower limit", phys_info.low)
                self.signal_add_infinite_line(c, "physical outlier upper limit", phys_info.high)

                for index, row in data_c.iterrows():
                    if row[c] > phys_info.high or row[c] < phys_info.low:
                        novelty_data[row[time_column]] = 1

            result[c] = novelty_data
        current_status = 33.3
        self.signal_percentage(int(current_status) % 100)

        # Further Iterations
        while any(current_diffs[k] >= thresholds[k] for k in current_diffs):
            self.s_value += 0.05
            self.s_value = min(1, self.s_value)
            step_length = (100 - current_status) / 2
            if step_length < 0.1:
                break
            for c in used_columns:
                clipped_dataset[c][clipped_dataset[time_column].isin([k for k, v in result[c].items() if v == 1])] = np.nan

            imputation_accumulated = pd.DataFrame(0, columns=datasets.columns, index=datasets.index)
            for i in range(50):
                temp_imputation = pd.DataFrame(np.nan, columns=datasets.columns, index=datasets.index)
                mask = np.random.random((datasets.values.shape[0], datasets.values.shape[1] - 1))

                imputation_res = BaseImputation().base_imputation(dataframe=clipped_dataset.where(
                    np.concatenate((np.full((datasets.values.shape[0], 1), True), mask >= (1 / 3)), axis=1),
                    other=np.nan), method_string='neural inter mask')
                loc_mask = np.concatenate((np.full((datasets.values.shape[0], 1), False), mask >= (1 / 3)), axis=1)
                loc_other = imputation_res[imputation_res[time_column].isin(datasets[time_column].values)]
                loc_other.reset_index(drop=True, inplace=True)
                temp_imputation = temp_imputation.where(loc_mask, other=loc_other)

                loc_mask = np.concatenate(
                    (np.full((datasets.values.shape[0], 1), True), ((mask < (1 / 3)) | (mask >= (2 / 3)))), axis=1)
                imputation_res = BaseImputation().base_imputation(
                    dataframe=clipped_dataset.where(loc_mask, other=np.nan), method_string='neural inter mask')
                loc_other = imputation_res[imputation_res[time_column].isin(datasets[time_column].values)]
                loc_other.reset_index(drop=True, inplace=True)
                temp_imputation = temp_imputation.where(loc_mask, other=loc_other)

                loc_mask = np.concatenate((np.full((datasets.values.shape[0], 1), True), mask < (2 / 3)), axis=1)
                imputation_res = BaseImputation().base_imputation(
                    dataframe=clipped_dataset.where(loc_mask, other=np.nan), method_string='neural inter mask')
                loc_other = imputation_res[imputation_res[time_column].isin(datasets[time_column].values)]
                loc_other.reset_index(drop=True, inplace=True)
                temp_imputation = temp_imputation.where(loc_mask, other=loc_other)

                imputation_accumulated = imputation_accumulated + temp_imputation
                current_status += step_length/60
                self.signal_percentage(int(current_status) % 100)
            imputed_dataset = imputation_accumulated / 50

            for c in used_columns:
                novelty_data = result[c]
                data = clipped_dataset[[c]]
                if np.count_nonzero(~np.isnan(data.values)) > 2 and (current_diffs[c] >= thresholds[c]):
                    imputed_data = imputed_dataset[[c]].iloc[data.index]
                    data_diff = (data - imputed_data).abs()
                    sorted_data_diff = np.sort(data_diff.values, axis=None)
                    sorted_data_diff = sorted_data_diff[~np.isnan(sorted_data_diff)]
                    knee_result = KneeLocator(range(len(sorted_data_diff)), sorted_data_diff, S=self.s_value, curve='convex', direction='increasing')
                    threshold_value = knee_result.knee_y
                    if threshold_value > thresholds[c]:
                        logging.debug(c + " requires another iteration (%.2f > %.2f)" % (threshold_value, thresholds[c]))
                    current_diffs[c] = threshold_value
                    for index, row in data_diff.iterrows():
                        if row[c] > threshold_value:
                            novelty_data[datasets[time_column].values[index]] = 1
                else:
                    current_diffs[c] = 0
                result[c] = novelty_data
            current_status += step_length/6
            self.signal_percentage(int(current_status) % 100)

        self.signal_percentage(int(100) % 100)
        return result
