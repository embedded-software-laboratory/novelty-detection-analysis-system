import numpy as np
import pandas as pd
from ndas.algorithms.basedetector import BaseDetector  # Import the basedetector class
from ndas.misc.parameter import ArgumentType
from ndas.dataimputationalgorithms.base_imputation import BaseImputation
from ndas.extensions import plots
from kneed import KneeLocator


class NeuralInterSamplingDetectorWithPhysicalLimits(BaseDetector):
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
        super(NeuralInterSamplingDetectorWithPhysicalLimits, self).__init__(*args, **kwargs)
        self.register_parameter("S", ArgumentType.FLOAT, 0.9, 0, 10, tooltip="Parameter for the aggressiveness of error detection.")

    def detect(self, datasets, **kwargs) -> dict:
        """ Return novelties in given dataset

        This method needs to be overwritten from the Basedetector class.
        It has to contain the required logic for novelty detection.

        Parameters
        ----------
        datasets
            The dataset
        kwargs
            Additional parameters

        Returns
        -------
        novelties : dict
            A dict of plots with novelties with observation offset as key
        """
        self.s_value = float(kwargs["S"])

        clipped_dataset = datasets.copy(deep=True)
        for c in datasets.columns[1:10]:
            phys_info = self.get_physiological_information(c)
            if phys_info:
                clipped_dataset[c].where(clipped_dataset[c].between(phys_info.low, phys_info.high), other=np.nan, inplace=True)
        imputation_accumulated = pd.DataFrame(0, columns=datasets.columns, index=datasets.index)
        current_status = 5.0
        for i in range(50):
            self.signal_percentage(int(current_status) % 100)
            temp_imputation = pd.DataFrame(np.nan, columns=datasets.columns, index=datasets.index)
            mask = np.random.random(datasets.values.shape)
            temp_imputation = temp_imputation.where(mask>=(1/3), other= BaseImputation().base_imputation(dataframe=clipped_dataset.where(mask>=(1/3), other=np.nan), method_string='neural inter mask'))
            temp_imputation = temp_imputation.where((mask<(1/3)) | (mask>=(2/3)), other= BaseImputation().base_imputation(dataframe=clipped_dataset.where((mask<(1/3)) | (mask>=(2/3)), other=np.nan), method_string='neural inter mask'))
            temp_imputation = temp_imputation.where(mask<(2/3), other= BaseImputation().base_imputation(dataframe=clipped_dataset.where(mask<(2/3), other=np.nan), method_string='neural inter mask'))
            imputation_accumulated = imputation_accumulated + temp_imputation
            current_status += 1.4

        # Calculate the Neural Inter Imputation
        imputed_dataset = imputation_accumulated / 50
        # Get the additional arguments
        time_column = datasets.columns[0]
        used_columns = [col for col in datasets.columns if col in plots.get_available_plot_keys(datasets)]

        status_length = 25 / len(used_columns)
        current_status = 75.0
        result = {}
        for c in used_columns:
            self.signal_percentage(int(current_status) % 100)
            novelty_data = {}
            data = clipped_dataset[[c]]
            if np.count_nonzero(~np.isnan(data.values)) > 2:
                imputed_data = imputed_dataset[[c]].iloc[data.index]
                data_diff = (data - imputed_data).abs()
                sorted_data_diff = np.sort(data_diff.values, axis=None)
                sorted_data_diff = sorted_data_diff[~np.isnan(sorted_data_diff)]
                knee_result = KneeLocator(range(len(sorted_data_diff)), sorted_data_diff, S=self.s_value, curve='convex', direction='increasing')
                threshold_value = knee_result.knee_y
                for index, row in data_diff.iterrows():
                    if row[c] > threshold_value:
                        novelty_data[datasets[time_column].values[index]] = 1
                    else:
                        novelty_data[datasets[time_column].values[index]] = 0
            data_c = datasets[[time_column, c]]
            phys_info = self.get_physiological_information(c)

            if phys_info:
                self.signal_add_infinite_line(c, "physical outlier lower limit", phys_info.low)
                self.signal_add_infinite_line(c, "physical outlier upper limit", phys_info.high)

                for index, row in data_c.iterrows():
                    if row[c] > phys_info.high or row[c] < phys_info.low:
                        novelty_data[row[time_column]] = 1

            result[c] = novelty_data
            current_status += status_length
        return result
