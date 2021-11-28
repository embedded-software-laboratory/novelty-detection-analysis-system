import numpy as np
import pandas as pd
from ndas.algorithms.basedetector import BaseDetector  # Import the basedetector class
from ndas.misc.parameter import ArgumentType
from ndas.dataimputationalgorithms.base_imputation import BaseImputation
from kneed import KneeLocator


class NeuralInterSamplingDetector(BaseDetector):
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
        super(NeuralInterSamplingDetector, self).__init__(*args, **kwargs)

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
        imputation_accumulated = pd.DataFrame(0, columns=datasets.columns, index=datasets.index)
        current_status = 0.0
        for i in range(50):
            self.signal_percentage(int(current_status) % 100)
            temp_imputation = pd.DataFrame(np.nan, columns=datasets.columns, index=datasets.index)
            mask = np.random.random(datasets.values.shape)
            temp_imputation = temp_imputation.where(mask>=(1/3), other= BaseImputation().base_imputation(dataframe=datasets.where(mask>=(1/3), other=np.nan), method_string='neural inter mask'))
            temp_imputation = temp_imputation.where((mask<(1/3)) | (mask>=(2/3)), other= BaseImputation().base_imputation(dataframe=datasets.where((mask<(1/3)) | (mask>=(2/3)), other=np.nan), method_string='neural inter mask'))
            temp_imputation = temp_imputation.where(mask<(2/3), other= BaseImputation().base_imputation(dataframe=datasets.where(mask<(2/3), other=np.nan), method_string='neural inter mask'))
            imputation_accumulated = imputation_accumulated + temp_imputation
            current_status += 1.5

        # Calculate the Neural Inter Imputation
        imputed_dataset = imputation_accumulated / 50
        # Get the additional arguments
        time_column = datasets.columns[0]

        status_length = 25 / len(datasets.columns[1:10])
        current_status = 75.0
        result = {}
        for c in datasets.columns[1:10]:
            self.signal_percentage(int(current_status) % 100)
            novelty_data = {}
            data = datasets[[c]]
            if np.count_nonzero(~np.isnan(data.values)) > 0:
                imputed_data = imputed_dataset[[c]].iloc[data.index]
                data_diff = (data - imputed_data).abs()
                sorted_data_diff = np.sort(data_diff.values, axis=None)
                sorted_data_diff = sorted_data_diff[~np.isnan(sorted_data_diff)]
                threshold_value = KneeLocator(range(len(sorted_data_diff)), sorted_data_diff, S=1, curve='convex', direction='increasing').knee_y

                for index, row in data_diff.iterrows():
                    if row[c] > threshold_value:
                        novelty_data[datasets[time_column].values[index]] = 1
                    else:
                        novelty_data[datasets[time_column].values[index]] = 0

            result[c] = novelty_data
            current_status += status_length

        return result
