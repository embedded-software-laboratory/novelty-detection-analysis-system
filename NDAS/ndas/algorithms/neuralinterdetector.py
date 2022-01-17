import numpy as np
from ndas.algorithms.basedetector import BaseDetector  # Import the basedetector class
from ndas.dataimputationalgorithms.base_imputation import BaseImputation
from ndas.extensions import plots
from kneed import KneeLocator


class NeuralInterDetector(BaseDetector):
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
        super(NeuralInterDetector, self).__init__(*args, **kwargs)

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

        # Calculate the Neural Inter Imputation
        imputed_dataset = BaseImputation().base_imputation(dataframe=datasets, method_string='neural inter')

        time_column = datasets.columns[0]
        used_columns = [col for col in datasets.columns if col in plots.get_registered_plot_keys()]

        status_length = 90 / len(used_columns)
        current_status = 10.0
        self.signal_percentage(int(current_status) % 100)
        result = {}
        for c in used_columns:
            novelty_data = {}
            data = datasets[[c]]
            if np.count_nonzero(~np.isnan(data.values)) > 0:
                imputed_data = imputed_dataset[[c]].iloc[data.index]
                data_diff = (data - imputed_data).abs()
                sorted_data_diff = np.sort(data_diff.values, axis=None)
                sorted_data_diff = sorted_data_diff[~np.isnan(sorted_data_diff)]
                if (len(sorted_data_diff)) > 1:
                    threshold_value = KneeLocator(range(len(sorted_data_diff)), sorted_data_diff, S=0.9, curve='convex',
                                              direction='increasing').knee_y
                else:
                    threshold_value = None
                if threshold_value != None:
                    for index, row in data_diff.iterrows():
                        if row[c] > threshold_value:
                            novelty_data[datasets[time_column].values[index]] = 1
                        else:
                            novelty_data[datasets[time_column].values[index]] = 0

            result[c] = novelty_data
            current_status += status_length
            self.signal_percentage(int(current_status) % 100)

        return result
