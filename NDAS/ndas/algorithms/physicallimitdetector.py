from ndas.algorithms.basedetector import BaseDetector  # Import the basedetector class
from ndas.extensions import plots


class PhysicalLimitDetector(BaseDetector):
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
        super(PhysicalLimitDetector, self).__init__(*args, **kwargs)

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

        # Get the additional arguments
        time_column = datasets.columns[0]
        used_columns = [col for col in datasets.columns if col in plots.get_registered_plot_keys()]
        status_length = 90 / len(used_columns)
        current_status = 10.0
        result = {}
        for c in used_columns:
            self.signal_percentage(int(current_status + status_length) % 100)

            data = datasets[[time_column, c]]
            novelty_data = {}
            phys_info = self.get_physiological_information(c)

            if phys_info:
                self.signal_add_infinite_line(c, "physical outlier lower limit", phys_info.low)
                self.signal_add_infinite_line(c, "physical outlier upper limit", phys_info.high)
                for index, row in data.iterrows():
                    if row[c] > phys_info.high or row[c] < phys_info.low:
                        novelty_data[row[time_column]] = 1
                    else:
                        novelty_data[row[time_column]] = 0
            else:
                for index, row in data.iterrows():
                    novelty_data[row[time_column]] = 0

            result[c] = novelty_data

        return result
