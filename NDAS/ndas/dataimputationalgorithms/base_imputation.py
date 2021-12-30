import numpy as np
from . import pandas_interpolation
import pandas as pd
from scipy.stats import mode
import logging


class BaseImputation:
    """
    Defines general imputation behaviour and calls selected Method
    """

    def __init__(self):
        self.Pandas_Interpolation = pandas_interpolation.PandasInterpolation()
        self.Methods = {'interpolate': self.Pandas_Interpolation.simple_interpolation,  'spline': self.Pandas_Interpolation.spline_interpolation, 'neural inter': self.Pandas_Interpolation.neural_interpolation, 'neural inter mask': self.Pandas_Interpolation.neural_interpolation_mask, 'neural inter mask round': self.Pandas_Interpolation.neural_interpolation_mask_round}

    def base_imputation(self, dataframe, spacing_multiplier=1, current_spacing=5, method_string='neural inter mask round', lim_dir='forward', lim_are='inside'):
        """
        Method for Imputation. First find and fill Missing gaps, then increase density depending on spacing_multiplier
        """
        time_column = dataframe.columns[0]
        differences = dataframe[time_column].diff()

        if any(x < 0 for x in differences.values):
            logging.warning("Time Column is not monotonically increasing, aborting imputation and returning input.")
            return dataframe

        current_spacing = mode(differences.values, axis=None, nan_policy='omit')[0][0]
        imputation_dataframe = dataframe

        for idx, val in reversed(list(enumerate(differences))):
            if val > current_spacing:
                num_inserted_rows = int(val // current_spacing)
                empty_rows = pd.DataFrame(np.nan, index=(range(num_inserted_rows)), columns=dataframe.columns)
                imputation_dataframe = pd.concat([imputation_dataframe.iloc[:idx], empty_rows, imputation_dataframe.iloc[idx:]])

        nans = np.where(np.empty_like(imputation_dataframe.values), np.nan, np.nan)
        data = np.hstack([imputation_dataframe.values] + [nans] * (spacing_multiplier - 1)).reshape(-1, imputation_dataframe.shape[1])

        if spacing_multiplier > 1:
            imputation_dataframe = pd.DataFrame((data[:-(spacing_multiplier - 1)]), columns=dataframe.columns)
        else:
            imputation_dataframe = pd.DataFrame(data, columns=dataframe.columns)

        return self.Methods[method_string](imputation_dataframe, lim_dir=lim_dir, lim_are=lim_are)
