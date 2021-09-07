import numpy as np
from ndas.dataimputationalgorithms import pandas_interpolation
import pandas as pd


class BaseImputation:
    """
    Defines general imputation behaviour and calls selected Method
    """

    def __init__(self):
        self.Pandas_Interpolation = pandas_interpolation.PandasInterpolation()
        self.Methods = {'interpolate': self.Pandas_Interpolation.simple_interpolation,  'polynomial': self.Pandas_Interpolation.polynomial_interpolation,  'spline': self.Pandas_Interpolation.spline_interpolation}

    def base_imputation(self, dataframe, spacing_multiplier, current_spacing, method_string):
        """
        Method for Imputation. First find and fill Missing gaps, then increase density depending on spacing_multiplier
        """
        imputation_dataframe = dataframe
        differences = dataframe['tOffset'].diff()

        for idx, val in reversed(list(enumerate(differences))):
            if val > current_spacing:
                num_inserted_rows = int(val // current_spacing)
                empty_rows = pd.DataFrame((np.nan), index=(range(num_inserted_rows)), columns=(dataframe.columns))
                imputation_dataframe = pd.concat([imputation_dataframe.iloc[:idx], empty_rows, imputation_dataframe.iloc[idx:]])

        nans = np.where(np.empty_like(imputation_dataframe.values), np.nan, np.nan)
        data = np.hstack([imputation_dataframe.values] + [nans] * (spacing_multiplier - 1)).reshape(-1, imputation_dataframe.shape[1])

        if spacing_multiplier > 1:
            imputation_dataframe = pd.DataFrame((data[:-(spacing_multiplier - 1)]), columns=(dataframe.columns))
        else:
            imputation_dataframe = pd.DataFrame(data, columns=(dataframe.columns))

        return self.Methods[method_string](imputation_dataframe)
