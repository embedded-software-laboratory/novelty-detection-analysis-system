import numpy as np
from . import pandas_interpolation
from . import sklearn_imputation
from . import keras_imputation
import pandas as pd
from scipy.stats import mode
import logging


class BaseImputation:
    """
    Defines general imputation behaviour and calls selected Method
    """

    def __init__(self):
        self.Pandas_Interpolation = pandas_interpolation.PandasInterpolation()
        self.Sklearn_Imputation = sklearn_imputation.SklearnImputation()
        self.Keras_Imputation = keras_imputation.KerasImputation()
        # self.Methods = {'uniform random': self.Sklearn_Imputation.uni_random_imputation, 'normal random': self.Sklearn_Imputation.norm_random_imputation, 'adjusted normal random': self.Sklearn_Imputation.fit_norm_random_imputation, 'mean': self.Sklearn_Imputation.mean_imputation, 'median': self.Sklearn_Imputation.median_imputation, 'mean non-fit': self.Sklearn_Imputation.mean_nf_imputation, 'median non-fit': self.Sklearn_Imputation.median_nf_imputation, 'interpolate': self.Pandas_Interpolation.simple_interpolation,  'polynomial': self.Pandas_Interpolation.polynomial_interpolation,  'spline': self.Pandas_Interpolation.spline_interpolation, 'neural inter': self.Pandas_Interpolation.neural_interpolation, 'neural inter mask': self.Pandas_Interpolation.neural_interpolation_mask, 'neural inter mask round': self.Pandas_Interpolation.neural_interpolation_mask_round, 'mice': self.Sklearn_Imputation.mice_imputation, 'knn': self.Sklearn_Imputation.knn_imputation, 'bayesian':  self.Sklearn_Imputation.bayesian_imputation, 'mlp': self.Sklearn_Imputation.mlp_imputation, 'svr': self.Sklearn_Imputation.svr_imputation, 'tree': self.Sklearn_Imputation.tree_imputation, 'extra_tree': self.Sklearn_Imputation.extra_tree_imputation, 'ransac': self.Sklearn_Imputation.ransac_imputation, 'sgd': self.Sklearn_Imputation.sgd_imputation, 'ada_boost': self.Sklearn_Imputation.ada_imputation, 'mice_ts': self.Keras_Imputation.mice_imputation, 'bayesian_ts':  self.Keras_Imputation.bayesian_imputation, 'mlp_ts': self.Keras_Imputation.mlp_imputation, 'svr_ts': self.Keras_Imputation.svr_imputation, 'tree_ts': self.Keras_Imputation.tree_imputation, 'extra_tree_ts': self.Keras_Imputation.extra_tree_imputation, 'ransac_ts': self.Keras_Imputation.ransac_imputation, 'sgd_ts': self.Keras_Imputation.sgd_imputation, 'ada_boost_ts': self.Keras_Imputation.ada_imputation, 'keras lstm': self.Keras_Imputation.kerlstm_imputation, 'keras nn': self.Keras_Imputation.kernn_imputation, 'keras nn sliding window': self.Keras_Imputation.kernn_sw_imputation, 'keras lstm sliding window': self.Keras_Imputation.kerlstm_sw_imputation, 'keras gan 8 epochs sliding window': self.Keras_Imputation.kergan_sw_imputation8, 'keras gan 10 epochs sliding window': self.Keras_Imputation.kergan_sw_imputation5, 'keras gan 5 epochs sliding window': self.Keras_Imputation.kergan_sw_imputation10, 'keras gan 15 epochs sliding window': self.Keras_Imputation.kergan_sw_imputation15, 'keras gan 20 epochs sliding window': self.Keras_Imputation.kergan_sw_imputation20}
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
