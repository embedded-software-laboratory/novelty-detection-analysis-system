import os.path


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge, RANSACRegressor, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from . import keras_imputation as ki
import sys
sys.path.append(os.path.realpath(".\\ndas"))
import hickle as hkl
from os import listdir


class SklearnImputation:
    """
    Implements imputation methods from scikit-learn using IterativeImputer
    """

    def mice_imputation(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=True):
        lr = LinearRegression(positive=True)
        imputer = IterativeImputer(estimator=lr, missing_values=np.nan, max_iter=15, initial_strategy='median', skip_complete=True, min_value=0, max_value=1, tol=1e-5, imputation_order='roman')
        if already_fit:
            imputer = hkl.load('C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\lr.hkl')
        return self.iterative_imputation(dataframe, imputer, already_fit=already_fit)

    def bayesian_imputation(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=True):
        bay = BayesianRidge(tol=1e-5)
        imputer = IterativeImputer(estimator=bay, missing_values=np.nan, max_iter=15, initial_strategy='median', skip_complete=True, min_value=0, max_value=1, tol=1e-5, imputation_order='roman')
        if already_fit:
            imputer = hkl.load('C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\bay.hkl')
        return self.iterative_imputation(dataframe, imputer, already_fit=already_fit)

    def mlp_imputation(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=True):
        mlp = MLPRegressor(activation='tanh', learning_rate='adaptive', learning_rate_init=0.002, tol=1e-5, warm_start=True, early_stopping=True)
        imputer = IterativeImputer(estimator=mlp, missing_values=np.nan, max_iter=15, initial_strategy='median', skip_complete=True, min_value=0, max_value=1, tol=1e-5, imputation_order='roman')
        if already_fit:
            imputer = hkl.load('C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\mlp.hkl')
        return self.iterative_imputation(dataframe, imputer, already_fit=already_fit)

    def svr_imputation(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=True):
        svr = LinearSVR(tol=1e-5, loss='squared_epsilon_insensitive', dual=False)
        imputer = IterativeImputer(estimator=svr, missing_values=np.nan, max_iter=15, initial_strategy='median', skip_complete=True, min_value=0, max_value=1, tol=1e-5, imputation_order='roman')
        if already_fit:
            imputer = hkl.load('C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\svr.hkl')
        return self.iterative_imputation(dataframe, imputer, already_fit=already_fit)

    def tree_imputation(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=True):
        dtr = DecisionTreeRegressor(criterion='friedman_mse', min_samples_split=3, min_samples_leaf=2, ccp_alpha=0.01)
        imputer = IterativeImputer(estimator=dtr, missing_values=np.nan, max_iter=15, initial_strategy='median', skip_complete=True, min_value=0, max_value=1, tol=1e-5, imputation_order='roman')
        if already_fit:
            imputer = hkl.load('C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\dtr.hkl')
        return self.iterative_imputation(dataframe, imputer, already_fit=already_fit)

    def extra_tree_imputation(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=True):
        etr = ExtraTreeRegressor(criterion='friedman_mse', min_samples_split=3, min_samples_leaf=2, ccp_alpha=0.01)
        imputer = IterativeImputer(estimator=etr, missing_values=np.nan, max_iter=15, initial_strategy='median', skip_complete=True, min_value=0, max_value=1, tol=1e-5, imputation_order='roman')
        if already_fit:
            imputer = hkl.load('C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\etr.hkl')
        return self.iterative_imputation(dataframe, imputer, already_fit=already_fit)

    def ransac_imputation(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=True):
        ransac = RANSACRegressor(loss='squared_error')
        imputer = IterativeImputer(estimator=ransac, missing_values=np.nan, max_iter=15, initial_strategy='median', skip_complete=True, min_value=0, max_value=1, tol=1e-5, imputation_order='roman')
        if already_fit:
            imputer = hkl.load('C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\ransac.hkl')
        return self.iterative_imputation(dataframe, imputer, already_fit=already_fit)

    def sgd_imputation(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=True):
        sgd = SGDRegressor(tol=1e-5, learning_rate='adaptive', early_stopping=True, warm_start=True)
        imputer = IterativeImputer(estimator=sgd, missing_values=np.nan, max_iter=15, initial_strategy='median', skip_complete=True, min_value=0, max_value=1, tol=1e-5, imputation_order='roman')
        if already_fit:
            imputer = hkl.load('C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\sgd.hkl')
        return self.iterative_imputation(dataframe, imputer, already_fit=already_fit)

    def ada_imputation(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=True):
        ada = AdaBoostRegressor(loss='square')
        imputer = IterativeImputer(estimator=ada, missing_values=np.nan, max_iter=15, initial_strategy='median', skip_complete=True, min_value=0, max_value=1, tol=1e-5, imputation_order='roman')
        if already_fit:
            imputer = hkl.load('C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\ada.hkl')
        return self.iterative_imputation(dataframe, imputer, already_fit=already_fit)

    def mean_imputation(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=True):
        imputer = SimpleImputer(strategy='mean')
        if already_fit:
            imputer = hkl.load('C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\mean.hkl')
        return self.iterative_imputation(dataframe, imputer, already_fit=already_fit)

    def median_imputation(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=True):
        imputer = SimpleImputer(strategy='median')
        if already_fit:
            imputer = hkl.load('C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\median.hkl')
        return self.iterative_imputation(dataframe, imputer, already_fit=already_fit)

    def mean_nf_imputation(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=False):
        imputer = SimpleImputer(strategy='mean')
        return self.iterative_imputation(dataframe, imputer, already_fit=False)

    def median_nf_imputation(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=False):
        imputer = SimpleImputer(strategy='median')
        return self.iterative_imputation(dataframe, imputer, already_fit=False)

    def uni_random_imputation(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=False):
        return self.random_imputation(dataframe, fit_to_dataframe=False, use_uniform=True)

    def norm_random_imputation(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=False):
        return self.random_imputation(dataframe, fit_to_dataframe=False, use_uniform=False)

    def fit_norm_random_imputation(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=False):
        return self.random_imputation(dataframe, fit_to_dataframe=True, use_uniform=False)

    def iterative_imputation(self, dataframe, imp, already_fit=False):
        time_series = dataframe.iloc[:, 0:1]
        other_series = dataframe.iloc[:, 1:10]
        non_series_data = dataframe.iloc[:, 10:]
        interpolated_time_series = time_series.interpolate(limit_area=None, limit_direction='both')
        interpolated_non_series_data = non_series_data.interpolate(method='nearest', limit_area=None, limit_direction='both')
        pre_computed = pd.concat([interpolated_time_series, other_series, interpolated_non_series_data], axis=1)
        encoded = self.encode_to_zero_one_range(np.concatenate([pre_computed.iloc[:, 1:10], pre_computed.iloc[:, 11:]], axis=1))
        if already_fit:
            imputated_other_series = imp.transform(encoded)
        else:
            imputated_other_series = imp.fit_transform(pre_computed.values[:, 1:])
        list_imputed_other_series_columns = []
        j = 0
        for i in range(0, 9):
            if already_fit:
                list_imputed_other_series_columns.append(imputated_other_series[:, j])
                j += 1
            elif other_series.iloc[:, i].count() > 0:
                list_imputed_other_series_columns.append(imputated_other_series[:, j])
                j += 1
            else:
                list_imputed_other_series_columns.append(other_series.values[:, i])
        if already_fit:
            pd_imputated_other_series = pd.DataFrame(data=self.decode_from_zero_one_range(np.asarray(list_imputed_other_series_columns).T), index=other_series.index, columns=other_series.columns)
        else:
            pd_imputated_other_series = pd.DataFrame(data=np.asarray(list_imputed_other_series_columns).T, index=other_series.index, columns=other_series.columns)
        result = pd.concat([interpolated_time_series, pd_imputated_other_series, interpolated_non_series_data], axis=1)
        return result

    def random_imputation(self, dataframe, fit_to_dataframe=False, use_uniform=False):
        time_series = dataframe.iloc[:, 0:1]
        other_series = dataframe.iloc[:, 1:10]
        non_series_data = dataframe.iloc[:, 10:]
        interpolated_time_series = time_series.interpolate(limit_area=None, limit_direction='both')
        interpolated_non_series_data = non_series_data.interpolate(method='nearest', limit_area=None, limit_direction='both')
        pre_computed = pd.concat([interpolated_time_series, other_series, interpolated_non_series_data], axis=1)
        encoded_np = self.encode_to_zero_one_range( np.concatenate([pre_computed.iloc[:, 1:10], pre_computed.iloc[:, 11:]], axis=1))[:, 0:9]
        for i in range(0, 9):
            column = encoded_np[:, i]
            mask_nan = np.isnan(column)
            mu, sigma = 0.5, 0.125
            if fit_to_dataframe and np.sum(~np.isnan(column)) > 0:
                mu, sigma = np.nanmean(column), np.nanstd(column)
            if use_uniform:
                encoded_np[mask_nan, i] = np.clip(np.random.random(size=mask_nan.sum()), 0, 1)
            else:
                encoded_np[mask_nan, i] = np.clip(np.random.normal(mu, sigma, size=mask_nan.sum()), 0, 1)
        pd_imputated_other_series = pd.DataFrame(data=self.decode_from_zero_one_range(encoded_np), index=dataframe.index, columns=dataframe.columns[1:10])
        result = pd.concat([interpolated_time_series, pd_imputated_other_series, interpolated_non_series_data], axis=1)
        return result

    def knn_imputation(self, dataframe, lim_dir='forward', lim_are='inside'):
        time_series = dataframe.iloc[:, 0:1]
        other_series = dataframe.iloc[:, 1:10]
        non_series_data = dataframe.iloc[:, 10:]
        interpolated_time_series = time_series.interpolate()
        interpolated_non_series_data = non_series_data.interpolate(method='nearest')
        imp = KNNImputer(missing_values=np.nan, weights='distance')
        imputated_other_series = imp.fit_transform(
            pd.concat([interpolated_time_series, other_series, non_series_data], axis=1))
        list_imputed_other_series_columns = []
        j = 0
        for i in range(0, 9):
            if other_series.iloc[:, i].count() > 0:
                list_imputed_other_series_columns.append(imputated_other_series[:, j + 1])
                j += 1
            else:
                list_imputed_other_series_columns.append(other_series.values[:, i])
        pd_imputated_other_series = pd.DataFrame(data=np.asarray(list_imputed_other_series_columns).T,
                                                 index=other_series.index, columns=other_series.columns)
        result = pd.concat([interpolated_time_series, pd_imputated_other_series, interpolated_non_series_data], axis=1)
        return result

    def encode_to_zero_one_range(self, dataframe):
        '''
        (exclude time_offset and ID)
        Input: Matrix num_samples x 56
        Output: Matrix num_samples x 61
        one_hot encoding for gender and ethnicity
        0-1 Range normalization for all values
        '''
        default_value_min = [28, 20, 5, 35, 20, 25, 60, 18, -10]
        default_value_max = [44, 220, 50, 280, 180, 210, 100, 49, 50]
        value_lines = dataframe[:, 0:9]
        encoded_value_lines = (value_lines - np.array(default_value_min)) / (np.array(default_value_max)-np.array(default_value_min))
        gender = dataframe[:, 9]
        age = dataframe[:, 10]
        ethnicity = dataframe[:, 11]
        body_mass = dataframe[:, 12:14]
        icd_codes = dataframe[:, 14:]
        return np.concatenate([encoded_value_lines, self.get_one_hot(gender, 3)[:, 1:], np.atleast_2d(age/90).T, self.get_one_hot(ethnicity, 6)[:, 1:], body_mass/250, icd_codes], axis=1)

    def decode_from_zero_one_range(self, dataframe):
        '''
        (excludes time_offset and ID)
        Input: Matrix num_samples x 9
        Output: Matrix num_samples x 9
        0-1 Range denormalization for all values
        '''
        default_value_min = [28, 20, 5, 35, 20, 25, 60, 18, -10]
        default_value_max = [44, 220, 50, 280, 180, 210, 100, 49, 50]
        value_lines = dataframe[:, 0:9]
        decoded_value_lines = (value_lines * (np.array(default_value_max)-np.array(default_value_min))) + np.array(default_value_min)
        return decoded_value_lines

    def get_one_hot(self, targets, nb_classes):
        res = np.eye(nb_classes)[np.array(targets.astype(int)).reshape(-1)]
        return res.reshape(list(targets.shape) + [nb_classes])
