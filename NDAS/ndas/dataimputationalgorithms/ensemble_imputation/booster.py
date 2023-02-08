from .utils import feature_generation, data_loader, inverse_MinMaxScaler
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from os.path import exists
import pandas as pd
import os

class Booster:

    def __init__(self):
        self.bsts = list()

    def xgb_predict_feature(self, data, feature):
        bst = self.bsts[feature]
        features = feature_generation(data, feature)
        x = xgb.DMatrix(features)
        if bst:
            y_pred = bst.predict(x)
        else:
            y_pred = np.nan_to_num(data[:,feature])
        return y_pred

    def light_predict_feature(self, data, feature):
        bst = self.bsts[feature]
        features = feature_generation(data, feature)
        if bst:
            y_pred = bst.predict(features)
        else:
            # y_pred = np.nan_to_num(data[:,feature])
            y_pred = np.full(data[:,feature].shape, np.nan)
        return y_pred

    def predict(self, data, booster):
        print(f"Predicting with {booster}")
        num_features = data.shape[1]
        prediction = list()
        for feature in range(num_features):
            if booster == "LIGHT":
                y_pred = self.light_predict_feature(data, feature)
            elif booster == "XGB":
                y_pred = self.xgb_predict_feature(data, feature)
            prediction.append(y_pred)
        prediction = np.transpose(prediction)
        # Fill 0 to the missing values
        prediction = np.nan_to_num(prediction)
        return prediction

    def light_predict(self, data):
        return self.predict(data, "LIGHT")

    def xgb_predict(self, data):
        return self.predict(data, "XGB")

    def load_light_model(self, dim):
        self.bsts = list()
        for feature in range(dim):
            dir = os.path.dirname(__file__)
            path = f'{dir}/weights/light_feature_{feature}.txt'
            if exists(path):
                bst = lgb.Booster(model_file=path)
                self.bsts.append(bst)
            else:
                self.bsts.append(None)

    def load_xgb_model(self, dim):
        self.bsts = list()
        for feature in range(dim):
            dir = os.path.dirname(__file__)
            path = f'{dir}/weights/xgb_feature_{feature}.txt'
            if exists(path):
                bst = xgb.Booster(model_file=path)
                self.bsts.append(bst)
            else:
                self.bsts.append(None)

    def lightGbm_imputation(self, dataframe, **kwargs):
        return self.imputation(dataframe, "light")

    def xgb_imputation(self, dataframe, **kwargs):
        return self.imputation(dataframe, "xgb")

    def imputation(self, dataframe, model):
        # Interpolate the first column of the dataframe
        dataframe.iloc[:, 0] = dataframe.iloc[:, 0].interpolate()
        data, mask, mins, maxs= data_loader(dataframe)

        # interpolate the first column
        dim = data.shape[-1]
        if model == "light":
            self.load_light_model(dim)
            imputed_x = self.light_predict(data)
        else:
            self.load_xgb_model(dim)
            imputed_x = self.xgb_predict(data)
        
        data = inverse_MinMaxScaler(imputed_x, mins, maxs)
        
        # Convert to dataframe
        imputed_dataframe = pd.DataFrame(data, columns=dataframe.columns)

        # Combine the imputed dataframe with the original dataframe using the mask
        combined_dataframe = dataframe.where(mask, imputed_dataframe)

        return combined_dataframe




