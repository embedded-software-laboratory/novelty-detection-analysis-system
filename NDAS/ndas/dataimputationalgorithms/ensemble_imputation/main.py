from .utils import data_loader, inverse_MinMaxScaler
from .booster import Booster
import numpy as np
import pandas as pd

def lightGbm_imputation(dataframe):
  return imputation(dataframe, "light")

def xgb_imputation(dataframe):
  return imputation(dataframe, "xgb")

def imputation(dataframe, model):
  
  data, mins, maxs= data_loader(dataframe)
  dim = data.shape[-1]
  booster = Booster( dim)
  if model == "light":
    booster.load_light_model()
    imputed_x = booster.light_predict(data)
  else:
    booster.load_xgb_model()
    imputed_x = booster.xgb_predict(data)

  return imputed_x, mins, maxs

def main():
  # generate pandas dataframe with random missing values of shape (1000, 15)
  # and missing rate of 0.2
  df = pd.DataFrame(np.random.rand(1000, 15))
  df.iloc[np.random.randint(0, 1000, 200), np.random.randint(0, 15, 200)] = np.nan

  booster = Booster()
  data = booster.lightGbm_imputation(df)
  print(data.shape)

