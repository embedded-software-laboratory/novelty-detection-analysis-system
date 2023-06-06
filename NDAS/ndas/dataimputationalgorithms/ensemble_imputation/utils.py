import numpy as np

def MinMaxScaler(data):      
  """Normalization tool: Min Max Scaler.
  
  Args:
    - data: raw input data
    
  Returns:
    - normalized_data: minmax normalized data
    - norm_parameters: normalization parameters for rescaling if needed
  """  
  min_val = np.nanmin(data, axis = 0)
  max_val = np.nanmax(data, axis = 0) + 1e-8
  normalized_data = (data-min_val) / (max_val - min_val)
  
  norm_parameters = {'min_val': min_val, 'max_val': max_val}
  return normalized_data, norm_parameters

def inverse_MinMaxScaler(normalized_data, mins, maxs):
  """Inverse normalization tool: Min Max Scaler.
  
  Args:
    - normalized_data: normalized data
    - mins: minimum values for each feature
    - maxs: maximum values for each feature
    
  Returns:
    - data: original data
  """  
  data = normalized_data * (maxs - mins) + mins
  return data

def data_loader(dataframe):

    data = dataframe.to_numpy()
    data, rescaling_vals = MinMaxScaler(data)
    # mask = np.ones(data.shape)
    # mask[np.isnan(data)] = 0
    mask = np.full(data.shape, True)
    mask[np.isnan(data)] = False
    
    # create numpy array that has same shape as data and is filled with rescaling values
    mins = np.full(data.shape, rescaling_vals["min_val"])
    maxs = np.full(data.shape, rescaling_vals["max_val"])

    return data, mask, mins, maxs


def feature_generation(data, feature):
    x = list()
    rows = data.shape[0]
    num_features = data.shape[1]
    for row in range(3):
        horizontal = np.concatenate([data[row, :feature], data[row, feature + 1:]])
        temp_before = np.empty((num_features*(3-row)))
        temp_before[:] = np.nan
        before = np.concatenate([temp_before, data[:row,:].flatten()])
        after = data[row+1:row+4,:].flatten()
        feature_list = np.concatenate([horizontal, before, after])
        x.append(feature_list)
                                       
    for row in range(3, rows-3):
        feature_list = np.concatenate([data[row, :feature], data[row, feature + 1:], data[row-3:row, :].flatten(), data[row+1:row+4, :].flatten()])
        x.append(feature_list)

    for row in range(rows-3, rows):
        horizontal = np.concatenate([data[row, :feature], data[row, feature + 1:]])
        before = data[row-3:row,:].flatten()
        temp_after = np.empty((num_features*(4-(rows-row))))
        temp_after[:] = np.nan
        after = np.concatenate([data[row+1:,:].flatten(), temp_after])
        feature_list = np.concatenate([horizontal, before, after])
        x.append(feature_list)

    return np.asarray(x)

