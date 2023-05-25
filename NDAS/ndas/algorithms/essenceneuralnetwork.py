import numpy as np
import pandas as pd
from ndas.algorithms.basedetector import BaseDetector
from ndas.dataimputationalgorithms.base_imputation import BaseImputation
from sklearn.preprocessing import StandardScaler


class EssenceNeuralNetwork(BaseDetector):

    def __init__(self, *args, **kwargs):
        super(EssenceNeuralNetwork, self).__init__(*args, **kwargs)

    @staticmethod
    def compute_output(input, weights, biases, last_layer=False):
        """Returns the output of the layer given the previous layer as input"""
        if len(input.shape) == 1:  # fixes the shape of the input layer to be a 2d array
            input = np.reshape(input, (1, -1))
        nodes = input @ weights
        nodes = np.add(nodes, np.tile(biases, (nodes.shape[0], 1)))
        if last_layer:
            # The last layer uses a different activation
            nodes = nodes - np.max(nodes)
            nodes = np.exp(nodes)
            sum_total = np.sum(nodes, 1)
            nodes = np.divide(nodes, np.tile(sum_total, (nodes.shape[1], 1)).transpose())
        else:
            # The binning activation function
            if np.max(np.abs((input - np.round(input)))) < 1e-3:
                nodes = np.round(nodes)
            np.putmask(nodes, nodes > 1, 1)
            np.putmask(nodes, nodes < -1, -1)
            np.putmask(nodes, np.logical_and(nodes > -1, nodes < 1), 0)
            nodes += 1
            nodes /= 2

        return nodes

    def detect(self, datasets, **kwargs) -> dict:
        result = {}
        # Load the network weights and the standard scaler parameters
        """
            The input npz file contains the network parameters
            1. The weights and biases of the layers
            2. The scale, mean and var parameters of the StandardScaler
            3. The values used for the Additional Mean Imputation
        """
        file_path = "ndas\\algorithms\\ENN_weights\\"
        network_file = np.load(file_path+"basic_enn.npz", allow_pickle=True)
        weights = network_file['weights']
        biases = network_file['biases']
        scaler = StandardScaler()
        scaler.scale_ = network_file['scaler_scale']
        scaler.mean_ = network_file['scaler_mean']
        scaler.var_ = network_file['scaler_var']
        input_weights = network_file['input_means'][()]
        # Preprocess the input
        para_labels = ['TempC', 'HR', 'RR', 'pSy', 'pDi', 'pMe', 'SaO2', 'etCO2', 'CVP']
        mask_labels = ['TempC_mask', 'HR_mask', 'RR_mask', 'pSy_mask', 'pDi_mask', 'pMe_mask', 'SaO2_mask',
                       'etCO2_mask', 'CVP_mask']

        # Add any missing columns and do data imputation
        df = datasets.reindex(datasets.columns.union(para_labels, sort=False), axis=1)
        df.rename(columns=lambda x:  "_" + x if x != 'tOffset' else x, inplace=True)
        imputed_dataset = BaseImputation().base_imputation(dataframe=df, method_string='neural inter')
        imputed_dataset.rename(columns=lambda x: x[1:] if x != 'tOffset' else x, inplace=True)
        df.rename(columns=lambda x: x[1:] if x != 'tOffset' else x, inplace=True)
        # Add the mask values and replace the remaining null values
        df1 = df.reset_index()
        for col in para_labels:
            df1[col + '_mask'] = df1[col].isna()
        data = imputed_dataset.merge(df1[["tOffset"] + mask_labels], how="inner", on="tOffset")
        data[para_labels] = data[para_labels].fillna(input_weights)
        # standardize the dataset and prepare the time offsets
        inputs = scaler.transform(data[para_labels + mask_labels])
        tOffsets = data["tOffset"]

        # Compute the outputs of the network
        nodes = inputs
        for l_i, _ in enumerate(weights):
            nodes = self.compute_output(nodes, weights[l_i], biases[l_i], last_layer=(l_i == len(weights) - 1))
        if len(nodes.shape) == 1:
            nodes = np.reshape(nodes, (1, -1))

        # Get labels
        labels = np.zeros(nodes.shape)
        class_order = np.argsort(-nodes, axis=1)
        for i, l in enumerate([0., 1.]):
            np.place(labels, class_order == i, l)
        labels = labels[:, 0]

        # Prepare the outputs for NDAS
        # All measurements of a data point are set as Sensor Errors
        for column in datasets.columns[1:]:
            if column not in para_labels:
                continue
            novelties = {}
            for idx, label in enumerate(labels):
                novelties[tOffsets[idx]] = label
            result[column] = novelties
        return result

