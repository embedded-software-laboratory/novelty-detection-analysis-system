import numpy as np
import pandas as pd
from ndas.algorithms.basedetector import BaseDetector
from ndas.dataimputationalgorithms.base_imputation import BaseImputation
from sklearn.preprocessing import StandardScaler

class EssenceNeuralNetworkMod(BaseDetector):

    def __init__(self, *args, **kwargs):
        super(EssenceNeuralNetworkMod, self).__init__(*args, **kwargs)

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
        df.rename(columns=lambda x: "_" + x if x != 'tOffset' else x, inplace=True)
        imputed_dataset = BaseImputation().base_imputation(dataframe=df, method_string='neural inter')
        imputed_dataset.rename(columns=lambda x: x[1:] if x != 'tOffset' else x, inplace=True)
        df.rename(columns=lambda x: x[1:] if x != 'tOffset' else x, inplace=True)
        # Add the mask values and replance the remaining null values
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

        for col in para_labels:
            if col in datasets.columns[1:]:
                result[col] = {}

        # Prepare the outputs for NDAS
        # Only measurements with the highest influence on the output are selected
        for idx, label in enumerate(labels):
            # If the label is 0, then the data point is not a novelty
            if label == 0:
                for column in result:
                    result[column][tOffsets[idx]] = 0
            else:
                X_input = inputs[idx]
                # Find which measurement affects the subconcept output the most
                weighted_diff_inputs, diff_output = self.compute_diff_input_weights(X_input, weights[0], biases[0])
                subc_output = self.compute_output(self.compute_output(inputs[idx], weights[0], biases[0]),
                                                                       weights[1], biases[1])
                msk_sc = np.where(subc_output > 0, True, False)[0]
                weighted_subc_inputs, out = self.compute_subc_input_weights(diff_output.reshape(-1,1), msk_sc, weights[1], biases[1])
                msk_diff = (diff_output > 0)
                clean_subc_inputs = np.array(weighted_subc_inputs)[msk_diff]
                final_weighted_inputs = weighted_diff_inputs[:, msk_diff] @ clean_subc_inputs
                # The contributions of the mask values are ignored
                final_contributions = final_weighted_inputs.mean(axis=1)[:9]
                # Only positive contributions are considered as sensor errors, the rest are considered normal measurements
                indices = np.nonzero(final_contributions > 0)
                v_indices = np.nonzero(final_contributions <= 0)
                for final_index in indices[0]:
                    if para_labels[final_index] in result:
                        result[para_labels[final_index]][tOffsets[idx]] = 1
                for final_index in v_indices[0]:
                    if para_labels[final_index] in result:
                        result[para_labels[final_index]][tOffsets[idx]] = 0

        return result

    @staticmethod
    def compute_output(input, weights, biases, last_layer=False):
        """Returns the output of the layer given the previous layer as input"""
        if len(input.shape) == 1:  # fixes the shape of the input layer to be a 2d array
            input = np.reshape(input, (1, -1))
        nodes = input @ weights
        nodes = np.add(nodes, np.tile(biases, (nodes.shape[0], 1)))
        if last_layer:
            nodes = nodes - np.max(nodes)
            nodes = np.exp(nodes)
            sum_total = np.sum(nodes, 1)
            nodes = np.divide(nodes, np.tile(sum_total, (nodes.shape[1], 1)).transpose())
        else:
            if np.max(np.abs((input - np.round(input)))) < 1e-3:
                nodes = np.round(nodes)
            np.putmask(nodes, nodes > 1, 1)
            np.putmask(nodes, nodes < -1, -1)
            np.putmask(nodes, np.logical_and(nodes > -1, nodes < 1), 0)
            nodes += 1
            nodes /= 2

        return nodes

    @staticmethod
    def compute_diff_input_weights(x_pt, weights, biases):
        """Computes the weighted importance of each input on the output of the differentia neuron"""
        x_pt = x_pt.reshape(-1, 1)
        weighted_inputs = np.multiply(weights, x_pt)
        out = weighted_inputs.sum(0) + biases
        scaled_inputs = np.divide(weighted_inputs, np.abs(out))
        return scaled_inputs, out

    @staticmethod
    def compute_subc_input_weights(X_input, msk, weights, biases):
        """Computes the weighted importance of each input on the a subconcept neruon"""
        weighted_inputs = np.multiply(weights[:, msk], X_input)
        out = weighted_inputs.sum(0) + biases[msk]
        scaled_inputs = np.divide(weighted_inputs, np.abs(out))
        return scaled_inputs, out

