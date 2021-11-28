import pandas as pd
import numpy as np
from scipy import stats
import sklearn
import statistics
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.impute._base import _check_inputs_dtype
from sklearn.base import clone
from sklearn.utils import _safe_indexing, is_scalar_nan
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.utils._mask import _get_mask
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge, RANSACRegressor, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
import logging
import random
from itertools import chain
import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
from keras.wrappers.scikit_learn import KerasRegressor
import hickle as hkl
from . import sklearn_imputation as si
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, LSTM, Reshape, Input, Concatenate, ReLU
from keras.metrics import RootMeanSquaredError, MeanAbsolutePercentageError
from keras.optimizer_v2 import nadam
from keras.activations import relu
from datetime import datetime
import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
import multiprocessing


class TimeSeriesIterativeImputer(IterativeImputer):
    def __init__(
        self,
        estimator=None,
        *,
        missing_values=np.nan,
        sample_posterior=False,
        max_iter=10,
        tol=1e-3,
        n_nearest_features=None,
        initial_strategy="mean",
        imputation_order="ascending",
        skip_complete=False,
        min_value=-np.inf,
        max_value=np.inf,
        verbose=0,
        random_state=None,
        add_indicator=False,
        fit_file_path="C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\",
    ):
        super(TimeSeriesIterativeImputer, self).__init__(
            estimator=estimator,
            missing_values=missing_values,
            sample_posterior=False,
            max_iter=max_iter,
            tol=tol,
            n_nearest_features=n_nearest_features,
            initial_strategy=initial_strategy,
            imputation_order=imputation_order,
            skip_complete=skip_complete,
            min_value=min_value,
            max_value=max_value,
            verbose=verbose,
            random_state=random_state,
            add_indicator=add_indicator,
        )
        if sample_posterior:
            logging.warning("sample_posterior: True not supported . Fallback to False.")
        self.fit_file_path = fit_file_path

    def _impute_one_feature(
        self,
        X_filled,
        mask_missing_values,
        feat_idx,
        neighbor_feat_idx,
        estimator=None,
        fit_mode=True
    ):
        if estimator is None and fit_mode is False:
            raise ValueError(
                "If fit_mode is False, then an already-fitted "
                "estimator should be passed in."
            )

        if estimator is None:
            estimator = clone(self._estimator)

        missing_row_mask = mask_missing_values[:, feat_idx]
        if fit_mode:
            batch_split = hkl.load(self.fit_file_path + "batch_splits.hkl")
            batch_split_agg = [sum(batch_split[:i + 1]) for i in range(len(batch_split))][:-1]
            X_split = np.split(X_filled, batch_split_agg)
            missing_split = np.split(missing_row_mask, batch_split_agg)
            list_of_y_scope_arrays = []
            list_of_batches = []
            for batch, missing in zip(X_split, missing_split):
                if not missing[0]:
                    y_pad = np.pad(batch[:, feat_idx], 10, constant_values=self.initial_imputer_.statistics_[feat_idx])
                    y_scope_array = np.stack([np.roll(y_pad, shift=i, axis=0) for i in chain(range(-10, 0), range(1, 11))], axis=1)
                    list_of_y_scope_arrays.append(y_scope_array[10:-10])
                    list_of_batches.append(batch)

            if type(estimator) is KerasRegressor:
                list_full_batches = []
                list_partial_batches = []
                list_full_y = []
                list_partial_y = []
                for batch, y_scope_array in zip(list_of_batches, list_of_y_scope_arrays):
                    X_train = np.concatenate([batch[:, neighbor_feat_idx], y_scope_array], axis=1)
                    split_x = np.split(X_train, range(48, X_train.shape[0], 48))
                    list_full_batches += split_x[:-1]
                    list_partial_batches.append(split_x[-1])
                    y_train = batch[:, feat_idx]
                    split_y = np.split(y_train, range(48, X_train.shape[0], 48))
                    list_full_y += split_y[:-1]
                    list_partial_y.append(split_y[-1])
                dataset = tf.data.Dataset.from_tensor_slices((list_full_batches, list_full_y))
                # for par_batch, par_y in zip(list_partial_batches, list_partial_y):
                #     dataset = dataset.concatenate(tf.data.Dataset.from_tensors((par_batch, par_y)))
                # print(dataset)
                dataset = dataset.shuffle(1024).repeat(3)
                print("Current Feature Index: "+str(feat_idx))
                print(datetime.now())
                estimator.fit(dataset, y=None, epochs=3, shuffle=False, batch_size=48, verbose=2)
            else:
                X_train = np.concatenate([np.concatenate(list_of_batches, axis=0)[:, neighbor_feat_idx], np.concatenate(list_of_y_scope_arrays, axis=0)], axis=1)
                y_train = np.concatenate(list_of_batches, axis=0)[:, feat_idx]
                estimator.fit(X_train, y_train)
        else:
            y_pad = np.pad(X_filled[:, feat_idx], 10, constant_values=self.initial_imputer_.statistics_[feat_idx])
            stacked_y_scope_array = np.stack([np.roll(y_pad, shift=i, axis=0) for i in chain(range(-10, 0), range(1, 11))], axis=1)[10:-10]
        # if no missing values, don't predict
        if np.sum(missing_row_mask) == 0:
            return X_filled, estimator

        if fit_mode:
            list_of_test_y_scope_arrays = []
            list_of_test_batches = []
            list_imputed_result_keras = np.array([])
            for batch, missing in zip(X_split, missing_split):
                y_pad = np.pad(batch[:, feat_idx], 10, constant_values=self.initial_imputer_.statistics_[feat_idx])
                y_scope_array = np.stack(
                    [np.roll(y_pad, shift=i, axis=0) for i in chain(range(-10, 0), range(1, 11))], axis=1)
                list_of_test_y_scope_arrays.append(y_scope_array[10:-10])
                list_of_test_batches.append(batch[:, neighbor_feat_idx])
                if type(estimator) is KerasRegressor:
                    if missing[0]:
                        list_imputed_result_keras = np.append(list_imputed_result_keras, estimator.predict(np.concatenate([list_of_test_batches[-1], list_of_test_y_scope_arrays[-1]], axis=1), batch_size=128))
                    else:
                        list_imputed_result_keras = np.append(list_imputed_result_keras, batch[:, feat_idx])
            X_test = np.concatenate([np.concatenate(list_of_test_batches, axis=0), np.concatenate(list_of_test_y_scope_arrays, axis=0)], axis=1)

            if type(estimator) is KerasRegressor:
                imputed_values = list_imputed_result_keras
                filestring = "KerasModel_" + str(feat_idx) + "_" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + ".h5"
                estimator.model.save(filestring)
                estimator = filestring
            else:
                imputed_values = estimator.predict(X_test)

        else:
            X_test = np.concatenate([X_filled[:, neighbor_feat_idx], stacked_y_scope_array], axis=1)

            if type(self._estimator) is KerasRegressor:
                predictor = clone(self._estimator)
                predictor.model = load_model("C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\"+estimator)
                imputed_values = predictor.predict(X_test, batch_size=128)
            else:
                imputed_values = estimator.predict(X_test)

        imputed_values = np.clip(
            imputed_values, self._min_value[feat_idx], self._max_value[feat_idx]
        )

        # update the feature
        X_filled[missing_row_mask, feat_idx] = imputed_values[missing_row_mask]
        return X_filled, estimator

    def _initial_imputation(self, X, in_fit=False):
        if is_scalar_nan(self.missing_values):
            force_all_finite = "allow-nan"
        else:
            force_all_finite = True

        X = self._validate_data(
            X,
            dtype=FLOAT_DTYPES,
            order="F",
            reset=in_fit,
            force_all_finite=force_all_finite,
        )
        _check_inputs_dtype(X, self.missing_values)

        X_missing_mask = _get_mask(X, self.missing_values)
        mask_missing_values = X_missing_mask.copy()
        if self.initial_imputer_ is None:
            self.initial_imputer_ = SimpleImputer(
                missing_values=self.missing_values, strategy=self.initial_strategy
            )
            self.initial_imputer_.fit(X)
        if in_fit and os.path.isfile(self.fit_file_path+"batch_splits.hkl"):
            batch_split = hkl.load(self.fit_file_path+"batch_splits.hkl")
            batch_split_agg = [sum(batch_split[:i+1]) for i in range(len(batch_split))][:-1]
            X_split = np.split(X, batch_split_agg)
            X_split_interpolated = []
            for batch in X_split:
                X_split_interpolated.append(pd.DataFrame(batch).interpolate(limit_area=None, limit_direction='both').to_numpy())
            Xb = np.concatenate(X_split_interpolated)
        elif not in_fit:
            Xb = pd.DataFrame(X).interpolate(limit_area=None, limit_direction='both').to_numpy()
        X_filled = self.initial_imputer_.transform(Xb)

        valid_mask = np.flatnonzero(
            np.logical_not(np.isnan(self.initial_imputer_.statistics_))
        )
        Xt = X[:, valid_mask]
        mask_missing_values = mask_missing_values[:, valid_mask]

        return Xt, X_filled, mask_missing_values, X_missing_mask


class KerasNNSlidingWindow:
    def __init__(self, window_size=48, fit_file_path="C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\"):
        if window_size > 50:
            self.window_size = 50
        elif window_size < 0:
            self.window_size = 1
        else:
            self.window_size = window_size
        self.input_dim = (window_size, 122)
        self.model = self.build_nn()
        self.fit_file_path = fit_file_path
        self.curr_epoch = 0
        self.filestring_prefix = "KerasModel_NN_SW_"

    def build_nn(self):
        model = Sequential()

        model.add(Dense(128, input_shape=self.input_dim, kernel_initializer='lecun_normal', activation='selu'))
        model.add(Dense(64, kernel_initializer='lecun_normal', activation='selu'))
        model.add(Dense(32, kernel_initializer='lecun_normal', activation='selu'))
        model.add(Dense(9, activation='tanh'))
        model.summary()
        model.compile(optimizer='nadam', loss='huber', metrics=[RootMeanSquaredError()])
        return model

    def fit(self, training_data, epochs=20, batch_size=128):
        if type(self.model) is str:
            self.model = load_model(self.fit_file_path+self.model)

        batch_split = hkl.load(self.fit_file_path + "batch_splits.hkl")
        batch_split_agg = [sum(batch_split[:i + 1]) for i in range(len(batch_split))][:-1]
        x_split = np.split(training_data, batch_split_agg)
        x_split_interpolated = []
        for batch in x_split:
            x_split_interpolated.append(pd.DataFrame(batch).interpolate(limit_area=None, limit_direction='both').to_numpy())
        xb = np.concatenate(x_split_interpolated)
        initial_imputer_ = SimpleImputer(missing_values=np.nan, strategy="median")
        x_filled = initial_imputer_.fit_transform(xb)
        x_filled_split = np.split(x_filled, batch_split_agg)
        samples = []
        targets = []
        for dataset in x_filled_split:
            samples += [dataset[i:i+self.window_size] for i in range(len(dataset)+1-self.window_size)]
            targets += [dataset[i:i+self.window_size, 0:9] for i in range(len(dataset)+1-self.window_size)]
        samples, targets = zip(*random.sample(list(zip(samples, targets)), 128000))
        input_data = np.stack(samples)
        input_targets = np.stack(targets)
        drop_mask = np.random.random(input_targets.shape)
        drop_mask = np.where(drop_mask <= 0.3, 0, 1)
        drop_ones = np.ones((input_data.shape[0], input_data.shape[1], input_data.shape[2]-input_targets.shape[2]))
        drop = np.concatenate([drop_mask, drop_ones], axis=2)
        randoms = np.random.random(input_data.shape)
        epoch_input_data = np.where(drop, input_data, randoms)
        epoch_input_data = np.concatenate([epoch_input_data, drop], axis=2)
        print("Epoch "+ str(self.curr_epoch+1)+"/"+str(epochs))
        self.curr_epoch += 1
        self.model.fit(epoch_input_data, input_targets, batch_size=batch_size, epochs=1, validation_split=0.05)
        filestring = self.filestring_prefix + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + ".h5"
        self.model.save(filestring)
        self.model = filestring

    def transform(self, test_data, batch_size=128):
        self.model = load_model(self.fit_file_path+self.model)
        test_samples = [test_data[i:i+self.window_size] for i in range(len(test_data)+1-self.window_size)]
        test_input = np.stack(test_samples)
        mask = 1*~np.isnan(test_input)
        randoms = np.random.random(test_input.shape)
        input_data = np.where(mask, test_input, randoms)
        input_data = np.concatenate([input_data, mask], axis=2)
        results = self.model.predict(input_data, batch_size=batch_size)
        result_matrix = np.empty((results.shape[0]+results.shape[1]-1, results.shape[1], results.shape[2]))
        result_matrix[:] = np.nan
        for i in range(results.shape[1]):
            result_matrix[i:i+results.shape[0], i] = results[:, i]
        colapsed_result = np.nanmean(result_matrix, axis=1)
        colapsed_result = np.clip(colapsed_result, 0, 1)
        test_data[:, 0:9][np.isnan(test_data)[:, 0:9]] = colapsed_result[np.isnan(test_data)[:, 0:9]]
        return test_data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


class KerasLSTMSlidingWindow(KerasNNSlidingWindow):
    def __init__(self, window_size=48, fit_file_path="C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\"):
        super(KerasLSTMSlidingWindow, self).__init__(window_size=window_size, fit_file_path=fit_file_path)
        self.filestring_prefix = "KerasModel_LSTM_SW_"

    def build_nn(self):
        model = Sequential()

        model.add(LSTM(256, input_shape=self.input_dim))
        model.add(Dense(256, kernel_initializer='lecun_normal', activation='selu'))
        model.add(Dense(512, kernel_initializer='lecun_normal', activation='selu'))
        model.add(Dense(432, activation='tanh'))
        model.add(Reshape((48,9)))
        model.summary()
        model.compile(optimizer='nadam', loss='huber', metrics=[RootMeanSquaredError()])
        return model


class KerasGANSlidingWindow:
    def __init__(self, window_size=48, fit_file_path="C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\"):
        if window_size > 50:
            self.window_size = 50
        elif window_size < 0:
            self.window_size = 1
        else:
            self.window_size = window_size
        self.input_dim = (self.window_size, 61*2)
        self.gen = self.build_model("GEN")
        self.gen.compile(optimizer=nadam.Nadam(name="Gen_Nadam"), loss='mean_squared_error', metrics=[RootMeanSquaredError()])
        self.dis = self.build_model("DIS")
        self.dis.compile(optimizer=nadam.Nadam(name="Dis_Nadam"), loss='binary_crossentropy', metrics=['accuracy'])
        # self.dis.trainable = False
        # self.combined = self.build_combined(self.gen, self.dis)
        # self.combined.compile(optimizer=nadam.Nadam(name="Comb_Nadam"), loss=['huber', 'binary_crossentropy'], loss_weights=[0.97, 0.03], metrics={'Generator': RootMeanSquaredError()})
        # self.dis.trainable = True
        self.combined = "This will not be saved but has to be built from dis and gen"
        self.fit_file_path = fit_file_path
        self.curr_epoch = 0
        self.filestring_prefix = "KerasModel_GAN_SW_"

    def build_model(self, name):
        data_input = Input((self.input_dim[0], int(self.input_dim[1]/2)))
        mask_input = Input((self.input_dim[0], int(self.input_dim[1]/2)))
        concat_in = Concatenate(name=name+"_Concat_in")([data_input, mask_input])
        concat_in_aft = Dense(36, kernel_initializer='lecun_normal', activation='selu', name=name+"_Dense0")(concat_in)
        dense_one_out_pre = Reshape((self.window_size*36,), name=name+"_Res_in")(concat_in_aft)
        dense_one_out = Dense(self.window_size*10, kernel_initializer='lecun_normal', activation='selu', name=name+"_Dense1")(dense_one_out_pre)
        lstm_one_out = LSTM(self.window_size*5, input_shape=(self.input_dim[0], int(self.input_dim[1]/2)), name=name+"_LSTM")(data_input)
        concat_two_out = Concatenate(name=name+"_Concat2")([dense_one_out, lstm_one_out])
        combined_dense_two_out = Dense(self.window_size*10, kernel_initializer='lecun_normal', activation='selu', name=name+"_Dense2")(concat_two_out)
        combined_dense_three_out = Dense(self.window_size*9, kernel_initializer='lecun_normal', activation='selu', name=name+"_Dense3")(combined_dense_two_out)
        normalized_out = ReLU(max_value=1, name=name+"_Relu")(combined_dense_three_out)
        reshaped_normalized_out = Reshape((self.window_size, 9), name=name+"_Res")(normalized_out)
        model = Model(inputs=[data_input, mask_input], outputs=reshaped_normalized_out, name=name)
        model.summary()
        return model

    def build_generator(self):
        model = Sequential(name="Generator")

        model.add(LSTM(256, input_shape=self.input_dim, name="Gen_LSTM"))
        model.add(Dense(256, kernel_initializer='lecun_normal', activation='selu', name="Gen_Dense1"))
        model.add(Dense(512, kernel_initializer='lecun_normal', activation='selu', name="Gen_Dense2"))
        model.add(Dense(self.window_size*9, activation='relu', name="Gen_Dense3"))
        model.add(ReLU(max_value=1, name="Gen_Relu"))
        model.add(Reshape((self.window_size, 9), name="Gen_Res"))
        model.summary()
        return model

    def build_discriminator(self):
        model = Sequential(name="Discriminator")

        model.add(LSTM(256, input_shape=self.input_dim, name="Dis_LSTM"))
        model.add(Dense(256, kernel_initializer='lecun_normal', activation='selu', name="Dis_Dense1"))
        model.add(Dense(512, kernel_initializer='lecun_normal', activation='selu', name="Dis_Dense2"))
        model.add(Dense(self.window_size*9, activation='relu', name="Dis_Dense3"))
        model.add(ReLU(max_value=1, name="Dis_Relu"))
        model.add(Reshape((self.window_size, 9), name="Dis_Res"))
        model.summary()
        return model

    def build_combined(self, gen, dis):
        gen_data_input = Input((self.input_dim[0], int(self.input_dim[1]/2)))
        gen_mask_input = Input((self.input_dim[0], int(self.input_dim[1]/2)))
        dis_data_input_b = Input((self.input_dim[0], int(self.input_dim[1]/2)-9))
        dis_mask_input = Input((self.input_dim[0], int(self.input_dim[1]/2)))
        gen_out = gen([gen_data_input, gen_mask_input])
        concat_out = Concatenate()([gen_out, dis_data_input_b])
        dis_out = dis([concat_out, dis_mask_input])
        model = Model(inputs=[gen_data_input, gen_mask_input, dis_data_input_b, dis_mask_input], outputs=[gen_out, dis_out])
        model.summary()
        return model

    def fit(self, training_data, epochs=20, batch_size=64, batches_per_epoch=2000):
        if type(self.gen) is str:
            self.gen = load_model(self.fit_file_path+self.gen)
        if type(self.dis) is str:
            self.dis = load_model(self.fit_file_path+self.dis)
        if self.curr_epoch > 2:
            self.dis.trainable = False
            self.combined = self.build_combined(self.gen, self.dis)
            self.combined.compile(optimizer=nadam.Nadam(name="Comb_Nadam"), loss=['mean_squared_error', 'binary_crossentropy'], loss_weights=[(1 - (self.curr_epoch-2)/100), (self.curr_epoch-2)/100], metrics={'GEN': RootMeanSquaredError()})
            self.dis.trainable = True

        batch_split = hkl.load(self.fit_file_path + "batch_splits.hkl")
        batch_split_agg = [sum(batch_split[:i + 1]) for i in range(len(batch_split))][:-1]
        x_split = np.split(training_data, batch_split_agg)
        x_split_interpolated = []
        for batch in x_split:
            x_split_interpolated.append(pd.DataFrame(batch).interpolate(limit_area=None, limit_direction='both').to_numpy())
        xb = np.concatenate(x_split_interpolated)
        initial_imputer_ = SimpleImputer(missing_values=np.nan, strategy="median")
        x_filled = initial_imputer_.fit_transform(xb)
        x_filled_split = np.split(x_filled, batch_split_agg)
        samples_big = []
        targets_big = []
        for dataset in x_filled_split:
            samples_big += [dataset[i:i+self.window_size] for i in range(len(dataset)+1-self.window_size)]
            targets_big += [dataset[i:i+self.window_size, 0:9] for i in range(len(dataset)+1-self.window_size)]

        n_samples = batch_size * batches_per_epoch

        samples_a, targets_a = zip(*random.sample(list(zip(samples_big, targets_big)), n_samples))
        input_data_a = np.stack(samples_a)
        input_targets_a = np.stack(targets_a)
        drop_mask_a = np.random.random(input_targets_a.shape)
        drop_mask_a = np.where(drop_mask_a <= 0.3, 0, 1)
        drop_ones_a = np.ones((input_data_a.shape[0], input_data_a.shape[1], input_data_a.shape[2]-input_targets_a.shape[2]))
        drop_a = np.concatenate([drop_mask_a, drop_ones_a], axis=2)
        randoms_a = np.random.random(input_data_a.shape)
        epoch_input_data_a = np.where(drop_a, input_data_a, randoms_a)
        # epoch_input_data_a = np.concatenate([epoch_input_data_a, drop_a], axis=2)
        print("Finished Sampling A")
        dis_input_data_a = epoch_input_data_a[:, :, 9:]
        dis_mask_a = drop_a
        masking_v_a = np.random.randint(low=0, high=9, size=(dis_input_data_a.shape[0], dis_input_data_a.shape[1]))
        masking_a = np.zeros((masking_v_a.size, 9), dtype=np.uint8)
        masking_a[np.arange(masking_v_a.size), masking_v_a.ravel()] = 1
        masking_a.shape = masking_v_a.shape + (9,)
        dis_mask_a[:, :, :9][masking_a] = 0.5
        print("Finished Masking A")

        samples_b, targets_b = zip(*random.sample(list(zip(samples_big, targets_big)), n_samples*2))
        input_data_b = np.stack(samples_b)
        input_targets_b = np.stack(targets_b)
        drop_mask_b = np.random.random(input_targets_b.shape)
        drop_mask_b = np.where(drop_mask_b <= 0.3, 0, 1)
        drop_ones_b = np.ones((input_data_b.shape[0], input_data_b.shape[1], input_data_b.shape[2] - input_targets_b.shape[2]))
        drop_b = np.concatenate([drop_mask_b, drop_ones_b], axis=2)
        randoms_b = np.random.random(input_data_b.shape)
        epoch_input_data_b = np.where(drop_b, input_data_b, randoms_b)
        # epoch_input_data_b = np.concatenate([epoch_input_data_b, drop_b], axis=2)
        print("Finished Sampling B")

        dis_input_data_b = epoch_input_data_b[:, :, 9:]
        dis_mask_b = drop_b
        masking_v_b = np.random.randint(low=0, high=9, size=(dis_input_data_b.shape[0], dis_input_data_b.shape[1]))
        masking_b = np.zeros((masking_v_b.size, 9), dtype=np.uint8)
        masking_b[np.arange(masking_v_b.size), masking_v_b.ravel()] = 1
        masking_b.shape = masking_v_b.shape + (9,)
        dis_mask_b[:, :, :9][masking_b] = 0.5
        print("Finished Masking B")
        dis_loss=[]
        dis_acc=[]
        gen_loss=[]
        gen_rmse=[]
        for i in range(batches_per_epoch):

            gen_res = self.gen.predict([epoch_input_data_a[i*batch_size:(i+1)*batch_size], drop_a[i*batch_size:(i+1)*batch_size]])
            gen_res[drop_mask_a[i*batch_size:(i+1)*batch_size]] = input_targets_a[i*batch_size:(i+1)*batch_size][drop_mask_a[i*batch_size:(i+1)*batch_size]]
            dis_input_data = np.concatenate([gen_res, dis_input_data_a[i*batch_size:(i+1)*batch_size]], axis=2)

            dis_metrics = self.dis.train_on_batch([dis_input_data, dis_mask_a[i*batch_size:(i+1)*batch_size]], drop_mask_a[i*batch_size:(i+1)*batch_size])
            dis_loss += [dis_metrics[0]]
            dis_acc += [dis_metrics[1]]
            if self.curr_epoch > 2:
                comb_metrics = self.combined.train_on_batch([epoch_input_data_b[2*i*batch_size:(2*i+1)*batch_size], drop_b[2*i*batch_size:(2*i+1)*batch_size], dis_input_data_b[2*i*batch_size:(2*i+1)*batch_size], dis_mask_b[2*i*batch_size:(2*i+1)*batch_size]], [input_targets_b[2*i*batch_size:(2*i+1)*batch_size], np.ones((batch_size, input_targets_b.shape[1], input_targets_b.shape[2]))])
                gen_loss += comb_metrics[:2]
                gen_rmse += [comb_metrics[2]]
            else:
                gen_metrics = self.gen.train_on_batch([epoch_input_data_b[(2*i+1)*batch_size:2*(i+1)*batch_size], drop_b[(2*i+1)*batch_size:2*(i+1)*batch_size]], input_targets_b[(2*i+1)*batch_size:2*(i+1)*batch_size])
                gen_loss += [gen_metrics[0]]
                gen_rmse += [gen_metrics[1]]
            print("\rProgress: %.1f%%, [D loss: %.3f, acc.: %.2f%%] [G loss: %.3f, rmse: %.3f]" % ((i+1)/batches_per_epoch*100, statistics.mean(dis_loss[-min(50,(i+1)):]), 100*statistics.mean(dis_acc[-min(50,(i+1)):]), statistics.mean(gen_loss[-min(50,(i+1)):]), statistics.mean(gen_rmse[-min(50,(i+1)):])), end="")

        self.curr_epoch += 1
        print("Epochs "+ str(self.curr_epoch)+"/"+str(epochs))
        filestring = self.filestring_prefix + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + ".h5"
        self.gen.save("GEN"+filestring)
        self.gen = "GEN"+filestring
        self.dis.save("DIS" + filestring)
        self.dis = "DIS" + filestring
        # self.combined.save("COM" + filestring)
        # self.combined = "COM" + filestring

    def transform(self, test_data, batch_size=128):
        self.gen = load_model(self.fit_file_path+self.gen)
        test_samples = [test_data[i:i+self.window_size] for i in range(len(test_data)+1-self.window_size)]
        test_input = np.stack(test_samples)
        mask = 1*~np.isnan(test_input)
        randoms = np.random.random(test_input.shape)
        input_data = np.where(mask, test_input, randoms)
        input_data = np.concatenate([input_data, mask], axis=2)
        results = self.gen.predict(input_data, batch_size=batch_size)
        result_matrix = np.empty((results.shape[0]+results.shape[1]-1, results.shape[1], results.shape[2]))
        result_matrix[:] = np.nan
        for i in range(results.shape[1]):
            result_matrix[i:i+results.shape[0], i] = results[:, i]
        colapsed_result = np.nanmean(result_matrix, axis=1)
        colapsed_result = np.clip(colapsed_result, 0, 1)
        test_data[:, 0:9][np.isnan(test_data)[:, 0:9]] = colapsed_result[np.isnan(test_data)[:, 0:9]]
        return test_data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


class KerasImputation:
    """
    Implements imputation methods from scikit-learn and Keras using the custom IterativeImputer class defined above
    """
    def __init__(self):
        self.skli = si.SklearnImputation()

    def mice_imputation(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=True):
        lr = LinearRegression(positive=True)
        imputer = TimeSeriesIterativeImputer(estimator=lr, missing_values=np.nan, max_iter=15, initial_strategy='median',
                                   skip_complete=True, min_value=0, max_value=1, tol=1e-6, imputation_order='roman')
        if already_fit:
            imputer = hkl.load('C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\lr_ts.hkl')
        return self.skli.iterative_imputation(dataframe, imputer, already_fit=already_fit)

    def bayesian_imputation(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=True):
        bay = BayesianRidge(tol=1e-6)
        imputer = TimeSeriesIterativeImputer(estimator=bay, missing_values=np.nan, max_iter=15, initial_strategy='median',
                                   skip_complete=True, min_value=0, max_value=1, tol=1e-6, imputation_order='roman')
        if already_fit:
            imputer = hkl.load('C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\bay_ts.hkl')
        return self.skli.iterative_imputation(dataframe, imputer, already_fit=already_fit)

    def mlp_imputation(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=True):
        mlp = MLPRegressor(activation='tanh', learning_rate='adaptive', learning_rate_init=0.002, tol=1e-6,
                           warm_start=True, early_stopping=True)
        imputer = TimeSeriesIterativeImputer(estimator=mlp, missing_values=np.nan, max_iter=15, initial_strategy='median',
                                   skip_complete=True, min_value=0, max_value=1, tol=1e-6, imputation_order='roman')
        if already_fit:
            imputer = hkl.load('C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\mlp_ts.hkl')
        return self.skli.iterative_imputation(dataframe, imputer, already_fit=already_fit)

    def svr_imputation(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=True):
        svr = LinearSVR(tol=1e-6, loss='squared_epsilon_insensitive', dual=False)
        imputer = TimeSeriesIterativeImputer(estimator=svr, missing_values=np.nan, max_iter=15, initial_strategy='median',
                                   skip_complete=True, min_value=0, max_value=1, tol=1e-6, imputation_order='roman')
        if already_fit:
            imputer = hkl.load('C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\svr_ts.hkl')
        return self.skli.iterative_imputation(dataframe, imputer, already_fit=already_fit)

    def tree_imputation(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=True):
        dtr = DecisionTreeRegressor(criterion='friedman_mse', min_samples_split=3, min_samples_leaf=2, ccp_alpha=0.01)
        imputer = TimeSeriesIterativeImputer(estimator=dtr, missing_values=np.nan, max_iter=15, initial_strategy='median',
                                   skip_complete=True, min_value=0, max_value=1, tol=1e-6, imputation_order='roman')
        if already_fit:
            imputer = hkl.load('C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\dtr_ts.hkl')
        return self.skli.iterative_imputation(dataframe, imputer, already_fit=already_fit)

    def extra_tree_imputation(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=True):
        etr = ExtraTreeRegressor(criterion='friedman_mse', min_samples_split=3, min_samples_leaf=2, ccp_alpha=0.01)
        imputer = TimeSeriesIterativeImputer(estimator=etr, missing_values=np.nan, max_iter=15, initial_strategy='median',
                                   skip_complete=True, min_value=0, max_value=1, tol=1e-6, imputation_order='roman')
        if already_fit:
            imputer = hkl.load('C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\etr_ts.hkl')
        return self.skli.iterative_imputation(dataframe, imputer, already_fit=already_fit)

    def ransac_imputation(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=True):
        ransac = RANSACRegressor(loss='squared_error')
        imputer = TimeSeriesIterativeImputer(estimator=ransac, missing_values=np.nan, max_iter=15, initial_strategy='median',
                                   skip_complete=True, min_value=0, max_value=1, tol=1e-6, imputation_order='roman')
        if already_fit:
            imputer = hkl.load('C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\ransac_ts.hkl')
        return self.skli.iterative_imputation(dataframe, imputer, already_fit=already_fit)

    def sgd_imputation(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=True):
        sgd = SGDRegressor(tol=1e-6, learning_rate='adaptive', early_stopping=True, warm_start=True)
        imputer = TimeSeriesIterativeImputer(estimator=sgd, missing_values=np.nan, max_iter=15, initial_strategy='median',
                                   skip_complete=True, min_value=0, max_value=1, tol=1e-6, imputation_order='roman')
        if already_fit:
            imputer = hkl.load('C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\sgd_ts.hkl')
        return self.skli.iterative_imputation(dataframe, imputer, already_fit=already_fit)

    def ada_imputation(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=True):
        ada = AdaBoostRegressor(loss='square')
        imputer = TimeSeriesIterativeImputer(estimator=ada, missing_values=np.nan, max_iter=15, initial_strategy='median',
                                   skip_complete=True, min_value=0, max_value=1, tol=1e-6, imputation_order='roman')
        if already_fit:
            imputer = hkl.load('C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\ada_ts.hkl')
        return self.skli.iterative_imputation(dataframe, imputer, already_fit=already_fit)

    def kerlstm_imputation(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=True):
        kerlstm = KerasRegressor(build_fn=self.create_lstm, epochs=3, batch_size=48, verbose=0)
        imputer = TimeSeriesIterativeImputer(estimator=kerlstm, missing_values=np.nan, max_iter=2, initial_strategy='median',
                                   skip_complete=True, min_value=0, max_value=1, tol=1e-6, imputation_order='roman')
        if already_fit:
            imputer = hkl.load('C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\kerlstm_ts.hkl')
        return self.skli.iterative_imputation(dataframe, imputer, already_fit=already_fit)

    def kernn_imputation(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=True):
        kernn = KerasRegressor(build_fn=self.create_nn, epochs=3, batch_size=48, verbose=0)
        imputer = TimeSeriesIterativeImputer(estimator=kernn, missing_values=np.nan, max_iter=2, initial_strategy='median',
                                   skip_complete=True, min_value=0, max_value=1, tol=1e-6, imputation_order='roman')
        if already_fit:
            imputer = hkl.load('C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\kernn_ts.hkl')
        return self.skli.iterative_imputation(dataframe, imputer, already_fit=already_fit)

    def kernn_sw_imputation(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=True):
        if already_fit:
            imputer = hkl.load('C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\kernn_sw.hkl')
        else:
            imputer = KerasNNSlidingWindow()

        return self.skli.iterative_imputation(dataframe, imputer, already_fit=already_fit)

    def kerlstm_sw_imputation(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=True):
        if already_fit:
            imputer = hkl.load('C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\kerlstm_sw.hkl')
        else:
            imputer = KerasLSTMSlidingWindow()

        return self.skli.iterative_imputation(dataframe, imputer, already_fit=already_fit)

    def kergan_sw_imputation20(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=True):
        if already_fit:
            imputer = hkl.load('C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\kergan_sw.hkl')
        else:
            imputer = KerasGANSlidingWindow()

        return self.skli.iterative_imputation(dataframe, imputer, already_fit=already_fit)

    def kergan_sw_imputation5(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=True):
        if already_fit:
            imputer = hkl.load('C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\kergan_sw.hkl')
            imputer.gen = "GENKerasModel_GAN_SW_08-11-2021_20-48-18.h5"
        else:
            imputer = KerasGANSlidingWindow()

        return self.skli.iterative_imputation(dataframe, imputer, already_fit=already_fit)

    def kergan_sw_imputation8(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=True):
        if already_fit:
            imputer = hkl.load('C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\kergan_sw.hkl')
            imputer.gen = "GENKerasModel_GAN_SW_08-11-2021_21-17-09.h5"
        else:
            imputer = KerasGANSlidingWindow()

        return self.skli.iterative_imputation(dataframe, imputer, already_fit=already_fit)

    def kergan_sw_imputation10(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=True):
        if already_fit:
            imputer = hkl.load('C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\kergan_sw.hkl')
            imputer.gen = "GENKerasModel_GAN_SW_08-11-2021_21-36-05.h5"
        else:
            imputer = KerasGANSlidingWindow()

        return self.skli.iterative_imputation(dataframe, imputer, already_fit=already_fit)

    def kergan_sw_imputation15(self, dataframe, lim_dir='forward', lim_are='inside', already_fit=True):
        if already_fit:
            imputer = hkl.load('C:\\Users\\AlexanderKruschewsky\\PycharmProjects\\pythonProject\\kergan_sw.hkl')
            imputer.gen = "GENKerasModel_GAN_SW_09-11-2021_11-44-42.h5"
        else:
            imputer = KerasGANSlidingWindow()

        return self.skli.iterative_imputation(dataframe, imputer, already_fit=already_fit)

    def create_lstm(self, name_add=("_"+str(datetime.now())).replace(":", ".").replace(" ", "_")):
        model = Sequential(name="lstm"+name_add)
        model.add(Reshape((1, 80), input_shape=(80,), name="lstm_reshape"+name_add))
        model.add(LSTM(96, name="lstm_lstm"+name_add))
        model.add(Dense(48, kernel_initializer='lecun_normal', activation='selu', name="lstm_dense1"+name_add))
        model.add(Dense(16, kernel_initializer='lecun_normal', activation='selu', name="lstm_dense2"+name_add))
        model.add(Dense(1, kernel_initializer='lecun_normal', activation='selu', name="lstm_dense3"+name_add))
        model.compile(optimizer='nadam', loss='huber', metrics=[RootMeanSquaredError(), MeanAbsolutePercentageError()])
        model.summary()
        return model

    def create_nn(self, name_add=("_"+str(datetime.now())).replace(":", ".").replace(" ", "_")):
        model = Sequential(name="nn"+name_add)
        model.add(Dense(192, input_shape=(80, ), kernel_initializer='lecun_normal', activation='selu', name="nn_dense1"+name_add))
        model.add(Dense(48, kernel_initializer='lecun_normal', activation='selu', name="nn_dense2"+name_add))
        model.add(Dense(16, kernel_initializer='lecun_normal', activation='selu', name="nn_dense3"+name_add))
        model.add(Dense(1, kernel_initializer='lecun_normal', activation='selu', name="nn_dense4"+name_add))
        model.compile(optimizer='nadam', loss='huber', metrics=[RootMeanSquaredError(), MeanAbsolutePercentageError()])
        model.summary()
        return model
