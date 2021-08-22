import bisect
import copy
import math

import numpy as np
import scipy as sp
from scipy import stats
from scipy.spatial import distance

from ndas.algorithms.basedetector import BaseDetector
from ndas.misc.parameter import ArgumentType


class SW_ABSAD(BaseDetector):
    """
    The unmodified SW-ABSAD algorithm
    """

    def __init__(self, *args, **kwargs):
        super(SW_ABSAD, self).__init__(*args, **kwargs)

        self.register_parameter("replace_zeroes", ArgumentType.BOOL, False)
        self.register_parameter("replace_phys_outlier", ArgumentType.BOOL, False)
        self.register_parameter("use_cl_modification", ArgumentType.BOOL, True)
        self.register_parameter("windowLength", ArgumentType.INTEGER, 144, 1)
        self.register_parameter("bandWidth", ArgumentType.FLOAT, 0.5, 0, 1,
                                tooltip="Parameter for the kernel density estimation.")
        self.register_parameter("theta", ArgumentType.FLOAT, 0.2, 0, 0.99,
                                tooltip="Parameter to calculate relevant subspaces.")
        self.register_parameter("confidence_soll", ArgumentType.FLOAT, 0.99, 0, 1)
        self.register_parameter("k_nearest_neighbor", ArgumentType.INTEGER, 60, 0,
                                tooltip="Use the k nearest neighbors for estimation.")
        self.register_parameter("shared_nearest_neighbor", ArgumentType.INTEGER, 30, 0,
                                tooltip="Two points are in the reference subset of each other if they have at least this many of the same k nearest neighbors.")
        self.register_parameter("use_columns", ArgumentType.STRING, "",
                                tooltip="Use only this list of columns to run the algorithmn. Leave empty for all columns.")

    def get_cosine_similarity(self, x, y):
        return abs(np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y))))

    def get_relevant_points(self, snn, data):
        _ds = np.copy(data)
        _ds = np.append(_ds, np.c_[snn], axis=1)  # number of nearest neighbors for index
        _ds = _ds[np.where(_ds[:, self.num_data_dimensions] > self.shared_nearest_neighbor)]
        _ds = np.delete(_ds, self.num_data_dimensions, 1)
        return _ds

    def get_column_subset(self, datasets, columns):
        _df = datasets.iloc[:, 0].to_frame()
        for column_number in columns:
            single_column = datasets.iloc[:, int(column_number)]
            _df = _df.join(single_column)
        return _df

    def calculate_subspace(self, current_vector, mean_vector):
        '''
        Dabei wird der jte Wert des Vektors jeweils einzeln mit jeder anderen Dimension,
        die nicht j ist, hier dargestellt durch 𝑙𝑗− verrechnet, um den Durchschnittswert
        über alle n-1 Kombinationen an 2D Räumen zu bekommen, die die Dimension j
        enthalten. Sollte der Datenpunkt in einer Dimension genau mit dem Mittelwert des
        Referenzdatensatzes übereinstimmen, also der Wert des Vektor gleich 0 sein, wird
        dieser Wert durch einen kleinen positiven Wert, zum Beispiel 10−5 ersetzt um eine
        Division durch 0 zu verhindern.

        Liegt der Durchschnittswert für eine Dimension j unter einem Grenzwert, wird
        diese Dimension als unwichtig angesehen und im Folgenden nicht mehr betrachtet.
        Der Grenzwert T wird mithilfe von Formel 4.2, als durchschnittlicher Wert aller
        paarweisen Cosinus Werte aller Dimensionen multipliziert mit 1 + 𝜃, definiert.

        Berechnung (4.1)

        Parameters
        ----------
        current_vector
        mean_vector
        '''
        l_vector = current_vector - mean_vector
        l_vector[l_vector == 0] = 10 ** -5

        pcos = [0.0 for _ in range(self.num_data_dimensions)]
        for j in range(self.num_data_dimensions):
            pcosj = 0.0
            for h in range(self.num_data_dimensions):
                if (j != h):
                    pcosj = pcosj + ((abs(l_vector[j] / math.sqrt(l_vector[j] ** 2 + l_vector[h] ** 2))))

            pcos[j] = pcosj * (1 / (self.num_data_dimensions - 1))

        # Berechnung (4.2)
        g_treshhold = (1 + self.theta) * (1 / self.num_data_dimensions) * sum(pcos)
        pcos_above_treshold = (pcos > g_treshhold)
        return pcos_above_treshold.astype(int)

    def calculate_cl(self, los_sliding_window):
        '''
        Das CL wird mithilfe eines Kernel Density Estimator (KDE) berechnet. Der KDE
        gibt dabei auf Basis der vorliegenden Stichprobe an LOSs eine Wahrscheinlichkeitsdichteverteilung
        an, die die Wahrscheinlichkeit des Vorliegens eines Punktes mit
        diesem Wert wiedergeben soll. Der Kernel ist hierbei die Gewichtung der Punkte,
        die zur Berechnung einer entsprechenden Kurve verwendet werden. Besonders häufig
        wird der Gaußsche Kernel verwendet.
        '''
        min_value = los_sliding_window[los_sliding_window != 0]

        '''
        Da Werte die genau bei 0 liegen die
        Kurve überproportional beeinflussen könnten, werden vor der Berechnung des CL
        alle 0 Werte der LOS Liste auf die Hälfte des Minimums der nicht Nullwerte gesetzt.
        '''
        for current_point in range(len(los_sliding_window)):
            if (los_sliding_window[current_point] == 0):
                los_sliding_window[current_point] = min(min_value) / 2

        kde = stats.gaussian_kde(los_sliding_window, self.bandWidth)

        '''
        In der konkreten Implementierung wird die
        gaussian_kde.integrate_box_1d Funktion des scipy.stats packages in Python gewählt.
        Dabei wird langsam ein Schwellenwert angenähert, bis das Konfidenzintervall
        einem vorgegebenem Wert entspricht. Als untere Schranke des evaluierten Bereichs wurde -10 gewählt,
        da die LOS-Werte mindestens Wert 0 haben und eine Anpassung der Verteilungskurve über 0 hinaus so
        in die Berechnung des CL mit einbezogen wird.
        '''
        CL = 0
        confidence_interval = 0
        while (confidence_interval < self.confidence_soll):  # first training: and CL < 5
            CL = CL + 0.001
            confidence_interval = kde.integrate_box_1d(-10, CL)

        return CL

    def calculate_los(self, point_vector, mean_vector):
        return (np.absolute(point_vector - mean_vector) / (
            np.linalg.norm(point_vector - mean_vector)))

    def calculate_mahalanobis(self, points, subspaces):
        '''
        Zur Berechnung der Mahalanobisdistanz sind der Mittelwertvektor
        und die invertierte Kovarianzmatrix zwischen dem Datenpunkt und
        dem Referenzdatensatz nötig. Der LOS jedes Punktes wird in einer gleichnamigen
        Matrix gespeichert. Das Risiko des Auftretens einer nonsingulären Kovarianz soll
        dabei durch die zuvor beschriebenen Mechanismen zur Beseitigung stark korrelierter
        Werte zwischen Datenpunkt und Referenzdatensatz und durch die entsprechende
        Wahl der Parameter im KNN und SNN-Algorithmus, zur Vermeidung eines leeren
        Referenzdatensatzes, reduziert werden.

        Parameters
        ----------
        points
        subspaces
        '''

        mahalanobis_ds = np.vstack([points, subspaces])
        mahalanobis_ds = mahalanobis_ds[:, ~np.any(np.isnan(mahalanobis_ds), axis=0)]

        mahalanobis_vector = mahalanobis_ds[len(mahalanobis_ds) - 1]
        mahalanobis_ds = np.delete(mahalanobis_ds, (len(mahalanobis_ds) - 1), axis=0)

        if (len(mahalanobis_vector) == 0):
            result = 0
        else:
            mean_vector = np.mean(mahalanobis_ds, axis=0)

            if (len(mahalanobis_vector) == 1):
                covariance_matrix = np.cov(mahalanobis_ds, rowvar=False)
                inverted_covariance_matrix = np.linalg.inv(covariance_matrix.reshape((1, 1)))
            else:
                cov = np.cov(mahalanobis_ds.T)
                det = np.linalg.det(cov)

                if (det == 0):
                    self.log('co-variance matrix is singular: using pseudo-inverse')
                    inverted_covariance_matrix = sp.linalg.pinv(cov)
                else:
                    inverted_covariance_matrix = sp.linalg.inv(cov)

            md = distance.mahalanobis(mahalanobis_vector, mean_vector, inverted_covariance_matrix)
            result = (1 / len(mahalanobis_vector)) * md

        return result

    def detect(self, df, **kwargs):
        self.replace_zeroes = bool(kwargs["replace_zeroes"])
        self.replace_phys_outlier = bool(kwargs["replace_phys_outlier"])
        self.use_cl_modification = bool(kwargs["use_cl_modification"])
        self.windowlength = int(kwargs["windowLength"])
        self.use_columns = kwargs["use_columns"].split(',')
        self.bandWidth = float(kwargs["bandWidth"])
        self.confidence_soll = float(kwargs["confidence_soll"])  # 0.95
        self.theta = float(kwargs["theta"])  # 0.2 # [0, 1)
        self.k_nearest_neighbor = int(kwargs["k_nearest_neighbor"])  # 60
        self.shared_nearest_neighbor = int(kwargs["shared_nearest_neighbor"])  # 30

        if (self.use_columns[0] != ''):
            df = self.get_column_subset(df, self.use_columns)

        if (self.replace_zeroes):
            df = df.replace(0, np.NaN)

        if (self.replace_phys_outlier):

            # temperature
            if ('temperature' in df.columns):
                df['temperature'] = np.where(df['temperature'] > 40, np.nan, df['temperature'])
                df['temperature'] = np.where(df['temperature'] < 30, np.nan, df['temperature'])
                self.signal_add_infinite_line('temperature', "physical outlier lower limit", 30)
                self.signal_add_infinite_line('temperature', "physical outlier upper limit", 40)

            # pamean
            if ('pamean' in df.columns):
                df['pamean'] = np.where(df['pamean'] > 98, np.nan, df['pamean'])
                df['pamean'] = np.where(df['pamean'] < 1, np.nan, df['pamean'])
                self.signal_add_infinite_line('pamean', "physical outlier lower limit", 1)
                self.signal_add_infinite_line('pamean', "physical outlier upper limit", 98)

            # pasystolic
            if ('pasystolic' in df.columns):
                df['pasystolic'] = np.where(df['pasystolic'] > 140, np.nan, df['pasystolic'])
                df['pasystolic'] = np.where(df['pasystolic'] < 5, np.nan, df['pasystolic'])
                self.signal_add_infinite_line('pasystolic', "physical outlier lower limit", 5)
                self.signal_add_infinite_line('pasystolic', "physical outlier upper limit", 140)

            # padiastolic
            if ('padiastolic' in df.columns):
                df['padiastolic'] = np.where(df['padiastolic'] > 93, np.nan, df['padiastolic'])
                df['padiastolic'] = np.where(df['padiastolic'] < 1, np.nan, df['padiastolic'])
                self.signal_add_infinite_line('padiastolic', "physical outlier lower limit", 1)
                self.signal_add_infinite_line('padiastolic', "physical outlier upper limit", 93)

            # systemicmean
            if ('systemicmean' in df.columns):
                df['systemicmean'] = np.where(df['systemicmean'] > 174, np.nan, df['systemicmean'])
                df['systemicmean'] = np.where(df['systemicmean'] < 2, np.nan, df['systemicmean'])
                self.signal_add_infinite_line('systemicmean', "physical outlier lower limit", 2)
                self.signal_add_infinite_line('systemicmean', "physical outlier upper limit", 174)

            # systemicdiastolic
            if ('systemicdiastolic' in df.columns):
                df['systemicdiastolic'] = np.where(df['systemicdiastolic'] > 138, np.nan, df['systemicdiastolic'])
                df['systemicdiastolic'] = np.where(df['systemicdiastolic'] < 24, np.nan, df['systemicdiastolic'])
                self.signal_add_infinite_line('systemicdiastolic', "physical outlier lower limit", 24)
                self.signal_add_infinite_line('systemicdiastolic', "physical outlier upper limit", 138)

            # systemicsystolic
            if ('systemicsystolic' in df.columns):
                df['systemicsystolic'] = np.where(df['systemicsystolic'] > 222, np.nan, df['systemicsystolic'])
                df['systemicsystolic'] = np.where(df['systemicsystolic'] < 47, np.nan, df['systemicsystolic'])
                self.signal_add_infinite_line('systemicsystolic', "physical outlier lower limit", 47)
                self.signal_add_infinite_line('systemicsystolic', "physical outlier upper limit", 222)

            # respiration
            if ('respiration' in df.columns):
                df['respiration'] = np.where(df['respiration'] > 52, np.nan, df['respiration'])
                df['respiration'] = np.where(df['respiration'] < 4, np.nan, df['respiration'])
                self.signal_add_infinite_line('respiration', "physical outlier lower limit", 4)
                self.signal_add_infinite_line('respiration', "physical outlier upper limit", 52)

            # heartrate
            if ('heartrate' in df.columns):
                df['heartrate'] = np.where(df['heartrate'] > 200, np.nan, df['heartrate'])
                df['heartrate'] = np.where(df['heartrate'] < 30, np.nan, df['heartrate'])
                self.signal_add_infinite_line('heartrate', "physical outlier lower limit", 30)
                self.signal_add_infinite_line('heartrate', "physical outlier upper limit", 200)

            # sao2
            if ('sao2' in df.columns):
                df['sao2'] = np.where(df['sao2'] > 100, np.nan, df['sao2'])
                df['sao2'] = np.where(df['sao2'] < 20.5, np.nan, df['sao2'])
                self.signal_add_infinite_line('sao2', "physical outlier lower limit", 20.5)
                self.signal_add_infinite_line('sao2', "physical outlier upper limit", 100)

        df = df.replace('.', np.nan)
        df = df.replace('', np.nan)

        time_column_name = df.columns[0]
        df_with_nan = copy.deepcopy(df)
        df = df.dropna(axis=0, how="any")  # delete rows if ANY nan
        df = df.sort_values(by=[time_column_name])  # sort by offset
        df_without_offset = df.drop(columns=[time_column_name])
        df_without_offset = df_without_offset.dropna(axis=0, how="all")  # did not exist in org algo

        # Tabelle, der Cosinus Distanz zwischen jedem Punktepaar
        distance_table = np.array([[np.nan for _ in range(self.windowlength)] for _ in range(self.windowlength)])

        # Die Punkte, die im ersten betrachteten sliding window sind, nicht normalisiert
        sliding_window_unnormalized = df_without_offset.head(self.windowlength).to_numpy()

        '''
        Alle Parameter werden in eine samples-Matrix eingefügt und z-normalisiert. Dies hat zur
        Folge, dass alle Parameter gleich gewichtet werden. Besonders wenn sich die Werte,
        wie etwa Körperkerntemperatur und Herzfrequenz, stark in der Höhe ihrer Messwerte
        und den absoluten Differenzen der Schwankungen unterscheiden, ist dies wichtig um
        Verzerrungen zu verhindern.
        '''
        sliding_window_normalized = copy.deepcopy(sliding_window_unnormalized)
        for column_number in range(np.size(sliding_window_normalized, 1)):
            zscore_column = stats.zscore(sliding_window_normalized[:, column_number],
                                         nan_policy='omit')  # was: propagate
            zscore_column = np.nan_to_num(zscore_column)
            sliding_window_normalized[:, column_number] = zscore_column

        # Dimensionen jedes Datenpunktes in samples_normalized
        self.num_data_dimensions = len(df_without_offset.columns)

        # Tabelle mit den Datenpunkten. Nicht relevante subspaces werden durch nan-Werte ersetzt
        relevant_subspaces = np.array(
            [[0.0 for _ in range(self.num_data_dimensions)] for _ in range(self.windowlength)])

        # gibt für jeden Punkt(Zeile) an, ob ein anderer Punkt(Index der Spalte) ein K-nearest neighbor
        kNN = np.array([[0 for _ in range(self.windowlength)] for _ in range(self.windowlength)])

        # Tabelle, die für jeden Punkt(Zeile) nach similarity absteigend geordnet den Index des Punktes
        # und des zugehörigen values angibt
        kNNDist = np.array([[(0.0, 0.0) for _ in range(self.windowlength)] for _ in range(self.windowlength)])

        # Tabelle gibt für jedes Knotenpaar die Anzahl an K-nearest neighbors an.
        sNN = np.array([[np.nan for _ in range(self.windowlength)] for _ in range(self.windowlength)])

        # Übersicht über das LOS aller Punkte
        LOS_complete = np.array([0.0 for _ in range(len(df_without_offset) + 10)])

        # Übersicht über die outlier Dimensonen aller Punkte und nicht nur der im aktuellen sliding window
        LOSsubsortges = np.array(
            [[np.nan for _ in range(self.num_data_dimensions)] for _ in range(len(df_without_offset) + 10)])

        # Control limt, anhand dessen entschieden wird, ob ein Datenpunkt ein zu hohes LOS hat.
        cl_complete_table = np.array([0.0 for _ in range(len(df_without_offset))])

        LOS_window = np.array([0.0 for j in range(self.windowlength)])

        # Tabelle die für jeden Punkt speichert, ob er ein outlier ist.
        outlier_table = np.array([np.nan for _ in range(len(df_without_offset.index) + 1)])

        # Confidenz Intervall Grenze für die Ermittlung des Control Limit
        diag_matrix = np.zeros((self.num_data_dimensions, self.num_data_dimensions), int)
        np.fill_diagonal(diag_matrix, 1)

        number_outlier_total = 0
        number_outlier_series = 0
        number_outlier_treshold = 5
        next_position = 0

        '''
        Die Bestimmung des Referenzdatensatzes jedes Punktes erfolgt mithilfe eines Shared
        Nearest Neighbor (SNN) Algorithmus. Der Referenzdatensatz soll für einen Punkt
        die Datenpunkte enthalten, die ebenfalls im ersten Sliding Window liegen und ihm
        besonders ähnlich sind. Dazu wird zunächst die Distanz zwischen allen Punkten
        paarweise berechnet und in einer Matrix gespeichert.
        '''
        for current_point in range(self.windowlength):
            for j in range(self.windowlength):
                distance_table[current_point, j] = self.get_cosine_similarity(sliding_window_normalized[current_point],
                                                                              sliding_window_normalized[j])

        '''
        Auf Basis dieser einfachen Distanzberechnung werden für
        jeden Datenpunkt seine k nächsten Nachbarn bestimmt. Dazu werden Punktindizes
        nach abnehmender Distanz zu dem Datenpunkt geordnet und die k ersten Indizes
        dieser Liste gewählt. Um möglichst einfach auf diese Werte zugreifen zu können
        und später auch einfach Änderungen für einzelne Punkte vornehmen zu können,
        wird in einer KNN-Matrix für jeden Punkt die Beziehung zu jedem anderen Punkt
        mit 1, für k nächster Nachbar, oder 0 kodiert. Zudem wird für jeden Datenpunkt
        eine Rangfolge seiner Nachbarn mit der zugehörigen Distanz als KNNDist-Matrix
        gespeichert.
        '''
        for current_point in range(self.windowlength):
            for j in range(self.windowlength):
                kNNDist[current_point, j] = (np.argsort(distance_table[current_point])[self.windowlength - (j + 1)],
                                             np.sort(distance_table[current_point])[self.windowlength - (j + 1)])
            for k in np.argsort(distance_table[current_point])[-self.k_nearest_neighbor:]:
                kNN[current_point, k] = 1

        '''
        Mit Hilfe der Kodierung der k nächsten Nachbarn kann die Nummer
        der geteilten Nachbarn für jedes Punktepaar als Summe über die Verundung der
        jeweiligen Zeilen in der KNN-Matrix bestimmt und in einer SNN-Matrix gespeichert
        werden. Eine Shared Nearest Neighbor Nummer kann definiert werden, die angibt
        wie viele gemeinsame k nächsten Nachbarn zwei Punkte haben müssen, um zum
        Referenzdatensatz des jeweils anderen Datenpunkts gerechnet zu werden.
        '''
        for current_point in range(self.windowlength):
            for j in range(self.windowlength):
                is_in_knn_bool = np.logical_and(kNN[current_point], kNN[j])
                sNN[current_point, j] = sum([int(e) for e in is_in_knn_bool])

        '''
        Im vierten Arbeitsschritt werden die relevanten Subspaces jedes Datenpunkts bestimmt,
        alle anderen müssen nicht bei der Ausreißererkennung betrachtet werden.
        Eine relevante Subspace ist eine Dimension, die sich stark von den anderen Werten
        dieser Dimension der Punkte im Referenzdatensatz, unterscheidet. Zur Bestimmung
        der relevanten Dimensionen wird zunächst der Vektor als Differenz zwischen dem
        Datenvektor des aktuell betrachteten Datenpunkts und dem Mittelwert des Referenzdatensatzes
        berechnet. Zur Bestimmung von relevanten Dimensionen wird der
        Cosinus zwischen dem Vektor und dem Vektor der zugehörigen Achse betrachtet.
        Ist dieser Wert nahe 0, kann angenommen werden, dass in dieser Dimension der
        Unterschied zwischen dem Punkt und seinem Referenzdatensatz gering ist und die
        Dimension wahrscheinlich keine Abweichung zu den anderen Punkten aufweist. Ist
        der Wert hingegen dem Wert 1 nahe, liegt eine signifikante Abweichung vor und
        die Dimension sollte auf jeden Fall weiterhin berücksichtigt werden.
        '''
        for current_point in range(self.windowlength):
            relevant_points = self.get_relevant_points(snn=sNN[current_point], data=sliding_window_normalized)

            mean_vector = relevant_points.mean(axis=0)
            current_vector = sliding_window_normalized[current_point]

            subspace = self.calculate_subspace(current_vector, mean_vector)

            relevant_subspace_col = np.vstack([sliding_window_normalized[current_point], subspace])
            relevant_subspace_col[0, :][relevant_subspace_col[1, :] == 0] = np.nan
            relevant_subspace_col = np.delete(relevant_subspace_col, 1, 0)
            relevant_subspace_col = np.squeeze(relevant_subspace_col, axis=0)
            relevant_subspaces[current_point] = relevant_subspace_col

        matrix_subspace = np.copy(relevant_subspaces)
        matrix_subspace[~np.isnan(matrix_subspace)] = 1
        matrix_subspace = np.nan_to_num(matrix_subspace)

        sample_counter = 0

        while (sample_counter < self.windowlength):

            relevant_points = self.get_relevant_points(snn=sNN[sample_counter], data=sliding_window_normalized)

            '''
            Die Berechnung des Local Outlier Scores (LOS) erfolgt durch die Bestimmung
            der normalisierten Mahalanobisdistanz zu den Punkten im Referenzdatensatz, unter
            Berücksichtigung der relevanten Subspaces.
            '''
            if (np.all(matrix_subspace[sample_counter] == 0)):
                '''
                Falls keine Dimension des Datenpunktes
                als relevant betrachtet wird, kann der LOS direkt auf 0 gesetzt werden, da
                keine Dimension von den Werten des Referenzdatensatzes abweicht und somit der
                Datenpunkt auch kein Ausreißer sein kann.
                '''
                LOS_complete[sample_counter] = 0  # all subspaces irrelevant!
                LOS_window[sample_counter] = 0
            else:
                '''
                Der LOS berechnet sich als normierte Mahalanobisdistanz zwischen den 
                relevanten Dimensionen des Datenpunkts, die nicht NaN-Werte der Matrix relevantsub, 
                und diesen Dimensionen des Mittelwertvektors seines Referenzdatensatzes.
                '''
                ds = self.calculate_mahalanobis(points=relevant_points, subspaces=relevant_subspaces[sample_counter])
                LOS_complete[sample_counter] = ds
                LOS_window[sample_counter] = ds

            # LOS für jede Subspace
            mean_vector = relevant_points.mean(axis=0)
            point_vector = sliding_window_normalized[sample_counter]

            LOSsubsort_point = LOS_complete[sample_counter] * self.calculate_los(point_vector, mean_vector)
            LOSsubsortges[sample_counter] = np.argsort(LOSsubsort_point)

            # Iteration step
            sample_counter = sample_counter + 1

        '''
        Im letzten Schritt erfolgt die Bestimmung des Control limits (CL), das den Wert des
        LOS angibt, anhand dessen Ausreißer von normalen Punkten unterschieden werden.
        '''
        CL = self.calculate_cl(LOS_window)

        for current_point in range(self.windowlength):
            cl_complete_table[current_point] = CL

        while (sample_counter < len(df_without_offset)):
            '''
            Im Anschluss an die Trainingsphase wird jeweils ein neuer Punkt zum Sliding
            Window hinzugefügt und als möglicher Ausreißer getestet. Dieser Punkt ersetzt den
            frühesten Wert im jeweiligen Sliding Window in der samples-Matrix.
            '''
            self.signal_percentage(int((sample_counter / len(df_without_offset)) * 100))

            _next_position_save = next_position
            _samples_normalized_save = copy.deepcopy(sliding_window_normalized)
            _distance_table_save = copy.deepcopy(distance_table)

            sNNnew = copy.deepcopy(sNN)
            _kNN_save = copy.deepcopy(kNN)
            _kNNist_save = copy.deepcopy(kNNDist)
            _sNNnew_save = copy.deepcopy(sNNnew)
            _relevant_subspaces_save = copy.deepcopy(relevant_subspaces)
            _matrix_subspace_save = copy.deepcopy(matrix_subspace)
            _los_window_save = copy.deepcopy(LOS_window)

            new_point = df_without_offset.iloc[sample_counter]
            sliding_window_unnormalized[next_position] = new_point

            current_point = next_position
            next_position = (next_position + 1) % self.windowlength

            '''
            Die Normalisierung wurde in der Implementierung so umgesetzt, dass für alle Punkte,
            die im aktuellen Sliding Window liegen, die ursprünglichen unnormalisierten Werte
            gewählt werden und diese spaltenweise z-normalisiert werden.
            '''
            sliding_window_normalized = copy.deepcopy(sliding_window_unnormalized)
            for column_number in range(np.size(sliding_window_normalized, 1)):
                zscore_column = stats.zscore(sliding_window_normalized[:, column_number],
                                             nan_policy='omit')  # was: propagate
                zscore_column = np.nan_to_num(zscore_column)
                sliding_window_normalized[:, column_number] = zscore_column

            '''
            Zur Bestimmung des Referenzdatensatzes muss die Distanztabelle geupdated werden
            '''
            for j in range(self.windowlength):
                distance_table[current_point, j] = self.get_cosine_similarity(sliding_window_normalized[current_point],
                                                                              sliding_window_normalized[j])

            '''
            Zur Bestimmung des Referenzdatensatzes muss die KNN Tabelle geupdated werden. Die
            KNN-Matrix kann für jeden Punkt j, abhängig von dem Wert des neu hinzugefügten
            Punktes und des aus dem Sliding Window herausgenommenen Punktes, angepasst
            werden.
            '''
            for l in range(self.windowlength):
                if (kNN[l, current_point] == 1):
                    next_neighbor = kNNDist[l, self.k_nearest_neighbor]
                    if (distance_table[l][current_point] <= next_neighbor[1]):
                        kNN[l, int(next_neighbor[0])] = 1
                        kNN[l, current_point] = 0

                else:
                    k_neighbor = kNNDist[l, self.k_nearest_neighbor]
                    if (distance_table[l, current_point] > k_neighbor[1]):
                        kNN[l, current_point] = 1
                        kNN[l, int(k_neighbor[0])] = 0

                kNNDist_without_current_point = [i for i in kNNDist[l] if i[0] != current_point]

                _keys = [r[1] for r in kNNDist_without_current_point]
                _index = bisect.bisect(_keys, distance_table[l, current_point])

                kNNDist_without_current_point.insert(_index, (current_point, distance_table[l, current_point]))
                kNNDist[l] = copy.deepcopy(kNNDist_without_current_point)

            '''
            Zur Bestimmung des Referenzdatensatzes muss die SNN Tabelle geupdated werden
            '''
            for j in range(self.windowlength):
                is_in_knn_bool = np.logical_and(kNN[current_point], kNN[j])
                sNNnew[current_point, j] = sum([int(e) for e in is_in_knn_bool])

            relevant_points = self.get_relevant_points(snn=sNN[current_point], data=sliding_window_normalized)

            mean_vector = relevant_points.mean(axis=0)
            point_vector = sliding_window_normalized[current_point]
            subspace = self.calculate_subspace(point_vector, mean_vector)

            relevant_subspace_col = np.vstack([sliding_window_normalized[current_point], subspace])
            relevant_subspace_col[0, :][relevant_subspace_col[1, :] == 0] = np.nan
            relevant_subspace_col = np.delete(relevant_subspace_col, 1, 0)
            relevant_subspace_col = np.squeeze(relevant_subspace_col, axis=0)

            relevant_subspaces[current_point] = relevant_subspace_col

            matrix_subspace = copy.deepcopy(relevant_subspaces)
            matrix_subspace[~np.isnan(matrix_subspace)] = 1  # new
            matrix_subspace = np.nan_to_num(matrix_subspace)  # replaces Nan with 0

            # if no subspace is relevant
            if (np.all(matrix_subspace[current_point] == 0)):
                LOS_complete[sample_counter] = 0
                LOS_window[current_point] = 0
            else:
                ds = self.calculate_mahalanobis(points=relevant_points, subspaces=relevant_subspaces[current_point])
                LOS_complete[sample_counter] = ds
                LOS_window[current_point] = ds

            LOSsubsort_point = LOS_complete[sample_counter] * self.calculate_los(point_vector, mean_vector)
            LOSsubsortges[sample_counter] = np.argsort(LOSsubsort_point)

            # Check if outlier
            if (LOS_complete[sample_counter] > CL):
                self.log('Outlier detected')
                outlier_table[sample_counter] = 1
            else:
                number_outlier_series = 0
                outlier_table[sample_counter] = 0

            # Update CL if needed
            if (LOS_complete[sample_counter] > 0):
                if (self.use_cl_modification):
                    if (sample_counter < self.windowlength):
                        CL = self.calculate_cl(LOS_complete[0:sample_counter + 1])
                    else:
                        CL = self.calculate_cl(LOS_complete[sample_counter - self.windowlength:sample_counter + 1])
                else:
                    if (LOS_complete[sample_counter] <= CL):
                        CL = self.calculate_cl(LOS_window)

            cl_complete_table[sample_counter] = CL

            # if the point was an outlier, reset everything
            if (outlier_table[sample_counter] == 1):

                next_position = _next_position_save
                sliding_window_normalized = copy.deepcopy(_samples_normalized_save)
                distance_table = copy.deepcopy(_distance_table_save)
                kNN = copy.deepcopy(_kNN_save)
                kNNDist = copy.deepcopy(_kNNist_save)
                relevant_subspaces = copy.deepcopy(_relevant_subspaces_save)
                matrix_subspace = copy.deepcopy(_matrix_subspace_save)
                LOS_window = copy.deepcopy(_los_window_save)

                number_outlier_total = number_outlier_total + 1
                number_outlier_series = number_outlier_series + 1

                if (number_outlier_series > number_outlier_treshold):
                    self.log('Multiple consecutive outlier detected (Treshold: %s)' % str(number_outlier_treshold))

            # Iteration step
            sample_counter = sample_counter + 1

        self.log('Outlier number: ' + str(number_outlier_total))
        self.log('Windowlength: ' + str(self.windowlength))
        self.log('Nearest neighbor number: ' + str(self.k_nearest_neighbor))
        self.log('SNN number: ' + str(self.shared_nearest_neighbor))
        self.log('Confidence Soll: ' + str(self.confidence_soll))
        self.log('Bandwidth: ' + str(self.bandWidth))
        self.log('Theta: ' + str(self.theta))
        self.log('Nullen bereinigt: ' + str(self.replace_zeroes))
        self.log('Phys. Outlier bereinigt: ' + str(self.replace_phys_outlier))

        df['outlier_table'] = outlier_table[:-1]
        df['relevantcol'] = (LOSsubsortges[:, self.num_data_dimensions - 1])[:-10]
        df['LOS'] = LOS_complete[:-10]
        df['controllimit'] = cl_complete_table

        result = {}
        for column_number in range(1, len(df.columns)):  # for every column
            column_name = df.columns[column_number]

            result[column_name] = {}
            outlier_for_column = {}
            _counter = 0

            for index, row in df.iterrows():  # store time => outlier in extra table
                outlier_for_column[row[0]] = row['outlier_table']

            # for every time value
            for key, value in outlier_for_column.items():

                if _counter < self.windowlength:
                    outlier_for_column[key] = -1
                elif (np.isnan(outlier_for_column[key]) or outlier_for_column[key] == 0):
                    # not a outlier!
                    outlier_for_column[key] = 0
                elif (outlier_for_column[key] == 1 and LOSsubsortges[
                    _counter, self.num_data_dimensions - 1] == column_number - 1):
                    # level 1 outlier!
                    outlier_for_column[key] = 1
                else:
                    # level 2 outlier!
                    outlier_for_column[key] = 2

                _counter = _counter + 1

            for index, row in df_with_nan.iterrows():
                if column_name in df_with_nan.columns:
                    if np.isnan(row[column_name]):
                        outlier_for_column[row[0]] = 1

            result[column_name] = outlier_for_column

        self.signal_add_plot("LOS", df[df.columns[0]], df['LOS'], df.columns[0], "LOS/CL")
        self.signal_add_line("LOS", "CL", df[df.columns[0]], df['controllimit'])

        return result
