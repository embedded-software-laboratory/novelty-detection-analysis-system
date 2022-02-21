from PyQt5.QtWidgets import *
import random
from ndas.utils import datagenerator


class DataInputForm:
    """
    Input form for data generator
    """

    def __init__(self, did, labeling_option=True, quota=True):
        """
        Creates gui elements and layout

        Parameters
        ----------
        did
        labeling_option
        quota
        """
        self.single_dimension_layout = QVBoxLayout()
        self.novelty_generators = []

        first_row_layout = QHBoxLayout()
        data_name_label = QLabel("Name:")
        self.data_name = QLineEdit("test-data-" + str(did))
        first_row_layout.addWidget(data_name_label)
        first_row_layout.addWidget(self.data_name)

        data_distribution_label = QLabel("Distribution:")
        self.data_distribution = QComboBox()
        self.data_distribution.setMinimumWidth(150)
        self.data_distribution.addItem("Gaussian", datagenerator.DataDistributionType.GAUSSIAN)
        self.data_distribution.addItem("Uniform", datagenerator.DataDistributionType.UNIFORM)
        self.data_distribution.addItem("Laplace", datagenerator.DataDistributionType.LAPLACE)
        self.data_distribution.setCurrentIndex(random.randint(0, 2))
        first_row_layout.addWidget(data_distribution_label)
        first_row_layout.addWidget(self.data_distribution)

        data_flow_label = QLabel("Flow:")
        self.data_flow = QComboBox()
        self.data_flow.setMinimumWidth(150)
        self.data_flow.addItem("None", datagenerator.DataFlowType.NONE)
        self.data_flow.addItem("Rising", datagenerator.DataFlowType.RISING)
        self.data_flow.addItem("Falling", datagenerator.DataFlowType.FALLING)
        self.data_flow.setCurrentIndex(random.randint(0, 2))
        first_row_layout.addWidget(data_flow_label)
        first_row_layout.addWidget(self.data_flow)

        data_mean_label = QLabel("μ:")
        self.data_mean = QDoubleSpinBox()
        self.data_mean.setValue(datagenerator.get_random_int(20, 110))
        self.data_mean.setSingleStep(0.1)
        self.data_mean.setMinimum(0)
        self.data_mean.setMaximum(1000)
        self.data_mean.setMinimumWidth(100)
        first_row_layout.addWidget(data_mean_label)
        first_row_layout.addWidget(self.data_mean)

        data_std_label = QLabel("σ:")
        self.data_std = QDoubleSpinBox()
        self.data_std.setValue(datagenerator.get_random_double(0.1, 0.8))
        self.data_std.setSingleStep(0.01)
        self.data_std.setMinimum(0)
        self.data_std.setMaximum(100)
        self.data_std.setMinimumWidth(100)
        first_row_layout.addWidget(data_std_label)
        first_row_layout.addWidget(self.data_std)

        second_row_layout = QHBoxLayout()

        single_novelties_groupbox = self.PointNoveltyGeneratorForm(labeling=labeling_option, quota=quota)
        collective_novelties_groupbox = self.CollectiveNoveltyGeneratorForm(labeling=labeling_option, quota=quota)
        missing_data_condition_change_groupbox = self.MissingDataConditionChangeNoveltyGeneratorForm(labeling=False)

        self.novelty_generators.append(single_novelties_groupbox)
        self.novelty_generators.append(collective_novelties_groupbox)
        self.novelty_generators.append(missing_data_condition_change_groupbox)

        second_row_layout.addWidget(single_novelties_groupbox.get_groupbox())
        second_row_layout.addWidget(collective_novelties_groupbox.get_groupbox())
        second_row_layout.addWidget(missing_data_condition_change_groupbox.get_groupbox())
        spac = QSpacerItem(20, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        second_row_layout.addItem(spac)

        self.single_dimension_layout.addLayout(first_row_layout)
        self.single_dimension_layout.addLayout(second_row_layout)

    def get_mu(self) -> float:
        """
        Returns the mean
        """
        return float(self.data_mean.value())

    def get_sigma(self) -> float:
        """
        Returns the std
        """
        return float(self.data_std.value())

    def get_name(self) -> str:
        """
        Returns the name
        """
        return str(self.data_name.text())

    def get_flow(self) -> any:
        """
        Returns the flow direction
        """
        return self.data_flow.currentData()

    def get_distribution(self) -> any:
        """
        Returns the statistical distribution
        """
        return self.data_distribution.currentData()

    def get_layout(self):
        """
        Returns the layout of this form
        """
        return self.single_dimension_layout

    class NoveltyGeneratorForm:
        """
        Form to define novelty generator parameters
        """

        def __init__(self, name, boundaries=True, quota=True, labeling=True):
            """
            Creates elements and layout for this form

            Parameters
            ----------
            name
            boundaries
            quota
            labeling
            """
            self.groupbox = QGroupBox(name, checkable=True, checked=bool(random.getrandbits(1)))

            layout = QVBoxLayout()
            self.groupbox.setLayout(layout)
            anomaly_option_layout = QHBoxLayout()

            if boundaries:
                range_layout = QHBoxLayout()
                range_label = QLabel("Boundaries:")
                self.start_range = QSpinBox()
                self.start_range.setMaximumWidth(68)
                self.start_range.setValue(15)
                self.start_range.setSingleStep(1)
                self.start_range.setMinimum(0)
                self.start_range.setMaximum(100)
                self.start_range.setSuffix("%")
                range_sep = QLabel("-")
                range_sep.setMaximumWidth(5)
                self.end_range = QSpinBox()
                self.end_range.setMaximumWidth(68)
                self.end_range.setValue(80)
                self.end_range.setSingleStep(1)
                self.end_range.setMinimum(0)
                self.end_range.setMaximum(100)
                self.end_range.setSuffix("%")

                range_layout.addWidget(range_label)
                range_layout.addWidget(self.start_range)
                range_layout.addWidget(range_sep)
                range_layout.addWidget(self.end_range)
                anomaly_option_layout.addLayout(range_layout)

            if quota:
                quota_layout = QHBoxLayout()
                quota_label = QLabel("Quota:")
                self.quota = QDoubleSpinBox()
                self.quota.setMaximumWidth(60)
                self.quota.setSingleStep(0.1)
                self.quota.setMinimum(0.1)
                self.quota.setMaximum(20)
                self.quota.setSuffix("%")
                quota_layout.addWidget(quota_label)
                quota_layout.addWidget(self.quota)
                anomaly_option_layout.addLayout(quota_layout)

            if labeling:
                auto_label_layout = QHBoxLayout()
                self.auto_label_checkbox = QCheckBox("Auto-Label")
                self.auto_label_checkbox.setChecked(True)
                auto_label_layout.addWidget(self.auto_label_checkbox)
                anomaly_option_layout.addLayout(auto_label_layout)

            layout.addLayout(anomaly_option_layout)

        def get_groupbox(self):
            """
            Returns the groupbox
            """
            return self.groupbox

        def get_groupbox_state(self):
            """
            Returns the state of the groupbox
            """
            return self.groupbox.isChecked()

        def get_range_start(self) -> float:
            """
            Returns the start range
            """
            return float(self.start_range.cleanText().replace(",", ".")) / 100

        def get_range_end(self) -> float:
            """
            Returns the end range
            """
            return float(self.end_range.cleanText().replace(",", ".")) / 100

        def get_quota(self) -> float:
            """
            Returns the quota
            """
            return float(self.quota.cleanText().replace(",", ".")) / 100

        def get_label_state(self) -> bool:
            """
            Returns if this novelty is enabled

            Returns
            -------

            """
            return bool(self.auto_label_checkbox.isChecked())

    class PointNoveltyGeneratorForm(NoveltyGeneratorForm):
        """
        Generator for point anomalies
        """

        def __init__(self, *args, **kwargs):
            self.name = "Point Anomalies"
            super().__init__(self.name, *args, **kwargs)

            self.quota.setValue(datagenerator.get_random_double(0.1, 3.0, 1))

    class CollectiveNoveltyGeneratorForm(NoveltyGeneratorForm):
        """
        Generator for collective anomalies
        """

        def __init__(self, *args, **kwargs):
            self.name = "Collective Anomalies"
            super().__init__(self.name, *args, **kwargs)

            self.quota.setValue(datagenerator.get_random_double(5.0, 10.0, 1))

    class MissingDataConditionChangeNoveltyGeneratorForm(NoveltyGeneratorForm):
        """
        Generator for temporal gaps
        """

        def __init__(self, *args, **kwargs):
            self.name = "Gap Condition Change"
            super().__init__(self.name, quota=False, *args, **kwargs)

        def get_label_state(self) -> bool:
            return False
