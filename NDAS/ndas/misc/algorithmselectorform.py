from PyQt5.QtWidgets import *

from ndas.extensions import algorithms
from ndas.misc import parameter


class AlgorithmInputForm:
    """
    The input form for algorithm selection
    """

    def __init__(self, aid):
        """
        Creates the widgets and layouts

        Parameters
        ----------
        aid
        """
        self.single_algorithm_layout = QVBoxLayout()
        self.algorithm_generators = []
        self.algorithm_additional_parameters = []
        self.algorithm_additional_options_layout = QHBoxLayout()

        first_row_layout = QHBoxLayout()

        active_analysis_algorithm_layout = QHBoxLayout()
        active_analysis_algorithm_label = QLabel("Active Algorithm:")
        self.active_analysis_algorithm = QComboBox()
        active_analysis_algorithm_layout.addWidget(active_analysis_algorithm_label)
        active_analysis_algorithm_layout.addWidget(self.active_analysis_algorithm)
        self.active_analysis_algorithm.currentIndexChanged.connect(lambda index: self.on_algorithm_change(index))

        algorithm_name_label = QLabel("Name:")
        self.algorithm_name = QLineEdit("auto:algo-" + str(aid))
        self.algorithm_name.setMaximumWidth(550)

        algorithm_list = algorithms.get_available_algorithms()
        for algorithm in algorithm_list:
            self.active_analysis_algorithm.addItem(algorithm)

        first_row_layout.addLayout(active_analysis_algorithm_layout)
        first_row_layout.addWidget(algorithm_name_label)
        first_row_layout.addWidget(self.algorithm_name)

        first_row_spacer = QSpacerItem(20, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        first_row_layout.addItem(first_row_spacer)

        second_row_layout = QHBoxLayout()
        second_row_layout.addLayout(self.algorithm_additional_options_layout)

        second_row_spacer = QSpacerItem(20, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
        second_row_layout.addItem(second_row_spacer)

        self.single_algorithm_layout.addLayout(first_row_layout)
        self.single_algorithm_layout.addLayout(second_row_layout)

    def get_conf(self):
        """
        Loads the configuration for algorithms
        """
        klass = algorithms.get_available_algorithms()[self.active_analysis_algorithm.currentIndex()]
        name = self.algorithm_name.text()
        if name.startswith("auto:"):
            name = name[5:]
        algorithm_parameter = self.get_param_list()
        return {'klass': klass, 'name': name, 'parameter': algorithm_parameter}

    def get_name(self) -> str:
        """
        Returns the name of the algorithm
        """
        name = self.algorithm_name.text()
        if name.startswith("auto:"):
            name = name[5:]
        return str(name)

    def get_layout(self):
        """
        Returns the layout of this form
        """
        return self.single_algorithm_layout

    def delete_items_of_layout(self, layout):
        """
        Removes elements of this form

        Parameters
        ----------
        layout
        """
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                else:
                    self.delete_items_of_layout(item.layout())

    def on_algorithm_change(self, index):
        """
        Adds elements based on algorithm selection to this form

        Parameters
        ----------
        index
        """
        for param in self.algorithm_additional_parameters:
            for i in range(self.algorithm_additional_options_layout.count()):
                layout_item = self.algorithm_additional_options_layout.itemAt(i)
                if layout_item.layout() == param:
                    self.delete_items_of_layout(layout_item.layout())
                    self.algorithm_additional_options_layout.removeItem(layout_item)
                    break

        additional_parameters = algorithms.get_algorithm_required_arguments(
            algorithms.get_available_algorithms()[index])
        q_layout = QHBoxLayout()

        for arg in additional_parameters:
            q_groupbox = QGroupBox()
            q_groupbox_layout = QHBoxLayout()
            q_groupbox.setLayout(q_groupbox_layout)

            q_label = QLabel(arg.argument_name + " = ", parent=q_groupbox)

            if arg.type == parameter.ArgumentType.INTEGER:
                q_input = QSpinBox(parent=q_groupbox)
                q_input.setMinimum(arg.minimum)
                q_input.setMaximum(arg.maximum)
                q_input.setValue(arg.default)
                q_input.valueChanged.connect(self.update_name_string)
            elif arg.type == parameter.ArgumentType.FLOAT:
                q_input = QDoubleSpinBox(parent=q_groupbox)
                q_input.setDecimals(3)
                q_input.setValue(arg.default)
                q_input.setSingleStep(0.1)
                q_input.setMinimum(arg.minimum)
                q_input.setMaximum(arg.maximum)
                q_input.valueChanged.connect(self.update_name_string)
            elif arg.type == parameter.ArgumentType.BOOL:
                q_input = QCheckBox(parent=q_groupbox)
                q_input.setChecked(arg.default)
                q_input.stateChanged.connect(self.update_name_string)
            else:
                q_input = QLineEdit(arg.default, parent=q_groupbox)
                q_input.textChanged.connect(self.update_name_string)

            if arg.tooltip is not None:
                q_label.setToolTip(arg.tooltip)
                q_input.setToolTip(arg.tooltip)

            q_groupbox_layout.addWidget(q_label)
            q_groupbox_layout.addWidget(q_input)

            q_layout.addWidget(q_groupbox)

        self.algorithm_additional_parameters.append(q_layout)
        self.algorithm_additional_options_layout.addLayout(q_layout)
        self.update_name_string()

    def update_name_string(self):
        cur_alg = self.active_analysis_algorithm.currentText()
        if self.algorithm_name.text().startswith("auto:"):
            if cur_alg:
                alg_string = 'auto:' + ''.join(c for c in cur_alg if c.isupper())
            var_string = "("
            for k, v in self.get_param_list().items():
                k_f = k[:min(3, len(k))]
                k_b = k[min(3, len(k)):]
                if isinstance(v, bool):
                    v_str = str(v)[:1]
                else:
                    v_str = str(v)
                var_string += (k_f+''.join(c for c in k_b if c.isupper())+":"+v_str+"|")
            var_string = var_string[:-1]
            if var_string[:-1]:
                var_string += ")"
            self.algorithm_name.setText(alg_string+" "+var_string)

    def get_param_list(self):
        """
        Returns the list of optional parameters for active algorithm
        """
        args = {}
        for param in self.algorithm_additional_parameters:
            for i in range(self.algorithm_additional_options_layout.count()):
                layout_item = self.algorithm_additional_options_layout.itemAt(i)
                if layout_item.layout() == param:
                    for x in range(0, layout_item.layout().count()):

                        groupbox = layout_item.layout().itemAt(x).widget().children()

                        label_item = groupbox[1]
                        input_item = groupbox[2]

                        label_text = label_item.text().replace(" = ", "")

                        if isinstance(input_item, QSpinBox):
                            input_text = input_item.value()
                        elif isinstance(input_item, QDoubleSpinBox):
                            input_text = input_item.value()
                        elif isinstance(input_item, QCheckBox):
                            input_text = input_item.isChecked()
                        else:
                            input_text = input_item.text().replace(" ", "")

                        args[label_text] = input_text
        return args
