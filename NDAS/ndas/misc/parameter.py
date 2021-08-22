from enum import Enum, auto


class AdditionalParameter:
    """
    Additional parameter for algorithms
    """

    def __init__(self, argument_name: str):
        """
        Defines default parameters

        Parameters
        ----------
        argument_name
        """
        self.argument_name = argument_name
        self.type = ArgumentType.STRING
        self.default = ""
        self.tooltip = None

    def set_type(self, parameter_type: 'ArgumentType'):
        """
        Sets the type of the additional parameter

        Parameters
        ----------
        parameter_type
        """
        self.type = parameter_type

    def set_default(self, value: any):
        """
        Sets the default value for the additional parameter

        Parameters
        ----------
        value
        """
        if self.type == ArgumentType.STRING:
            self.default = str(value)

        if self.type == ArgumentType.BOOL:
            self.default = bool(value)

        if self.type == ArgumentType.INTEGER:
            self.default = int(value)

        if self.type == ArgumentType.FLOAT:
            self.default = float(value)

    def set_tooltip(self, tip: str):
        """
        Sets the tooltip of the additional parameter

        Parameters
        ----------
        tip
        """
        self.tooltip = tip


class AdditionalNumberParameter(AdditionalParameter):
    """
    Additional parameter with type number
    """

    def __init__(self, *args, **kwargs):
        """
        Sets default min and max values

        Parameters
        ----------
        args
        kwargs
        """
        super(AdditionalNumberParameter, self).__init__(*args, **kwargs)
        self.minimum = -999999
        self.maximum = 999999

    def set_maximum(self, parameter_max):
        """
        Sets the maximum number

        Parameters
        ----------
        parameter_max
        """
        self.maximum = parameter_max

    def set_minimum(self, parameter_min):
        """
        Sets the minimum number

        Parameters
        ----------
        parameter_min
        """
        self.minimum = parameter_min


class ArgumentType(Enum):
    """
    Enum for the argument types
    """
    BOOL = auto()
    STRING = auto()
    INTEGER = auto()
    FLOAT = auto()
