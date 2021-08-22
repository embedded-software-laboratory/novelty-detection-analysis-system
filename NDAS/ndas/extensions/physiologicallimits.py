from ndas.utils import logger

physiological_data_types = {}


def init_physiological_info(config):
    """
    Initalizes the physiological infos from the config file.

    Parameters
    ----------
    config
    """
    for k, v in config.items():
        if add_physiological_dt(k):
            add_physiological_dt_high_limit(k, v["high"])
            add_physiological_dt_low_limit(k, v["low"])
            for alias in v["aliases"]:
                add_physiological_dt_alias(k, alias)


def add_physiological_dt(name: str) -> bool:
    """
    Adds a new physiological data type.

    Parameters
    ----------
    name
    """
    global physiological_data_types
    if name not in physiological_data_types.keys():
        physiological_data_types[name] = PhysiologicalDataType(name)
        logger.physicalinfo.debug("Added physiological data type: %s" % name)
        return True
    else:
        logger.physicalinfo.error("Physiological data type %s does already exist." % name)
        return False


def add_physiological_dt_high_limit(name: str, high: float) -> bool:
    """
    Adds a new high limit to a physiological data type.

    Parameters
    ----------
    name : str
        The name of the physiological data type
    high : float
        The high limit
    """
    global physiological_data_types
    if name not in physiological_data_types.keys():
        logger.physicalinfo.error("Physiological data type not found: %s" % name)
        return False
    else:
        physiological_data_types[name].set_high(high)
        logger.physicalinfo.debug("Added new high limit to physiological data type %s: %s" % (name, str(high)))
        return True


def add_physiological_dt_low_limit(name: str, low: float) -> bool:
    """
    Adds a new low limit to a physiological data type.

    Parameters
    ----------
    name : str
        The name of the physiological data type
    low : float
        The low limit
    """
    global physiological_data_types
    if name not in physiological_data_types.keys():
        logger.physicalinfo.error("Physiological data type not found: %s" % name)
        return False
    else:
        physiological_data_types[name].set_low(low)
        logger.physicalinfo.debug("Added new low limit to physiological data type %s: %s" % (name, str(low)))
        return True


def add_physiological_dt_alias(name: str, alias: str) -> bool:
    """
    Adds a new alias to a physiological data type

    Parameters
    ----------
    name : str
        The name of the physiological data type
    alias : str
        The alias to be added
    """
    global physiological_data_types
    if name not in physiological_data_types.keys():
        logger.physicalinfo.error("Physiological data type not found: %s" % name)
        return False
    else:
        if alias not in physiological_data_types[name].aliases:
            if not is_alias_in_use(alias):
                physiological_data_types[name].add_alias(alias)
                logger.physicalinfo.debug("Added new alias to physiological data type %s: %s" % (name, alias))
                return True
            else:
                logger.physicalinfo.error(
                    "The alias: %s is already in use and could not be added to data type: %s" % (alias, name))
                return False
        return False


def is_alias_in_use(alias: str) -> bool:
    """
    Checks if the alias is already in use in some physiological data type

    Parameters
    ----------
    alias : str
        The alias to check
    """
    global physiological_data_types
    for k, v in physiological_data_types.items():
        if alias in v.aliases:
            return True
    return False


def get_physical_dt(name: str):
    """
    Returns the physiological data type object

    Parameters
    ----------
    name : str
        The identifier of the physiological data type or an alias
    """
    global physiological_data_types
    if name not in physiological_data_types.keys():

        ''' Check aliases as well '''
        for k, v in physiological_data_types.items():
            if name in v.aliases:
                return physiological_data_types[k]

        logger.physicalinfo.error("Physiological data type not found: %s" % name)
        return False
    else:
        return physiological_data_types[name]


class PhysiologicalDataType:
    """
    Data type for physiological data
    """

    def __init__(self, pid):
        self.id = pid
        self.high = None
        self.low = None
        self.aliases = []

    def set_high(self, high: float):
        self.high = high

    def set_low(self, low: float):
        self.low = low

    def add_alias(self, alias: str):
        self.aliases.append(alias)

    def get_aliases(self):
        return self.aliases
