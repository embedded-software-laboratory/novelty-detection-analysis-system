from enum import Enum, auto

import numpy as np


def generate_collective_outliers(data, number, start_pct, end_pct):
    """
    Generates collective outliers

    Parameters
    ----------
    data
    number
    start_pct
    end_pct
    """
    result = {}

    _length = len(data)
    _max = np.max(data)
    _min = np.min(data)
    _std = np.std(data)
    _mean = np.mean(data)

    # gauss: 95% of data is within mean +- 2 std, so don't be in there
    _lower = np.round_(_mean - 2 * _std, decimals=1)
    _higher = np.round_(_mean + 2 * _std, decimals=1)

    n, p = number, 1
    c, r = divmod(n, p)
    bins = [c] * (p - r) + [c + 1] * r

    for single_bin in bins:
        lowest_b = int(start_pct * _length)
        highest_b = int(end_pct * _length)
        index = np.random.randint(lowest_b, highest_b)

        outlier = data[index]

        for i in range(single_bin):
            result[index + i] = outlier

    return result


def generate_condition_change_gap(data, start_pct, end_pct):
    """
    Generates temporal gap outliers

    Parameters
    ----------
    data
    start_pct
    end_pct
    """
    result = {}

    _length = len(data)
    _max = np.max(data)
    _min = np.min(data)
    _std = np.std(data)
    _mean = np.mean(data)

    _lower = np.round_(_mean - 2 * _std, decimals=1)
    _higher = np.round_(_mean + 2 * _std, decimals=1)

    # select random index in bounds:
    lowest_b = int(start_pct * _length)
    highest_b = int(end_pct * _length)
    index = np.random.randint(lowest_b, highest_b)

    # constant for condition change
    c = int(0.1 * _mean)
    gaps = 80

    # create gap for the next 5
    for i in range(gaps):
        result[index + i] = np.nan

    # for the remaining set, add constant for condition change
    for i in range(_length - index - gaps):
        result[index + i + gaps] = data[index + i + gaps] + c

    return result


def generate_point_outliers(data, number, start_pct, end_pct):
    """
    Generates point outliers

    Parameters
    ----------
    data
    number
    start_pct
    end_pct
    """
    result = {}

    _length = len(data)
    _max = np.max(data)
    _min = np.min(data)
    _std = np.std(data)
    _mean = np.mean(data)

    _minimum = np.min(data)  # np.round_(_mean - 2 * _std, decimals=1)
    _maximum = np.max(data)  # np.round_(_mean + 2 * _std, decimals=1)

    for _ in range(number):
        index = np.random.randint(int(start_pct * _length), int(end_pct * _length))
        if np.random.rand() < 0.5:
            modifier = (np.random.rand() ** 2) * _mean
            anomaly = np.round_(_minimum - modifier, decimals=1)
        else:
            modifier = (np.random.rand() ** 2) * _mean
            anomaly = np.round_(_maximum + modifier, decimals=1)
        result[index] = anomaly
    return result


def generate_flow(seed, pure_data, flow):
    """
    Generates custom flow direction

    Parameters
    ----------
    seed
    pure_data
    flow
    """
    rng = np.random.default_rng(int(seed))

    if flow == DataFlowType.RISING:
        pure_data = np.sort(pure_data)

    if flow == DataFlowType.FALLING:
        pure_data = np.flip(np.sort(pure_data))

    noise_data = 0.1 * rng.normal(1, np.std(pure_data), len(pure_data))
    return np.round_(pure_data + noise_data, decimals=2)


def generate_data(seed, distribution, mu, sigma, number_of_entries):
    """
    Generates arbitrary test data

    Parameters
    ----------
    seed
    distribution
    mu
    sigma
    number_of_entries
    """
    rng = np.random.default_rng(int(seed + mu + sigma))
    if distribution == DataDistributionType.GAUSSIAN:
        return np.round_(rng.normal(mu, sigma, number_of_entries), 2)
    elif distribution == DataDistributionType.UNIFORM:
        return np.round_(rng.uniform(mu - 3 * sigma, mu + 3 * sigma, number_of_entries), 2)
    elif distribution == DataDistributionType.LAPLACE:
        return np.round_(rng.laplace(mu, sigma, number_of_entries), 2)
    else:
        return []


def get_random_int(range_start, range_end) -> int:
    """
    Returns a random integer

    Parameters
    ----------
    range_start
    range_end
    """
    return int(np.random.rand() * (range_end - range_start) + range_start)


def get_random_double(range_start, range_end, decimals=2) -> float:
    """
    Returns a random double

    Parameters
    ----------
    range_start
    range_end
    decimals
    """
    return float(round(np.random.rand() * (range_end - range_start) + range_start, decimals))


class DataFlowType(Enum):
    """
    Types of flow directions
    """
    NONE = auto(),
    RISING = auto(),
    FALLING = auto()


class DataDistributionType(Enum):
    """
    Types of statistical distributions
    """
    GAUSSIAN = auto(),
    UNIFORM = auto(),
    LAPLACE = auto()
