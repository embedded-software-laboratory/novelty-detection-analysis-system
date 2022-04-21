from enum import Enum


REGULAR = '4e81bd'
TRAINING = '00afbb'
TIER2NOV = 'e7b800'
TIER1NOV = 'fc4e08'
LINECOLOR = 'fc4e08'
BLACK = '000000'
IGNORED = 'A9A9A9'
REPLACED = 'BD4EB8'
ADDED = 'ebb4ff'


def init_colors(config):
    """
    Initializes plot-colors

    Parameters
    ----------
    config
    """
    global REGULAR, TRAINING, TIER2NOV, TIER1NOV, LINECOLOR, IGNORED, REPLACED, ADDED

    REGULAR = config["regular"]
    TRAINING = config["training"]
    TIER2NOV = config["tier-2-novelty"]
    TIER1NOV = config["tier-1-novelty"]
    LINECOLOR = config["line-color"]
    IGNORED = config["ignored"]
    REPLACED = config["replaced"]
    ADDED = config["added"]
