from io import StringIO

import numpy as np
import pandas as pd

from ndas.importer.baseimporter import BaseImporter


class CSVImporter(BaseImporter):
    """
    Importer of CSV files
    """
    _datatype = "csv"  # unused

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        args
        kwargs
        """
        super(CSVImporter, self).__init__(*args, **kwargs)

    def get_dataframe(self, file):
        """
        Re-implemented file loading for data import

        Parameters
        ----------
        file
        """
        # first remove strings if there are any
        with open(file, "r") as raw_file:
            data = raw_file.read().replace('"', '').replace(";", ",")

        df = pd.read_csv(StringIO(data), dtype=np.float32, delimiter=',', na_values=['.', ''], skip_blank_lines=True)
        df = df.sort_values(df.columns[0])

        return df

    def get_labels(self, files):
        """
        Re-implemented method to return labels.
        Not in use in csv importer.

        Parameters
        ----------
        files
        """
        return []
