import re

import pandas as pd
import wfdb

from ndas.importer.baseimporter import BaseImporter


class WFMImporter(BaseImporter):
    """
    Importer of waveform files
    """
    _datatype = "wfm"  # unused

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        args
        kwargs
        """
        super(WFMImporter, self).__init__(*args, **kwargs)

        """
        Src: https://physionet.org/content/mimic3wdb-matched/1.0/
        
        Intermediate directory p04 contains all records with names that begin with p04 
        (patients with a subject_id between 40000 and 49999.)
        
        All files associated with patient 44083 are contained within the directory p04/p044083. 
        This directory contains two waveform records (p044083-2112-05-04-19-50 and p044083-2112-05-23-12-22) 
        and two corresponding numerics records (p044083-2112-05-04-19-50n and p044083-2112-05-23-12-22n), 
        recorded from two separate ICU stays.
    
        The master waveform header file for the first stay (p044083-2112-05-04-19-50.hea) indicates that 
        the record is 20342033 sample intervals (about 45 hours) in length, and begins at 19:50 on May 4, 2112. 
        This date, as with all dates in MIMIC-III, has been anonymized by shifting it by a random number of days 
        into the future. See header(5) in the WFDB Applications Guide for more information about the format of this file.
    
        This waveform record consists of 41 segments (3314767_0001 through to 3314767_0041), as indicated by 
        the master header file. The layout header file (3314767_layout.hea) indicates that 
        four ECG signals (II, AVR, V, and MCL) were recorded, along with a respiration signal, photoplethysmogram, 
        and arterial blood pressure. Not all of these signals are available simultaneously.
    
        The header file for segment number 4 (3314767_0004.hea) shows us that during this segment, five signals 
        are available: three ECG leads (II, V, and AVR), a respiration signal (RESP), and a PPG signal (PLETH).
    
        The numerics header file (p044083-2112-05-04-19-50n.hea) shows us that a variety of measurements were recorded, 
        including heart rate, invasive and non-invasive blood pressure, respiratory rate, ST segment elevation, 
        oxygen saturation, and cardiac rhythm statistics. Just as with waveforms, not all of these measurements 
        are available at all times.
        """

    def get_dataframe(self, files):
        """
        Loads the data from the waveform files in a dataframe

        Parameters
        ----------
        files
        """
        self.files = files
        self.files_without_path = [re.sub(r'^(.*[\\\/])', '', file) for file in files]

        # SRC: https://physionet.org/physiotools/wag/header-5.htm#sect5
        # p040337-2166-10-17-17-09/13 6 125 3922000 17:09:37.504 17/10/2166

        for single_wfm_h in self.get_waveform_headers(files):
            self.log("Inspecting %s" % single_wfm_h)

            """
            {'record_name': 'drive02',
             'n_sig': 5,
             'fs': 15.5,
             'counter_freq': None,
             'base_counter': None,
             'sig_len': 78056,
             'base_time': None,
             'base_date': None,
             'comments': [],
             'sig_name': ['ECG', 'foot GSR', 'HR', 'marker', 'RESP'],
             'p_signal': None,
             'd_signal': None,
             'e_p_signal': None,
             'e_d_signal': None,
             'file_name': ['drive02.dat',
              'drive02.dat',
              'drive02.dat',
              'drive02.dat',
              'drive02.dat'],
             'fmt': ['16', '16', '16', '16', '16'],
             'samps_per_frame': [32, 2, 1, 1, 2],
             'skew': [None, None, None, None, None],
             'byte_offset': [None, None, None, None, None],
             'adc_gain': [1000.0, 1000.0, 1.0001, 100.0, 500.0],
             'baseline': [0, 0, 0, 0, 0],
             'units': ['mV', 'mV', 'bpm', 'mV', 'mV'],
             'adc_res': [16, 16, 16, 16, 16],
             'adc_zero': [0, 0, 0, 0, 0],
             'init_value': [-1236, 1802, 75, 0, 5804],
             'checksum': [14736, 13501, -19070, -9226, -14191],
             'block_size': [0, 0, 0, 0, 0]}
            """
            header = wfdb.rdrecord(single_wfm_h[:-4])  # strip ending

            self.custom_labels = ["WAVE: " + str(n) + " (" + m + ")" for m, n in zip(header.units, header.sig_name)]
            self.custom_labels.insert(0, "sample (#)")

            df_wave = pd.DataFrame.from_records(header.p_signal, columns=header.sig_name)
            df_wave.insert(0, 'sample', range(header.sig_len))
            df_wave.dropna(how="all", inplace=True)

            return df_wave

    def get_labels(self, files):
        """
        Returns the labels from the waveform files

        Parameters
        ----------
        files
        """
        return self.custom_labels

    def get_full_segment_path(self, segment_name, add_file_ending=False):
        """
        Returns the full path of a single segment

        Parameters
        ----------
        segment_name
        add_file_ending
        """
        if add_file_ending:
            segment_name = segment_name + ".hea"
        match = list(filter(lambda x: re.search(segment_name, x), self.files))
        if not match:
            self.log("Invalid request. Full segment path not found for segment %s" % segment_name)
            return False
        else:
            return match[0]

    def get_waveform_headers(self, files):
        """
        Returns the headers for a waveform file

        Parameters
        ----------
        files
        """
        wfm_header = []

        for file in files:
            self.log("Reading waveform file: %s" % file)

            # first read the master header files
            if re.search("(\d\d\d\d-\d\d-\d\d-\d\d-\d\d.hea)", file):
                wfm_header.append(file)

        if not wfm_header:
            raise ValueError("Failed to detect any waveform master header files")

        return wfm_header
