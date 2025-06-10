# DataLoader
# This module provides classes for loading and processing various types of data, including Excel files, pickled files, and custom data structures.
# It includes classes for handling PPG, EDA, ECG, and Temperature data, along with utility functions for validation and transformation.

## TODO : review comment introduction


# Modules
import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import json

from .data_info import *
from path import Path


class DataLoader():

    filename: Path = None
    type: str = None
    dataInfo: DataInfo = None
    verbose: bool = False

    def __init__(self, filename: Path = None, type: str = None, verbose: bool = False):
        """
        Initializes the DataLoader with a filename.
        Args:
            filename (Path): The path to the data file.
        """
        if filename is not None :
            self.set_filename(filename, type)
        self.verbose = verbose

    def set_filename(self, filename: Path, type: str = None):
        """
        Sets the filename and type for the DataLoader.
        Args:
            filename (Path): The path to the data file.
            type (str): The type of the data file (Excel or JSON).
        """
        if isinstance(filename, Path):

            if type.lower() == "excel" or (type == None and filename.endswith('.xlsx')):
                    self.type = "Excel"
                    self.filename = filename

                    if self.verbose:
                        print(f"DataLoader initialized with Excel file: {self.filename}")

            elif type.lower() == "json" or (type == None and filename.endswith('.json')):
                    self.type = "Json"
                    self.filename = filename

                    if self.verbose:
                        print(f"DataLoader initialized with JSON file: {self.filename}")

            else:
                    raise ValueError("Unsupported file type. Please provide an Excel or JSON file.")
        else:
            raise ValueError("Filename must be a valid Path object.")

    def get_filename(self):
        """
        Returns the filename of the DataLoader.
        Returns:
        Path: The filename of the DataLoader.
        """
        return self.filename

    def get_type(self):
        """
        Returns the type of the DataLoader.
        Returns:
            str: The type of the DataLoader (Excel or JSON).
        """
        return self.type

    def get_data_info(self):
        """
        Returns the data information of the DataLoader.
        Returns:
            DataInfo: The data information of the DataLoader.
        """
        return self.dataInfo


    def load_data_from_excel(self):
        """
        Loads data from an Excel file.
        Returns:
            DataFrame: The loaded data.
        """

        file_content: pd.dataframe = None

        if self.type == "Excel":
            file_content = pd.ExcelFile(self.filename)
        else:
            raise ValueError("Invalid file type. Only Excel files are supported.")

            
        for sheet_name in file_content.sheet_names:
            try:
                data = pd.read_excel(xls, sheet_name=sheet_name)
                    
                if len(data) > 0:
                    # First row contains sampling rate
                    sampling_rate = int(data.iloc[0, 0])
                    # Actual data starts from third row (index 2)
                    signal_data = data.iloc[2:].reset_index(drop=True).squeeze()
                        
                    if sheet_name == "EDA_rs":
                        signal_data = signal_data.astype(float)

                        self.dataInfo.setHasEDA(True)
                        self.dataInfo.edaDataInfo.setSamplingRate(sampling_rate)

                        self.dataInfo.edaDataInfo.nb_channels = 1
                        self.dataInfo.edaDataInfo.rs_nb_samples = len(signal_data)
                        if sampling_rate > 0:
                            self.dataInfo.edaDataInfo.rs_duration = len(signal_data) / sampling_rate
                        else:
                            self.dataInfo.edaDataInfo.rs_duration = 0
                        self.dataInfo.edaDataInfo.file_path = self.filename

                        # TODO
                    
                    elif sheet_name == "EDA_session":
                        signal_data = signal_data.astype(float)

                        signal_data = signal_data.astype(float)

                        self.dataInfo.setHasEDA(True)
                        self.dataInfo.edaDataInfo.setSamplingRate(sampling_rate)

                        self.dataInfo.edaDataInfo.nb_channels = 1
                        self.dataInfo.edaDataInfo.rs_nb_samples = len(signal_data)
                        if sampling_rate > 0:
                            self.dataInfo.edaDataInfo.session_duration = len(signal_data) / sampling_rate
                        else:
                            self.dataInfo.edaDataInfo.session_duration = 0
                        self.dataInfo.edaDataInfo.file_path = self.filename

                    elif sheet_name == "BVP_rs":
                        signal_data = signal_data.astype(float)

                        # TODO

                    elif sheet_name == "BVP_session":
                        signal_data = signal_data.astype(float)

                        # TODO


                    
        except Exception as e:
            print(f"Error loading data from Excel file: {e}")
            return None

        
    ## TODO : Remy should do it
    def load_data_from_json(self):
        """
        Loads data from a JSON file.
        Returns:
            dict: The loaded data.
        """

        data: pd.dataframe = None

        if self.type == "Json":
            with open(self.filename, 'r') as f:
                data = json.load(f)
            return data
        else:
            raise ValueError("Invalid file type. Only JSON files are supported.")


        def load_data(self):
        """
        Loads data from the specified file based on its type.
        Returns:
            DataFrame or dict: The loaded data, either as a DataFrame (for Excel) or a dictionary (for JSON).
        """
        if self.type == "Excel":
            return self.load_data_from_excel()
        elif self.type == "Json":
            return self.load_data_from_json()
        else:
            raise ValueError("Unsupported file type. Please provide an Excel or JSON file.")












