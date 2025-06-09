# DataLoader

import os
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.data import Dataset



class ExcelDataLoader:
    def __init__(self, data_dir, data_files=None, load_transform=None, transform=None):
        self.data_dir = data_dir
        self.data_files = data_files if data_files else []
        self.load_transform = load_transform
        self.transform = transform
        self.data = self._load_data()

    def _load_data(self):
        """
        Loads data from the specified directory from Excel files and applies the load_transform if provided.

        Returns:
            list: List of loaded data samples.
        """
        data = []
        for file_name in self.data_files:
            file_path = os.path.join(self.data_dir, file_name)
            if os.path.isfile(file_path) and file_name.endswith('.xlsx'):
                sample = pd.read_excel(file_path)
                if self.load_transform:
                    sample = self.load_transform(sample)
                data.append(sample)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a data sample by index and applies the transform if provided.

        Args:
            idx (int): Index of the data sample to retrieve.

        Returns:
            sample: The data sample at the specified index, transformed if applicable.
        """
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __str__(self):
        return f"ExcelDataLoader with {len(self.data)} samples from {self.data_dir}"

    def verbose(self):
        if not os.path.exists(self.data_dir):
            print(f"Data directory {self.data_dir} does not exist.")
            return
        if not os.path.isdir(self.data_dir):
            print(f"Data directory {self.data_dir} is not a valid directory.")
            return
        if not self.load_transform and not self.transform:
            print("No transformations provided for loading or processing data.")
        if not self.data:
            print("No data loaded. Check if the directory contains valid Excel data files.")
            return
        if not self.data_files:
            print("No data files specified or found in the directory.")
            return
        print(f"ExcelDataLoader with {len(self.data)} samples from {self.data_dir}")
        print(f"Data files: {self.data_files}")
        

    def validate(self, verbose=True):
        """
        Validates the loaded data by checking if the specified files exist and are valid Excel files.
        Also checks if the data is loaded correctly and if each sample is a valid DataFrame.
        
        Args:
            verbose (bool): If True, prints detailed validation messages
        """
        # Check if files exist and are valid Excel files
        check = True
        for file_name in self.data_files:
            file_path = os.path.join(self.data_dir, file_name)
            if not os.path.isfile(file_path):
                if verbose:
                    print(f"File {file_path} does not exist.")
                check = False
            elif not file_name.endswith('.xlsx'):
                if verbose:
                    print(f"File {file_path} is not a valid Excel file.")
                check = False
        
        if not check:
            if verbose:
                print("Validation failed. Some files do not exist or are not valid Excel files.")
            return False
        
        if verbose:
            print("All specified files exist and are valid Excel files.")
        
        # Check if the data is loaded correctly
        if not self.data:
            if verbose:
                print("No data loaded. Check if the directory contains valid Excel data files.")
            return False
        
        if verbose:
            print(f"Loaded {len(self.data)} samples from {self.data_dir}.")

        # Check if each sample is a valid DataFrame
        for idx, sample in enumerate(self.data):
            if not isinstance(sample, pd.DataFrame):
                if verbose:
                    print(f"Sample at index {idx} is not a valid DataFrame.")
                return False
        
        if verbose:
            print("All samples are valid DataFrames.")
        
        return True



class UnknownDataLoader(Dataset):
    """
    A placeholder dataset class for unknown data types.
    This class can be extended to handle specific data loading and transformations.
    """
    def __init__(self, data_dir, data_files=None, load_transform=None, transform=None):
        self.data_dir = data_dir
        self.data_files = data_files if data_files else []
        self.load_transform = load_transform
        self.transform = transform
        self.data = self._load_data()

    def _load_data(self):
        return []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx] if idx < len(self.data) else None
    
    def __str__(self):
        return f"UnknownDataLoader with {len(self.data)} samples from {self.data_dir}"
    
    def verbose(self):
        print(f"UnknownDataLoader with {len(self.data)} samples from {self.data_dir}")
        if not self.data_files:
            print("No data files specified or found in the directory.")
        else:
            print(f"Data files: {self.data_files}")
        return self


class PPGdatainfo():
    def __init__(self, sampling_rate, data, nan_value, verbose=False):
        self.sampling_rate = sampling_rate  # Sampling rate in Hz
        self.data = data  # PPG data as a NumPy array or similar structure
        self.NAN = nan_value  # Value used to represent NaN in the data
        self.verbose = verbose  # Verbose mode for additional output

    def get_sampling_rate(self):
        return self.sampling_rate

    def get_data(self):
        return self.data

    def get_nan_value(self):
        return self.NAN

    def get_verbose(self):
        return self.verbose
    

class PPGdata():
    def __init__(self, ppg_info: PPGdatainfo = None):
        self.sampling_rate = None
        self.data = None
        self.NAN = None

        if ppg_info is not None:
            self.load_from_info(ppg_info)

    def set_sampling_rate(self, sampling_rate):
        if not isinstance(sampling_rate, (int, float)) or sampling_rate <= 0:
            raise ValueError("Sampling rate must be a positive number.")
        self.sampling_rate = sampling_rate

    def set_data(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array.")
        self.data = data

    def set_nan_value(self, nan_value):
        self.NAN = nan_value

    def get_nan_value(self):
        return self.NAN

    def get_sampling_rate(self):
        return self.sampling_rate

    def get_data(self):
        return self.data

    def validate_data_structure(self):
        if self.sampling_rate is None:
            raise ValueError("Sampling rate is not set.")
        if self.data is None:
            raise ValueError("Data is not set.")

    def load_from_info(self, ppg_info: PPGdatainfo):
        try:
            self.set_sampling_rate(ppg_info.get_sampling_rate())
            self.set_data(ppg_info.get_data())
            self.set_nan_value(ppg_info.get_nan_value())

            verbose = ppg_info.get_verbose()
            if verbose:
                print(f"Loaded PPG data with sampling rate: {self.sampling_rate} Hz")
                print(f"Data shape: {self.data.shape}")
                print(f"NAN value: {self.NAN}")
        except AttributeError as e:
            print(f"Error loading data from PPGdatainfo: {e}")
        self.validate_data_structure()
        return self

    def __str__(self):
        return (f"PPGdata with sampling rate: {self.sampling_rate} Hz, "
                f"data shape: {self.data.shape if self.data is not None else 'None'}, "
                f"NAN value: {self.NAN}")

    def data_info(self):
        """
        Returns a container-like dictionary containing the PPG data information.
        This structure groups related data and can include metadata or additional context.
        """
        data_container = {
            'metadata': {
                'description': 'PPG Data Information',
                'timestamp': datetime.now().isoformat()
            },
            'data': {
                'sampling_rate': self.sampling_rate,
                'data_shape': self.data.shape if self.data is not None else None,
                'nan_value': self.NAN
            }
        }

        if self.verbose:
            print("PPG Data Information:")
            print(f"  Sampling Rate: {self.sampling_rate} Hz")
            print(f"  Data Shape: {self.data.shape if self.data is not None else 'None'}")
            print(f"  NaN Value: {self.NAN}")
            print(f"  Timestamp: {datetime.now().isoformat()}")

        return data_container
    
class EDAdatainfo():
    def __init__(self, sampling_rate, data, nan_value, verbose=False):
        self.sampling_rate = sampling_rate
        self.data = data
        self.NAN = nan_value
        self.verbose = verbose

    def get_sampling_rate(self):
        return self.sampling_rate

    def get_data(self):
        return self.data

    def get_nan_value(self):
        return self.NAN

    def get_verbose(self):
        return self.verbose
    

class EDAdata:
    def __init__(self, eda_info=None):
        self.sampling_rate = None
        self.data = None
        self.NAN = None

        if eda_info is not None:
            self.load_from_info(eda_info)

    def set_sampling_rate(self, sampling_rate):
        if not isinstance(sampling_rate, (int, float)) or sampling_rate <= 0:
            raise ValueError("Sampling rate must be a positive number.")
        self.sampling_rate = sampling_rate

    def set_data(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array.")
        self.data = data

    def set_nan_value(self, nan_value):
        self.NAN = nan_value

    def get_nan_value(self):
        return self.NAN

    def get_sampling_rate(self):
        return self.sampling_rate

    def get_data(self):
        return self.data

    def validate_data_structure(self):
        if self.sampling_rate is None:
            raise ValueError("Sampling rate is not set.")
        if self.data is None:
            raise ValueError("Data is not set.")

        if np.isnan(self.data).any():
            if self.NAN is None:
                raise ValueError("Data contains NaN values but no NAN replacement value is set.")
            self.data[np.isnan(self.data)] = self.NAN

    def load_from_info(self, eda_info):
        try:
            self.set_sampling_rate(eda_info.get_sampling_rate())
            self.set_data(eda_info.get_data())
            self.set_nan_value(eda_info.get_nan_value())

            if eda_info.get_verbose():
                print(f"Loaded EDA data with sampling rate: {self.sampling_rate} Hz")
                print(f"Data shape: {self.data.shape}")
                print(f"NAN value: {self.NAN}")
        except AttributeError as e:
            print(f"Error loading data from EDAdatainfo: {e}")
        self.validate_data_structure()
        return self

    def __str__(self):
        return (f"EDAdata with sampling rate: {self.sampling_rate} Hz, "
                f"data shape: {self.data.shape if self.data is not None else 'None'}, "
                f"NAN value: {self.NAN}")
    
    def data_info(self):
        """
        Returns a container-like dictionary containing the PPG data information.
        This structure groups related data and can include metadata or additional context.
        """
        data_container = {
            'metadata': {
                'description': 'EDA Data Information',
                'timestamp': datetime.now().isoformat()
            },
            'data': {
                'sampling_rate': self.sampling_rate,
                'data_shape': self.data.shape if self.data is not None else None,
                'nan_value': self.NAN
            }
        }

        if self.verbose:
            print("EDA Data Information:")
            print(f"  Sampling Rate: {self.sampling_rate} Hz")
            print(f"  Data Shape: {self.data.shape if self.data is not None else 'None'}")
            print(f"  NaN Value: {self.NAN}")
            print(f"  Timestamp: {datetime.now().isoformat()}")

        return data_container
    

class ECGdatainfo(ExcelDataLoader):
    def __init__(self, sampling_rate, data, nan_value, verbose=False):
        self.sampling_rate = sampling_rate
        self.data = data
        self.NAN = nan_value
        self.verbose = verbose

    def get_sampling_rate(self):
        return self.sampling_rate

    def get_data(self):
        return self.data

    def get_nan_value(self):
        return self.NAN

    def get_verbose(self):
        return self.verbose

class ECGdata(ExcelDataLoader):
    def __init__(self, ecg_info=None):
        self.sampling_rate = None
        self.data = None
        self.NAN = None

        if ecg_info is not None:
            self.load_from_info(ecg_info)

    def set_sampling_rate(self, sampling_rate):
        if not isinstance(sampling_rate, (int, float)) or sampling_rate <= 0:
            raise ValueError("Sampling rate must be a positive number.")
        self.sampling_rate = sampling_rate

    def set_data(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array.")
        self.data = data

    def set_nan_value(self, nan_value):
        self.NAN = nan_value

    def get_nan_value(self):
        return self.NAN

    def get_sampling_rate(self):
        return self.sampling_rate

    def get_data(self):
        return self.data

    def validate_data_structure(self):
        if self.sampling_rate is None:
            raise ValueError("Sampling rate is not set.")
        if self.data is None:
            raise ValueError("Data is not set.")

        if np.isnan(self.data).any():
            if self.NAN is None:
                raise ValueError("Data contains NaN values but no NAN replacement value is set.")
            self.data[np.isnan(self.data)] = self.NAN

    def load_from_info(self, ecg_info):
        try:
            self.set_sampling_rate(ecg_info.get_sampling_rate())
            self.set_data(ecg_info.get_data())
            self.set_nan_value(ecg_info.get_nan_value())

            if ecg_info.get_verbose():
                print(f"Loaded ECG data with sampling rate: {self.sampling_rate} Hz")
                print(f"Data shape: {self.data.shape}")
                print(f"NAN value: {self.NAN}")
        except AttributeError as e:
            print(f"Error loading data from ECGdatainfo: {e}")
        self.validate_data_structure()
        return self

    def __str__(self):
        return (f"ECGdata with sampling rate: {self.sampling_rate} Hz, "
                f"data shape: {self.data.shape if self.data is not None else 'None'}, "
                f"NAN value: {self.NAN}")
    
    def data_info(self):
        """
        Returns a container-like dictionary containing the PPG data information.
        This structure groups related data and can include metadata or additional context.
        """
        data_container = {
            'metadata': {
                'description': 'ECG Data Information',
                'timestamp': datetime.now().isoformat()
            },
            'data': {
                'sampling_rate': self.sampling_rate,
                'data_shape': self.data.shape if self.data is not None else None,
                'nan_value': self.NAN
            }
        }

        if self.verbose:
            print("ECG Data Information:")
            print(f"  Sampling Rate: {self.sampling_rate} Hz")
            print(f"  Data Shape: {self.data.shape if self.data is not None else 'None'}")
            print(f"  NaN Value: {self.NAN}")
            print(f"  Timestamp: {datetime.now().isoformat()}")

        return data_container




class Temperaturedatainfo(ExcelDataLoader):
    def __init__(self, sampling_rate, data, nan_value, verbose=False):
        self.sampling_rate = sampling_rate
        self.data = data
        self.NAN = nan_value
        self.verbose = verbose

    def get_sampling_rate(self):
        return self.sampling_rate

    def get_data(self):
        return self.data

    def get_nan_value(self):
        return self.NAN

    def get_verbose(self):
        return self.verbose

class Temperaturedata(ExcelDataLoader):
    def __init__(self, temperature_info=None):
        self.sampling_rate = None
        self.data = None
        self.NAN = None

        if temperature_info is not None:
            self.load_from_info(temperature_info)

    def set_sampling_rate(self, sampling_rate):
        if not isinstance(sampling_rate, (int, float)) or sampling_rate <= 0:
            raise ValueError("Sampling rate must be a positive number.")
        self.sampling_rate = sampling_rate

    def set_data(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array.")
        self.data = data

    def set_nan_value(self, nan_value):
        self.NAN = nan_value

    def get_nan_value(self):
        return self.NAN

    def get_sampling_rate(self):
        return self.sampling_rate

    def get_data(self):
        return self.data

    def validate_data_structure(self):
        if self.sampling_rate is None:
            raise ValueError("Sampling rate is not set.")
        if self.data is None:
            raise ValueError("Data is not set.")

        if np.isnan(self.data).any():
            if self.NAN is None:
                raise ValueError("Data contains NaN values but no NAN replacement value is set.")
            self.data[np.isnan(self.data)] = self.NAN

    def load_from_info(self, temperature_info):
        try:
            self.set_sampling_rate(temperature_info.get_sampling_rate())
            self.set_data(temperature_info.get_data())
            self.set_nan_value(temperature_info.get_nan_value())

            if temperature_info.get_verbose():
                print(f"Loaded Temperature data with sampling rate: {self.sampling_rate} Hz")
                print(f"Data shape: {self.data.shape}")
                print(f"NAN value: {self.NAN}")
        except AttributeError as e:
            print(f"Error loading data from Temperaturedatainfo: {e}")
        self.validate_data_structure()
        return self

    def __str__(self):
        return (f"Temperaturedata with sampling rate: {self.sampling_rate} Hz, "
                f"data shape: {self.data.shape if self.data is not None else 'None'}, "
                f"NAN value: {self.NAN}")
    
    def data_info(self):
        """
        Returns a container-like dictionary containing the PPG data information.
        This structure groups related data and can include metadata or additional context.
        """
        data_container = {
            'metadata': {
                'description': 'Temperature Data Information',
                'timestamp': datetime.now().isoformat()
            },
            'data': {
                'sampling_rate': self.sampling_rate,
                'data_shape': self.data.shape if self.data is not None else None,
                'nan_value': self.NAN
            }
        }

        if self.verbose:
            print("Temperature Data Information:")
            print(f"  Sampling Rate: {self.sampling_rate} Hz")
            print(f"  Data Shape: {self.data.shape if self.data is not None else 'None'}")
            print(f"  NaN Value: {self.NAN}")
            print(f"  Timestamp: {datetime.now().isoformat()}")

        return data_container

class pickledDataLoader(ExcelDataLoader):
    """
    A DataLoader that loads data from a directory containing pickled files.
    """
    def __init__(self, data_dir, data_files=None, load_transform=None, transform=None):
        super().__init__(data_dir, data_files, load_transform, transform)

    def save_data(self, data, file_name):
        """
        Saves the provided data to a pickled file in the specified directory.

        Args:
            data (any): The data to be saved.
            file_name (str): The name of the file to save the data to.
        """
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        file_path = os.path.join(self.data_dir, file_name)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {file_path}")

    def _load_data(self):
        """
        Loads data from the specified directory from pickled files and applies the load_transform if provided.

        Returns:
            list: List of loaded data samples.
        """
        import pickle
        data = []
        for file_name in self.data_files:
            file_path = os.path.join(self.data_dir, file_name)
            if os.path.isfile(file_path) and file_name.endswith('.pkl'):
                with open(file_path, 'rb') as f:
                    sample = pickle.load(f)
                if self.load_transform:
                    sample = self.load_transform(sample)
                data.append(sample)
        return data
    def __str__(self):
        return f"PickledDataLoader with {len(self.data)} samples from {self.data_dir}"
    def verbose(self):
        if not os.path.exists(self.data_dir):
            print(f"Data directory {self.data_dir} does not exist.")
            return
        if not os.path.isdir(self.data_dir):
            print(f"Data directory {self.data_dir} is not a valid directory.")
            return
        if not self.load_transform and not self.transform:
            print("No transformations provided for loading or processing data.")
        if not self.data:
            print("No data loaded. Check if the directory contains valid pickled data files.")
            return
        if not self.data_files:
            print("No data files specified or found in the directory.")
            return
        print(f"PickledDataLoader with {len(self.data)} samples from {self.data_dir}")
        print(f"Data files: {self.data_files}")
    def validate(self, verbose=True):
        """
        Validates the loaded data by checking if the specified files exist and are valid pickled files.
        Also checks if the data is loaded correctly and if each sample is a valid object.
        Args:
            verbose (bool): If True, prints detailed validation messages
        """