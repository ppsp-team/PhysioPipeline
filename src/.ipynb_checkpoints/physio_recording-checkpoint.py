import sys
sys.path.append('../src/')

import pandas as pd
from pathlib import Path


class PhysioRecording:
    """A recording of a single recording."""

    eda: pd.DataFrame = pd.DataFrame(columns = ['raw', 'processed', 'features', 'epochs'])
    bvp: pd.DataFrame = pd.DataFrame(columns = ['raw', 'processed', 'features', 'epochs'])
    temperature: pd.DataFrame = pd.DataFrame(columns = ['raw', 'processed', 'features', 'epochs'])

    subject_id: int = None
    session_id: int = None

    physio_filepath: Path = None

    data_loaded = False
    verbose: bool = True

    def __init__(self, subject_id: int, session_id: int, verbose: bool = True) -> None:
        self.subject_id = subject_id
        self.session_id = session_id
        self.eda = {
            'raw': {},
            'processed': {},
            'features': {},
            'epochs': {}
        }
        self.bvp = {
            'raw': {},
            'processed': {},
            'features': {},
            'epochs': {}
        }
        self.temperature = {
            'raw': {},
            'processed': {},
            'features': {},
            'epochs': {}
        }
        self.verbose = verbose


    def set_physio_filepath(self, physio_filepath: Path) -> None:
        """Set the file path for the physio recording."""
        if not isinstance(physio_filepath, Path):
            raise TypeError("physio_filepath must be a Path object.")
        if not physio_filepath.suffix == '.xlsx':
            raise ValueError("Physio file must be an Excel file with .xlsx extension.")
        if not physio_filepath.exists():
            raise FileNotFoundError(f"Physio file {self.physio_filepath} does not exist.")

        self.physio_filepath = physio_filepath

    def get_physio_filepath(self) -> Path:
        """Get the file path for the physio recording."""
        if self.physio_filepath is None:
            print("Physio file path has not been set.")
            return None
        return self.physio_filepath

    def load_raw_data(self) -> None:
        """Load the physio data from the file."""
        if self.physio_filepath is None:
            raise ValueError("Physio file path has not been set.")

        if self.verbose:
            print(f"\nLoading raw data for session {self.session_id} and subject {self.subject_id}")

        xls = pd.ExcelFile(self.physio_filepath)
        if xls is None:
            raise ValueError(f"Error reading file: {self.physio_filepath}. Please check the file format and content.")

        for sheet_name in xls.sheet_names:

            try:

                if not sheet_name in ["EDA_rs", "EDA_session", "BVP_rs", "BVP_session", "TEMP_rs", "TEMP_session"]:
                    continue
                    
                data = pd.read_excel(xls, sheet_name=sheet_name)
                
                if len(data) > 0:
                    # First row contains sampling rate
                    sampling_rate = int(data.iloc[0, 0])
                    # Actual data starts from third row (index 2)
                    signal_data = data.iloc[2:].reset_index(drop=True).squeeze()
                    signal_data = pd.Series(signal_data.astype(float))

                    signal_pd = pd.DataFrame()

                    signal_pd["sampling_rate"] = sampling_rate
                    signal_pd["nb_channels"] = 1
                    signal_pd["nb_samples"] = len(signal_data)
                    signal_pd["duration"] = len(signal_data) / sampling_rate if sampling_rate > 0 else 0
                    signal_pd["data"] = signal_data
                        
                if sheet_name == "EDA_rs":
                    self.eda["raw"]["rs"] = signal_pd
                    if self.verbose:
                        print(f"Loaded resting state EDA data from {sheet_name} with {len(signal_data)} samples at {sampling_rate} Hz.")     
                    
                elif sheet_name == "EDA_session":
                    self.eda["raw"]["session"] = signal_pd
                    if self.verbose:
                        print(f"Loaded session EDA data from {sheet_name} with {len(signal_data)} samples at {sampling_rate} Hz.")

                elif sheet_name == "BVP_rs":
                    self.bvp["raw"]["rs"] = signal_pd
                    if self.verbose:
                        print(f"Loaded resting state BVP data from {sheet_name} with {len(signal_data)} samples at {sampling_rate} Hz.")

                elif sheet_name == "BVP_session":
                    self.bvp["raw"]["session"] = signal_pd
                    if self.verbose:
                        print(f"Loaded session BVP data from {sheet_name} with {len(signal_data)} samples at {sampling_rate} Hz.")
                    
                elif sheet_name == "TEMP_rs":
                    self.temperature["raw"]["rs"] = signal_pd
                    if self.verbose:
                        print(f"Loaded resting state Temperature data from {sheet_name} with {len(signal_data)} samples at {sampling_rate} Hz.")

                elif sheet_name == "TEMP_session":
                    self.temperature["raw"]["rs"] = signal_pd
                    if self.verbose:
                        print(f"Loaded session Temperature data from {sheet_name} with {len(signal_data)} samples at {sampling_rate} Hz.")                

            except Exception as e:
                raise IOError(f"Error loading physio data: {e}")

        self.data_loaded = True

    def process_data(self) -> None:
        """Process the physio data."""
        if not self.data_loaded:
            raise ValueError("Data has not been loaded. Please load the data before processing.")

        if self.verbose:
            print("Processing physio data... (this is a placeholder)")

        self.process_eda()
        self.process_bvp()
        self.process_temperature()



    def process_eda(self) -> None:
        """Process EDA data."""
        if not self.data_loaded:
            raise ValueError("Data has not been loaded. Please load the data before processing.")

        if self.verbose:
            print("Processing EDA data... (this is a placeholder)")


    def process_bvp(self) -> None:
        """Process BVP data."""
        if not self.data_loaded:
            raise ValueError("Data has not been loaded. Please load the data before processing.")

        if self.verbose:
            print("Processing BVP data... (this is a placeholder)")

    def process_temperature(self) -> None:
        """Process Temperature data."""
        if not self.data_loaded:
            raise ValueError("Data has not been loaded. Please load the data before processing.")

        if self.verbose:
            print("Processing Temperature data... (this is a placeholder)")



                