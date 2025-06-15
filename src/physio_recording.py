import sys
sys.path.append('../src/')

from typing import Dict, Any, Union
from pathlib import Path

import numpy as np
import pandas as pd

import neurokit2 as nk

class PhysioRecording:
    """A recording of a single recording."""

    eda: pd.DataFrame = pd.DataFrame(columns = ['raw', 'processed', 'epochs', 'features'])
    bvp: pd.DataFrame = pd.DataFrame(columns = ['raw', 'processed', 'epochs', 'features'])
    temperature: pd.DataFrame = pd.DataFrame(columns = ['raw', 'processed', 'epochs', 'features'])

    subject_id: int = None
    session_id: int = None

    physio_filepath: Path = None

    data_loaded = False
    data_processed = False
    features_extracted = False
    data_epoched = False

    verbose: bool = True

    def __init__(self, subject_id: int, session_id: int, verbose: bool = True) -> None:
        self.subject_id = subject_id
        self.session_id = session_id
        self.eda = {
            'raw': {},
            'processed': {},
            'epochs': {},
            'features': {}

        }
        self.bvp = {
            'raw': {},
            'processed': {},
            'epochs': {},
            'features': {}
        }
        self.temperature = {
            'raw': {},
            'processed': {},
            'epochs': {},
            'features': {}
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
            print(f"\tLoading raw data for session {self.session_id} and subject {self.subject_id}")

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

                    signal_pd: Dict[str, Any] = {}

                    signal_pd["sampling_rate"] = sampling_rate
                    signal_pd["nb_channels"] = 1
                    signal_pd["nb_samples"] = len(signal_data)
                    signal_pd["duration"] = len(signal_data) / sampling_rate if sampling_rate > 0 else 0
                    signal_pd["data"] = signal_data
                        
                if sheet_name == "EDA_rs":
                    self.eda["raw"]["rs"] = signal_pd
                    if self.verbose:
                        print(f"\t\tLoaded resting state EDA data from {sheet_name} with {len(signal_data)} samples at {sampling_rate} Hz.")     
                    
                elif sheet_name == "EDA_session":
                    self.eda["raw"]["session"] = signal_pd
                    if self.verbose:
                        print(f"\t\tLoaded session EDA data from {sheet_name} with {len(signal_data)} samples at {sampling_rate} Hz.")

                elif sheet_name == "BVP_rs":
                    self.bvp["raw"]["rs"] = signal_pd
                    if self.verbose:
                        print(f"\t\tLoaded resting state BVP data from {sheet_name} with {len(signal_data)} samples at {sampling_rate} Hz.")

                elif sheet_name == "BVP_session":
                    self.bvp["raw"]["session"] = signal_pd
                    if self.verbose:
                        print(f"\t\tLoaded session BVP data from {sheet_name} with {len(signal_data)} samples at {sampling_rate} Hz.")
                    
                elif sheet_name == "TEMP_rs":
                    self.temperature["raw"]["rs"] = signal_pd
                    if self.verbose:
                        print(f"\t\tLoaded resting state Temperature data from {sheet_name} with {len(signal_data)} samples at {sampling_rate} Hz.")

                elif sheet_name == "TEMP_session":
                    self.temperature["raw"]["rs"] = signal_pd
                    if self.verbose:
                        print(f"\t\tLoaded session Temperature data from {sheet_name} with {len(signal_data)} samples at {sampling_rate} Hz.")                

            except Exception as e:
                raise IOError(f"Error loading physio data: {e}")

        self.data_loaded = True

    def process_raw_data(self) -> None:
        """Process the physio data."""
        if not self.data_loaded:
            raise ValueError("Data has not been loaded. Please load the data before processing.")

        if self.verbose:
            print(f"\tProcessing physio data for subject {self.subject_id} and session {self.session_id}...")

        self.process_raw_eda()
        self.process_raw_bvp()
        self.process_raw_temperature()

        self.data_processed = True


    def process_raw_eda(self) -> None:
        """Process EDA data."""
        new_sampling_rate = 64  # Hz, desired sampling rate for EDA processing

        if not self.data_loaded:
            raise ValueError("Data has not been loaded. Please load the data before processing.")

        if self.verbose:
            print("\t\tProcessing EDA data... ")

        eda_signal_rs = self.eda["raw"]["rs"]["data"]
        eda_signal_session = self.eda["raw"]["session"]["data"]
        sampling_rate = self.eda["raw"]["rs"]["sampling_rate"]
        if eda_signal_rs.empty or eda_signal_session.empty:
            raise ValueError("EDA raw data is empty. Please load the data before processing.")
        
        eda_cleaned_signal_rs = nk.eda_clean(eda_signal_rs, sampling_rate=sampling_rate)
        eda_cleaned_signal_session = nk.eda_clean(eda_signal_session, sampling_rate=sampling_rate)
        if self.verbose:
            print(f"\t\tCleaned EDA resting state data with {len(eda_cleaned_signal_rs)} samples at {sampling_rate} Hz.")
            print(f"\t\tCleaned EDA session data with {len(eda_cleaned_signal_session)} samples at {sampling_rate} Hz.")

        eda_resampled_rs = nk.signal_resample(eda_cleaned_signal_rs, sampling_rate=sampling_rate, desired_sampling_rate=new_sampling_rate, method='interpolation')
        eda_resampled_session = nk.signal_resample(eda_cleaned_signal_session, sampling_rate=sampling_rate, desired_sampling_rate=new_sampling_rate, method='interpolation')
        if self.verbose:
            print(f"\t\tResampled EDA resting state data to {new_sampling_rate} Hz with {len(eda_resampled_rs)} samples.")
            print(f"\t\tResampled EDA session data to {new_sampling_rate} Hz with {len(eda_resampled_session)} samples.")

        eda_processed_rs, info = nk.eda_process(
            eda_resampled_rs,
            sampling_rate=new_sampling_rate)
        eda_processed_session, info = nk.eda_process(
            eda_resampled_session,
            sampling_rate=new_sampling_rate)

        self.eda["processed"]["rs"] = eda_processed_rs.to_dict()
        self.eda["processed"]["session"] = eda_processed_session.to_dict()
        self.eda["processed"]["rs"]["sampling_rate"] = new_sampling_rate
        self.eda["processed"]["session"]["sampling_rate"] = new_sampling_rate

        if self.verbose:
            print(f"\t\tProcessed EDA resting state data with {len(eda_cleaned_signal_rs)} samples at {sampling_rate} Hz.")
            print(f"\t\tProcessed EDA session data with {len(eda_cleaned_signal_session)} samples at {sampling_rate} Hz.")


    def process_raw_bvp(self) -> None:
        """Process BVP data."""
        if not self.data_loaded:
            raise ValueError("Data has not been loaded. Please load the data before processing.")

        if self.verbose:
            print("\t\tProcessing BVP data...")

        bvp_signal_rs = self.bvp["raw"]["rs"]["data"]
        bvp_signal_session = self.bvp["raw"]["session"]["data"]
        sampling_rate = self.bvp["raw"]["rs"]["sampling_rate"]

        bvp_processed_rs, info = nk.ppg_process(
                bvp_signal_rs, 
                sampling_rate=sampling_rate,
                method='elgendi',
                method_quality='templatematch'
            )

        bvp_processed_session, info = nk.ppg_process(
                bvp_signal_session, 
                sampling_rate=sampling_rate,
                method='elgendi',
                method_quality='templatematch'
            )

        self.bvp["processed"]["rs"] = bvp_processed_rs.to_dict()
        self.bvp["processed"]["session"] = bvp_processed_session.to_dict()
        self.bvp["processed"]["rs"]["sampling_rate"] = sampling_rate
        self.bvp["processed"]["session"]["sampling_rate"] = sampling_rate

        self.compute_rr_intervals_bvp()
        # TODO: implement HRV metrics computation for BVP data
        
        if self.verbose:
            print(f"\t\tProcessed BVP resting state data with {len(bvp_processed_rs)} samples at {sampling_rate} Hz.")
            print(f"\t\tProcessed BVP session data with {len(bvp_processed_session)} samples at {sampling_rate} Hz.")

    def process_raw_temperature(self) -> None:
        """Process Temperature data."""
        if not self.data_loaded:
            raise ValueError("Data has not been loaded. Please load the data before processing.")

        if self.verbose:
            print("\t\tProcessing Temperature data... not implemented yet ...")


    def compute_rr_intervals_bvp(self) -> None:
        """Extract features from BVP data."""
        if self.verbose:
            print("\t\tComputing RR intervals from BVP data...")

        bvp_processed_rs = self.bvp["processed"]["rs"]
        bvp_processed_session = self.bvp["processed"]["session"]
        sampling_rate = self.bvp["processed"]["rs"]["sampling_rate"]

        if self.verbose:
            print(f"\t\tExtracting RR intervals from BVP data with sampling rate {sampling_rate} Hz...")

        # Compute RR intervals
        rr_intervals_rs = compute_rr_intervals(bvp_processed_rs, sampling_rate, verbose=self.verbose)
        rr_intervals_session = compute_rr_intervals(bvp_processed_session, sampling_rate, verbose=self.verbose)
        if rr_intervals_rs is None or rr_intervals_session is None:
            raise ValueError("Failed to compute RR intervals from BVP data. Please check the data quality.")
        
        # Correct RR intervals
        min_rr = 300  # ms, physiological limit for minimum RR interval
        max_rr = 2000  # ms, physiological limit for maximum RR interval
        corrected_rr_rs = correct_rr_intervals(rr_intervals_rs, min_rr=min_rr, max_rr=max_rr, verbose=self.verbose)
        corrected_rr_session = correct_rr_intervals(rr_intervals_session, min_rr=min_rr, max_rr=max_rr, verbose=self.verbose)

        if self.verbose:
            print(f"\t\tExtracted {len(corrected_rr_rs)} RR intervals from resting state BVP data.")
            print(f"\t\tExtracted {len(corrected_rr_session)} RR intervals from session BVP data.")

        self.bvp["processed"]["rs"]["RR_Intervals"] = corrected_rr_rs

        self.bvp["processed"]["session"]["RR_Intervals"] = corrected_rr_session

        if self.verbose:
            print(f"\t\tComputed RR intervals for resting state BVP data with {len(corrected_rr_rs)} intervals.")
            print(f"\t\tComputed RR intervals for session BVP data with {len(corrected_rr_session)} intervals.")


    def epoch_time_serie_with_fixed_duration(self, signal_type: str, key: str, duration: int, overlap: int = 0) -> None:
        """Epoch time series with a fixed duration (in seconds)."""
        if self.verbose:
            print(f"\t\tEpoching {signal_type.upper()} time series for '{key}' with duration {duration}s and overlap {overlap}s.")

        if not self.data_processed:
            raise ValueError("Data has not been processed. Please process the data before epoching.")

        if duration <= 0:
            raise ValueError("Duration must be a positive integer.")

        if signal_type not in ["eda", "bvp", "temperature"]:
            raise ValueError("signal_type must be one of ['eda', 'bvp', 'temperature'].")

        # Retrieve the signal and sampling rate
        if signal_type == "eda":
            serie_rs = self.eda["processed"]["rs"][key]
            serie_session = self.eda["processed"]["session"][key]
            sampling_rate = self.eda["processed"]["rs"]["sampling_rate"]
        elif signal_type == "bvp":
            serie_rs = self.bvp["processed"]["rs"][key]
            serie_session = self.bvp["processed"]["session"][key]
            sampling_rate = self.bvp["processed"]["rs"]["sampling_rate"]
        elif signal_type == "temperature":
            serie_rs = self.temperature["processed"]["rs"][key]
            serie_session = self.temperature["processed"]["session"][key]
            sampling_rate = self.temperature["processed"]["rs"]["sampling_rate"]

        # Convert the signal to numpy array
        signal_session = np.asarray(list(serie_session.values())).flatten()
        timestamps = np.arange(len(signal_session)) / sampling_rate  # Generate timestamps in seconds

        # Epoch the signal
        epochs = segment_signal_epochs(signal_session, timestamps, epoch_len=duration, epoch_overlap=overlap)

        if signal_type == "eda":
            self.eda["epochs"]["rs"][key] = [np.asarray(serie_rs)] # Save the original signal
            self.eda["epochs"]["session"][key] = epochs
        elif signal_type == "bvp":
            self.bvp["epochs"]["rs"][key] = [np.asarray(serie_rs)]
            self.bvp["epochs"]["session"][key] = epochs
        elif signal_type == "temperature":
            self.temperature["epochs"]["rs"][key] = [np.asarray(serie_rs)]
            self.temperature["epochs"]["session"][key] = epochs
        
        if self.verbose:
            print(f"\t\tCreated {len(epochs)} epochs of {duration}s from {signal_type.upper()} '{key}' data.")
 
    def epoch_time_serie_with_fixed_number(self, signal_type: str, key: str, n_epochs: int) -> None:
        """Epoch time series into a fixed number of equal-length segments."""
        if self.verbose:
            print(f"\t\tEpoching {signal_type.upper()} time series for '{key}' into {n_epochs} equal epochs...")

        if not self.data_processed:
            raise ValueError("Data has not been processed. Please process the data before epoching.")

        if n_epochs <= 1:
            raise ValueError("Number of epochs must be greater than 1.")

        if signal_type not in ["eda", "bvp", "temperature"]:
            raise ValueError("signal_type must be one of ['eda', 'bvp', 'temperature'].")

        # Retrieve signal and sampling rate
        if signal_type == "eda":
            serie_rs = self.eda["processed"]["rs"][key]
            serie_session = self.eda["processed"]["session"][key]
            sampling_rate = self.eda["processed"]["rs"]["sampling_rate"]
        elif signal_type == "bvp":
            serie_rs = self.bvp["processed"]["rs"][key]
            serie_session = self.bvp["processed"]["session"][key]
            sampling_rate = self.bvp["processed"]["rs"]["sampling_rate"]
        elif signal_type == "temperature":
            serie_rs = self.temperature["processed"]["rs"][key]
            serie_session = self.temperature["processed"]["session"][key]
            sampling_rate = self.temperature["processed"]["rs"]["sampling_rate"]

        # Convert signal to numpy
        signal_session = np.asarray(list(serie_session.values())).flatten()
        timestamps = np.arange(len(signal_session)) / sampling_rate

        # Calculate epoch boundaries
        total_samples = len(signal_session)
        samples_per_epoch = total_samples // n_epochs
        epochs = [
            signal_session[i * samples_per_epoch: (i + 1) * samples_per_epoch]
            for i in range(n_epochs)
        ]

        # Handle trailing samples if not divisible
        remaining_samples = total_samples % n_epochs
        if remaining_samples > 0:
            # Optionally add remaining samples to the last epoch
            epochs[-1] = np.concatenate([epochs[-1], signal_session[-remaining_samples:]])

        # Store
        if signal_type == "eda":
            self.eda["epochs"]["rs"][key] = [np.asarray(serie_rs)]
            self.eda["epochs"]["session"][key] = epochs
        elif signal_type == "bvp":
            self.bvp["epochs"]["rs"][key] = [np.asarray(serie_rs)]
            self.bvp["epochs"]["session"][key] = epochs
        elif signal_type == "temperature":
            self.temperature["epochs"]["rs"][key] = [np.asarray(serie_rs)]
            self.temperature["epochs"]["session"][key] = epochs

        if self.verbose:
            print(f"\t\tCreated {len(epochs)} epochs of equal length for {signal_type.upper()} '{key}' data.")


    def epoch_time_serie_with_sliding_window(self, signal_type: str, key: str, duration: float, step: float) -> None:
        """Epoch time series using a sliding window of fixed duration and step (both in seconds)."""
        if self.verbose:
            print(f"\t\tSliding-window epoching {signal_type.upper()} time series for '{key}' with duration {duration}s and step {step}s...")

        if not self.data_processed:
            raise ValueError("Data has not been processed. Please process the data before epoching.")

        if duration <= 0 or step <= 0:
            raise ValueError("Duration and step must be positive.")

        if signal_type not in ["eda", "bvp", "temperature"]:
            raise ValueError("signal_type must be one of ['eda', 'bvp', 'temperature'].")

        # Retrieve the signal and sampling rate
        if signal_type == "eda":
            serie_rs = self.eda["processed"]["rs"][key]
            serie_session = self.eda["processed"]["session"][key]
            sampling_rate = self.eda["processed"]["rs"]["sampling_rate"]
        elif signal_type == "bvp":
            serie_rs = self.bvp["processed"]["rs"][key]
            serie_session = self.bvp["processed"]["session"][key]
            sampling_rate = self.bvp["processed"]["rs"]["sampling_rate"]
        elif signal_type == "temperature":
            serie_rs = self.temperature["processed"]["rs"][key]
            serie_session = self.temperature["processed"]["session"][key]
            sampling_rate = self.temperature["processed"]["rs"]["sampling_rate"]

        # Convert signal to numpy
        signal_session = np.asarray(list(serie_session.values())).flatten()
        total_samples = len(signal_session)
        window_size = int(duration * sampling_rate)
        step_size = int(step * sampling_rate)

        epochs = []
        for start in range(0, total_samples - window_size + 1, step_size):
            end = start + window_size
            epochs.append(signal_session[start:end])

        # Store
        if signal_type == "eda":
            self.eda["epochs"]["rs"][key] = [np.asarray(serie_rs)]
            self.eda["epochs"]["session"][key] = epochs
        elif signal_type == "bvp":
            self.bvp["epochs"]["rs"][key] = [np.asarray(serie_rs)]
            self.bvp["epochs"]["session"][key] = epochs
        elif signal_type == "temperature":
            self.temperature["epochs"]["rs"][key] = [np.asarray(serie_rs)]
            self.temperature["epochs"]["session"][key] = epochs

        if self.verbose:
            print(f"\t\tCreated {len(epochs)} sliding epochs of {duration}s every {step}s for {signal_type.upper()} '{key}'.")


    def epoch_intervals_serie_with_fixed_duration(self, signal_type: str, key: str, duration: float, overlap: float = 0.0) -> None:
        """Epoch interval series (e.g., RR, ISI) with a fixed duration (in seconds)."""
        if self.verbose:
            print(f"\t\tEpoching {signal_type.upper()} interval series for '{key}' with fixed duration {duration}s...")

        if not self.data_processed:
            raise ValueError("Data has not been processed. Please process the data before epoching.")

        if duration <= 0:
            raise ValueError("Duration must be a positive number.")

        duration_ms = duration * 1000  # convert to milliseconds
        overlap_ms = overlap * 1000  # convert to milliseconds

        if signal_type not in ["eda", "bvp", "temperature"]:
            raise ValueError("signal_type must be one of ['eda', 'bvp', 'temperature'].")

        if signal_type == "eda":
            interval_rs = self.eda["processed"]["rs"][key]
            interval_session = self.eda["processed"]["session"][key]
        elif signal_type == "bvp":
            interval_rs = self.bvp["processed"]["rs"][key]
            interval_session = self.bvp["processed"]["session"][key]
        elif signal_type == "temperature":
            interval_rs = self.temperature["processed"]["rs"][key]
            interval_session = self.temperature["processed"]["session"][key]

        def segment_fixed(intervals, duration_ms, overlap_ms):
            segments = []
            cumsum = np.cumsum(intervals)
            start = 0
            while start + duration_ms <= cumsum[-1]:
                end = start + duration_ms
                mask = (cumsum >= start) & (cumsum < end)
                segment = intervals[mask]
                if len(segment) > 0:
                    segments.append(segment)
                start += duration_ms - overlap_ms
            return segments

        if signal_type == "eda":
            self.eda["epochs"]["rs"][key] = [np.asarray(interval_rs)]
            self.eda["epochs"]["session"][key] = segment_fixed(np.asarray(interval_session), duration_ms, overlap_ms)
        elif signal_type == "bvp":
            self.bvp["epochs"]["rs"][key] = [np.asarray(interval_rs)]
            self.bvp["epochs"]["session"][key] = segment_fixed(np.asarray(interval_session), duration_ms, overlap_ms)
        elif signal_type == "temperature":
            self.temperature["epochs"]["rs"][key] = [np.asarray(interval_rs)]
            self.temperature["epochs"]["session"][key] = segment_fixed(np.asarray(interval_session), duration_ms, overlap_ms)

        if self.verbose:
            print(f"\t\tCreated {len(getattr(self, signal_type)['epochs']['session'][key])} epochs of {duration}s for {signal_type.upper()} '{key}' data.")

    def epoch_intervals_serie_with_fixed_number(self, signal_type: str, key: str, n_epochs: int) -> None:
        """Epoch interval series (e.g., RR, ISI) into a fixed number of segments."""
        if self.verbose:
            print(f"\t\tEpoching {signal_type.upper()} interval series for '{key}' into {n_epochs} equal parts...")

        if not self.data_processed:
            raise ValueError("Data has not been processed. Please process the data before epoching.")

        if n_epochs <= 1:
            raise ValueError("Number of epochs must be greater than 1.")

        if signal_type not in ["eda", "bvp", "temperature"]:
            raise ValueError("signal_type must be one of ['eda', 'bvp', 'temperature'].")

        if signal_type == "eda":
            interval_rs = self.eda["processed"]["rs"][key]
            interval_session = self.eda["processed"]["session"][key]
        elif signal_type == "bvp":
            interval_rs = self.bvp["processed"]["rs"][key]
            interval_session = self.bvp["processed"]["session"][key]
        elif signal_type == "temperature":
            interval_rs = self.temperature["processed"]["rs"][key]
            interval_session = self.temperature["processed"]["session"][key]

        def segment_equal_chunks(intervals, n):
            length = len(intervals)
            chunk_size = length // n
            segments = [intervals[i*chunk_size:(i+1)*chunk_size] for i in range(n - 1)]
            segments.append(intervals[(n - 1)*chunk_size:])
            return segments

        if signal_type == "eda":
            self.eda["epochs"]["rs"][key] = [np.asarray(interval_rs)]
            self.eda["epochs"]["session"][key] = segment_equal_chunks(np.asarray(interval_session), n_epochs)
        elif signal_type == "bvp":
            self.bvp["epochs"]["rs"][key] = [np.asarray(interval_rs)]
            self.bvp["epochs"]["session"][key] = segment_equal_chunks(np.asarray(interval_session), n_epochs)
        elif signal_type == "temperature":
            self.temperature["epochs"]["rs"][key] = [np.asarray(interval_rs)]
            self.temperature["epochs"]["session"][key] = segment_equal_chunks(np.asarray(interval_session), n_epochs)

        if self.verbose:
            print(f"\t\tCreated {len(getattr(self, signal_type)['epochs']['session'][key])} epochs of equal length for {signal_type.upper()} '{key}' data.")

    def epoch_intervals_serie_with_sliding_window(self, signal_type: str, key: str, duration: float, step: float) -> None:
        """Epoch interval series (e.g., RR, ISI) using sliding window (in seconds)."""
        if self.verbose:
            print(f"\t\tSliding-window epoching {signal_type.upper()} interval series for '{key}' with duration {duration}s and step {step}s...")

        if not self.data_processed:
            raise ValueError("Data has not been processed. Please process the data before epoching.")

        if duration <= 0 or step <= 0:
            raise ValueError("Duration and step must be positive.")

        duration_ms = duration * 1000
        step_ms = step * 1000

        if signal_type not in ["eda", "bvp", "temperature"]:
            raise ValueError("signal_type must be one of ['eda', 'bvp', 'temperature'].")

        if signal_type == "eda":
            interval_rs = self.eda["processed"]["rs"][key]
            interval_session = self.eda["processed"]["session"][key]
        elif signal_type == "bvp":
            interval_rs = self.bvp["processed"]["rs"][key]
            interval_session = self.bvp["processed"]["session"][key]
        elif signal_type == "temperature":
            interval_rs = self.temperature["processed"]["rs"][key]
            interval_session = self.temperature["processed"]["session"][key]
        
        def segment_sliding(intervals, duration_ms, step_ms):
            segments = []
            cumsum = np.cumsum(intervals)
            start = 0
            while start + duration_ms <= cumsum[-1]:
                end = start + duration_ms
                mask = (cumsum >= start) & (cumsum < end)
                segment = intervals[mask]
                if len(segment) > 0:
                    segments.append(segment)
                start += step_ms
            return segments

        if signal_type == "eda":
            self.eda["epochs"]["rs"][key] = [np.asarray(interval_rs)]
            self.eda["epochs"]["session"][key] = segment_sliding(np.asarray(interval_session), duration_ms, step_ms)
        elif signal_type == "bvp":
            self.bvp["epochs"]["rs"][key] = [np.asarray(interval_rs)]
            self.bvp["epochs"]["session"][key] = segment_sliding(np.asarray(interval_session), duration_ms, step_ms)
        elif signal_type == "temperature":
            self.temperature["epochs"]["rs"][key] = [np.asarray(interval_rs)]
            self.temperature["epochs"]["session"][key] = segment_sliding(np.asarray(interval_session), duration_ms, step_ms)
        
        if self.verbose:
            print(f"\t\tCreated {len(getattr(self, signal_type)['epochs']['session'][key])} sliding epochs of {duration}s every {step}s for {signal_type.upper()} '{key}' data.")

    def epoch_metric(self, signal_type: str, key: str, method: str, is_interval: bool = False, **kwargs: Dict[str, Any]) -> None:
        """Epoch a metric using a method."""
        if self.verbose:
            print(f"\t\tEpoching {signal_type.upper()} metric for '{key}' using {method}...")

        if method not in ["fixed_duration", "fixed_number", "sliding_window"]:
            raise ValueError("Method must be one of ['fixed_duration', 'fixed_number', 'sliding_window'].")

        if not self.data_processed:
            raise ValueError("Data has not been processed. Please process the data before epoching.")

        if signal_type not in ["eda", "bvp", "temperature"]:
            raise ValueError("signal_type must be one of ['eda', 'bvp', 'temperature'].")

        if method == "fixed_duration":
            if "duration" not in kwargs:
                duration = 60.0  # Default duration in seconds
            else:
                duration = kwargs["duration"] # Duration in seconds
            if "overlap" not in kwargs:
                overlap = 0.0  # Default overlap in seconds
            else:
                overlap = kwargs["overlap"]
            if is_interval:
                self.epoch_intervals_serie_with_fixed_duration(signal_type, key, duration, overlap)
            else:
                self.epoch_time_serie_with_fixed_duration(signal_type, key, duration, overlap)

        elif method == "fixed_number":
            if "n_epochs" not in kwargs:
                n_epochs = 10
            else:
                n_epochs = kwargs["n_epochs"]
            if is_interval:
                self.epoch_intervals_serie_with_fixed_number(signal_type, key, n_epochs)
            else:
                self.epoch_time_serie_with_fixed_number(signal_type, key, n_epochs)

        elif method == "sliding_window":
            if "duration" not in kwargs:
                duration = 60.0
            else:
                duration = kwargs["duration"]
            if "step" not in kwargs:
                step = 10.0
            else:
                step = kwargs["step"]
            if is_interval:
                self.epoch_intervals_serie_with_sliding_window(signal_type, key, duration, step)
            else:
                self.epoch_time_serie_with_sliding_window(signal_type, key, duration, step)

        if self.verbose:
            print(f"\t\tEpoching complete for {signal_type.upper()} metric '{key}' using {method} method.")

    def epoch_processed_signals(self, method: str = "fixed_duration", **kwargs):
        """Epoch the signal using the specified method."""
        if not self.data_processed:
            raise ValueError("Data has not been processed. Please process the data before epoching.")
        if method not in ["fixed_duration", "fixed_number", "sliding_window"]:
            raise ValueError("Method must be one of ['fixed_duration', 'fixed_number', 'sliding_window'].")
        if self.verbose:
            print(f"\tEpoching signal for subject {self.subject_id} and session {self.session_id} using method '{method}'...")

        self.eda["epochs"] = {"rs": {}, "session": {}, "method": method}
        self.bvp["epochs"] = {"rs": {}, "session": {}, "method": method}
        self.temperature["epochs"] = {"rs": {}, "session": {}, "method": method}

        # Epoch EDA data
        if "EDA_Tonic" in self.eda["processed"]["rs"].keys():
            self.epoch_metric("eda", "EDA_Tonic", method, **kwargs)
        if "EDA_Phasic" in self.eda["processed"]["rs"].keys():
            self.epoch_metric("eda", "EDA_Phasic", method, **kwargs)
        # Epoch BVP data
        if "RR_Intervals" in self.bvp["processed"]["rs"].keys():
            self.epoch_metric("bvp", "RR_Intervals", method, is_interval=True, **kwargs)
        # Epoch Temperature data

    '''
    def save_processed_data(self, output_path: Path) -> None:
        """Save the processed data to the specified output path."""
        if not self.data_loaded:
            raise ValueError("Data has not been loaded. Please load the data before saving.")

        if not isinstance(output_path, Path):
            raise TypeError("output_path must be a Path object.")
        
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        eda_df = pd.DataFrame(self.eda["processed"])
        bvp_df = pd.DataFrame(self.bvp["processed"])
        temperature_df = pd.DataFrame(self.temperature["processed"])

        eda_df.to_excel(output_path / f"eda_processed_{self.subject_id}_{self.session_id}.xlsx", index=False)
        bvp_df.to_excel(output_path / f"bvp_processed_{self.subject_id}_{self.session_id}.xlsx", index=False)
        temperature_df.to_excel(output_path / f"temperature_processed_{self.subject_id}_{self.session_id}.xlsx", index=False)

        if self.verbose:
            print(f"Processed data saved to {output_path}")
    '''




















def compute_rr_intervals(processed_ppg: pd.DataFrame, sampling_rate: int, verbose: bool = True) -> np.ndarray:
    """
    Compute RR intervals from processed PPG signal.
        
    Parameters:
    -----------
    processed_ppg : pd.DataFrame
        Processed PPG signal with peaks detected
    sampling_rate : int
        Sampling rate of the PPG signal in Hz
    verbose : bool
        Print verbose messages, by default True      
    Returns:
    --------
    np.ndarray or None
        Array of RR intervals in milliseconds
    """
    try:
        if 'PPG_Peaks' not in processed_ppg.keys():
            raise ValueError("Processed PPG data must contain 'PPG_Peaks' column with peak detection results.")

        peaks = np.array(list(processed_ppg['PPG_Peaks'].values()), dtype=int)
        #peaks = [1 if x else 0 for x in processed_ppg['PPG_Peaks'].values()]
        peak_indices = np.where(peaks == 1)[0]
            
        if len(peak_indices) < 2:
            raise ValueError("Not enough peaks detected for RR interval computation.")
            
        # Compute RR intervals in milliseconds
        rr_intervals = np.diff(peak_indices) * (1000 / sampling_rate)
        
        return rr_intervals
            
    except Exception as e:
        raise ValueError(f"Error computing RR intervals: {e}")
    

def correct_rr_intervals(rr_intervals: np.ndarray, 
                       min_rr: float = 0,
                       max_rr: float = 10000, verbose: bool = True) -> np.ndarray:
    """
    Detect and correct outliers in RR intervals.
        
    Parameters:
    -----------
    rr_intervals : np.ndarray
        Array of RR intervals in milliseconds
    min_rr, max_rr : float, optional
        Physiological limits (defaults from config)
    verbose : bool
        Whether to print debug information
    Returns:
    --------
    np.ndarray
        Corrected RR intervals in milliseconds
    """
        
    corrected_rr = []
    anomalies = 0
    
    i = 0
        
    while i < len(rr_intervals):

        current_rr = rr_intervals[i]
        
        if current_rr < min_rr:  # Too small -> Possible double peak
            anomalies += 1
            if corrected_rr and i + 1 < len(rr_intervals):
                corrected_rr[-1] += current_rr / 2
                rr_intervals[i + 1] += current_rr / 2
                    
        elif current_rr > max_rr:  # Too large -> Possible missed peaks
            anomalies += 1
            if corrected_rr and i + 1 < len(rr_intervals):
                reference = (corrected_rr[-1] + rr_intervals[i + 1])/2
            else:
                reference = np.median(rr_intervals)
            
            num_splits = max(1, int(np.round(current_rr / reference)))
            split_value = current_rr / num_splits
            corrected_rr.extend([split_value] * num_splits)

        else:
            corrected_rr.append(current_rr)
                
        i += 1
        
    anomaly_ratio = anomalies / len(rr_intervals)

    if verbose:
        print(f"\t\tDetected {anomalies} anomalies ({anomaly_ratio:.2%} of total RR intervals).")
        if anomaly_ratio > 0.1:
            print("\t\tWarning: High anomaly ratio detected, consider reviewing the data quality.")
    
    return np.array(corrected_rr)
    

def compute_hrv_metrics(peaks: Union[pd.Series, np.ndarray], 
                  sampling_rate: int, verbose: bool = True) -> Dict:
    """
    Compute heart rate variability metrics.
        
    Parameters:
    -----------
    peaks : pd.Series or np.ndarray
        Peak detection signal or RR intervals
    sampling_rate : int
        Sampling rate in Hz
    verbose : bool
        Print verbose messages, by default True
    
    Returns:
    --------
    dict or None
        Dictionary containing HRV metrics
    """
    
    try:

        '''
        # Compute different HRV domains
        hrv_time = safe_compute(
            nk.hrv_time, peaks, sampling_rate=sampling_rate, show=False,
            default=pd.DataFrame(), error_msg="HRV time domain computation failed"
        )
            
        hrv_freq = safe_compute(
            nk.hrv_frequency, peaks, sampling_rate=sampling_rate, show=False,
            default=pd.DataFrame(), error_msg="HRV frequency domain computation failed"
        )
            
        hrv_nonlinear = safe_compute(
            nk.hrv_nonlinear, peaks, sampling_rate=sampling_rate, show=False,
            default=pd.DataFrame(), error_msg="HRV nonlinear computation failed"
        )
            
        # Combine all metrics
        hrv_metrics = pd.concat([hrv_time, hrv_freq, hrv_nonlinear], axis=1)

            
        if hrv_metrics.empty:
            logger.warning("No HRV metrics could be computed")
            return {}
            
        result = hrv_metrics.to_dict('records')[0]
        logger.info(f"Computed {len(result)} HRV metrics")
        return result

        '''

    except Exception as e:
        raise ValueError(f"Error computing HRV metrics: {e}")
        return {}


def segment_signal_epochs(
    signal: np.ndarray,
    timestamps: np.ndarray,
    epoch_len: float = 60.0,
    epoch_overlap: float = 0.0
) -> list:
    """
    Segment a signal into fixed-length epochs/windows.

    Parameters
    ----------
    signal : np.ndarray
        1D array of signal values (e.g., heart rate)
    timestamps : np.ndarray
        1D array of timestamps (seconds) matching signal
    epoch_len : float
        Length of each epoch in seconds
    epoch_overlap : float
        Overlap between epochs in seconds

    Returns
    -------
    List of np.ndarray
        List of signal segments (each a 1D array)
    """
    assert len(signal) == len(timestamps), "Signal and timestamps must have same length"
    segments = []
    t_min = timestamps[0]
    t_max = timestamps[-1]
    step = epoch_len - epoch_overlap
    n_epochs = int(np.floor((t_max - t_min - epoch_overlap) / step)) + 1

    for i in range(n_epochs):
        start = t_min + i * step
        end = start + epoch_len
        mask = (timestamps >= start) & (timestamps < end)
        segment = signal[mask]
        if segment.size > 0:
            segments.append(segment)

    return segments
