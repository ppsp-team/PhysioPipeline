"""
This module provides a class for handling physiological recordings, including loading, processing, and epoching data.
"""

import sys
sys.path.append('../src/')

from typing import Dict, Any, Union
from pathlib import Path

import numpy as np
import pandas as pd

import neurokit2 as nk

class PhysioRecording:
    """
    A class to handle physiological recordings for a subject and session.
    It supports loading raw data from an Excel file, processing the data, and epoching the time series data.
    """

    def __init__(self, session_id: int, subject_id: int, seance_id: int, verbose: bool = True) -> None:
        self.session_id = session_id
        self.subject_id = subject_id
        self.seance_id = seance_id

        self.eda = {
            'raw': {},
            'processed': {},
            'epochs': {},

        }
        self.bvp = {
            'raw': {},
            'processed': {},
            'epochs': {},
        }
        self.temperature = {
            'raw': {},
            'processed': {},
            'epochs': {},
        }

        self.heartrate = {
            'raw': {},
            'processed': {},
            'epochs': {},
        }

        self.physio_filepath: Path = None

        self.data_loaded = False
        self.data_processed = False
        self.data_epoched = False

        self.verbose = verbose

    def set_physio_filepath(self, physio_filepath: Path) -> None:
        """
        Set the file path for the physio recording.
        Args:
            physio_filepath (Path): The file path for the physio recording.
            Raises:
                TypeError: If physio_filepath is not a Path object.
                ValueError: If physio_filepath does not have a .xlsx extension.
                FileNotFoundError: If the physio file does not exist.
        """
        if not isinstance(physio_filepath, Path):
            raise TypeError("physio_filepath must be a Path object.")
        if not physio_filepath.suffix == '.xlsx':
            raise ValueError("Physio file must be an Excel file with .xlsx extension.")
        if not physio_filepath.exists():
            raise FileNotFoundError(f"Physio file {self.physio_filepath} does not exist.")

        self.physio_filepath = physio_filepath
        if self.verbose:
            print(f"\tPhysio file path set to {self.physio_filepath}")

    def get_physio_filepath(self) -> Path:
        """
        Get the file path for the physio recording.
        Returns:
            Path: The file path for the physio recording.
        """
        if self.physio_filepath is None:
            print("Physio file path has not been set.")
            return None
        return self.physio_filepath

    def load_raw_data(self) -> None:
        """
        Load raw physiological data from the specified Excel file.
        Raises:
            ValueError: If the physio file path has not been set.
            IOError: If there is an error reading the physio file.
        """
        if self.physio_filepath is None:
            raise ValueError("Physio file path has not been set.")

        if self.verbose:
            print(f"\tLoading raw data for session {self.session_id} and subject {self.subject_id}")

        xls = pd.ExcelFile(self.physio_filepath)
        if xls is None:
            raise ValueError(f"Error reading file: {self.physio_filepath}. Please check the file format and content.")

        for sheet_name in xls.sheet_names:

            try:

                if not sheet_name in ["EDA", "BVP", "TEMP", "HR"]:
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
                        
                if sheet_name == "EDA":
                    self.eda["raw"] = signal_pd
                    if self.verbose:
                        print(f"\t\tLoaded resting state EDA data from {sheet_name} with {len(signal_data)} samples at {sampling_rate} Hz.")     
                    
                elif sheet_name == "BVP":
                    self.bvp["raw"] = signal_pd
                    if self.verbose:
                        print(f"\t\tLoaded resting state BVP data from {sheet_name} with {len(signal_data)} samples at {sampling_rate} Hz.")
                
                elif sheet_name == "TEMP":
                    self.temperature["raw"] = signal_pd
                    if self.verbose:
                        print(f"\t\tLoaded resting state Temperature data from {sheet_name} with {len(signal_data)} samples at {sampling_rate} Hz.")

                elif sheet_name == "HR":
                    self.heartrate["raw"] = signal_pd
                    if self.verbose:
                        print(f"\t\tLoaded resting state HR data from {sheet_name} with {len(signal_data)} samples at {sampling_rate} Hz.")

            except Exception as e:
                raise IOError(f"Error loading physio data: {e}")

        self.data_loaded = True

    def process_raw_data(self) -> None:
        """
        Process the raw physiological data for the subject and session.
        Raises:
            ValueError: If data has not been loaded before processing.
        """
        if not self.data_loaded:
            raise ValueError("Data has not been loaded. Please load the data before processing.")

        if self.verbose:
            print(f"\tProcessing physio data for subject {self.subject_id} and session {self.session_id}...")

        self.process_raw_eda()
        self.process_raw_bvp()
        self.process_raw_temperature()
        self.process_raw_heartrate()

        self.data_processed = True

        if self.verbose:
            print("\t\tProcessing complete.")


    def process_raw_eda(self) -> None:
        """Process EDA data."""
        new_sampling_rate = 64  # Hz, desired sampling rate for EDA processing

        if not self.data_loaded:
            raise ValueError("Data has not been loaded. Please load the data before processing.")

        if self.verbose:
            print("\t\tProcessing EDA data... ")

        eda_signal_session = self.eda["raw"]["data"]
        sampling_rate = self.eda["raw"]["sampling_rate"]
        if eda_signal_session.empty:
            raise ValueError("EDA raw data is empty. Please load the data before processing.")
        
        eda_cleaned_signal_session = nk.eda_clean(eda_signal_session, sampling_rate=sampling_rate)
        if self.verbose:
            print(f"\t\tCleaned EDA session data with {len(eda_cleaned_signal_session)} samples at {sampling_rate} Hz.")

        eda_resampled_session = nk.signal_resample(eda_cleaned_signal_session, sampling_rate=sampling_rate, desired_sampling_rate=new_sampling_rate, method='interpolation')
        if self.verbose:
            print(f"\t\tResampled EDA session data to {new_sampling_rate} Hz with {len(eda_resampled_session)} samples.")

        eda_processed_session, info = nk.eda_process(
            eda_resampled_session,
            sampling_rate=new_sampling_rate)

        self.eda["processed"] = eda_processed_session.to_dict()
        self.eda["processed"]["sampling_rate"] = new_sampling_rate

        if self.verbose:
            print(f"\t\tProcessed EDA session data with {len(eda_cleaned_signal_session)} samples at {sampling_rate} Hz.")


    def process_raw_bvp(self) -> None:
        """Process BVP data."""
        if not self.data_loaded:
            raise ValueError("Data has not been loaded. Please load the data before processing.")

        if self.verbose:
            print("\t\tProcessing BVP data...")

        bvp_signal_session = self.bvp["raw"]["data"]
        sampling_rate = self.bvp["raw"]["sampling_rate"]

        bvp_processed_session, info = nk.ppg_process(
                bvp_signal_session, 
                sampling_rate=sampling_rate,
                method='elgendi',
                method_quality='templatematch'
            )

        self.bvp["processed"] = bvp_processed_session.to_dict()
        self.bvp["processed"]["sampling_rate"] = sampling_rate

        self.compute_rr_intervals_bvp()
        # TODO: implement HRV metrics computation for BVP data
        
        if self.verbose:
            print(f"\t\tProcessed BVP session data with {len(bvp_processed_session)} samples at {sampling_rate} Hz.")

    def process_raw_temperature(self) -> None:
        """Process Temperature data."""
        if not self.data_loaded:
            raise ValueError("Data has not been loaded. Please load the data before processing.")

        if self.verbose:
            print("\t\tProcessing Temperature data... not implemented yet ...")

    def process_raw_heartrate(self) -> None:
        """Process Heart Rate data."""
        if not self.data_loaded:
            raise ValueError("Data has not been loaded. Please load the data before processing.")

        if self.verbose:
            print("\t\tProcessing Heart Rate data... ")

        # just copy sample rate and series from HR data
        heartrate_signal_session = self.heartrate["raw"]["data"]
        sampling_rate = self.heartrate["raw"]["sampling_rate"]

        if heartrate_signal_session.empty:
            raise ValueError("Heartrate raw data is empty. Please load the data before processing.")
        
        heartrate_processed_session = {index: value for index, value in enumerate(heartrate_signal_session)}
        
        self.heartrate["processed"] = {
            "Heartrate": heartrate_processed_session,
            "sampling_rate": sampling_rate
        }

        if self.verbose:
            print(f"\t\tProcessed Heart Rate session data with {len(heartrate_processed_session)} samples at {sampling_rate} Hz.")

    def compute_rr_intervals_bvp(self) -> None:
        """Extract features from BVP data."""
        if self.verbose:
            print("\t\tComputing RR intervals from BVP data...")

        bvp_processed_session = self.bvp["processed"]
        sampling_rate = self.bvp["processed"]["sampling_rate"]

        if self.verbose:
            print(f"\t\tExtracting RR intervals from BVP data with sampling rate {sampling_rate} Hz...")

        # Compute RR intervals
        rr_intervals_session = compute_rr_intervals(bvp_processed_session, sampling_rate, verbose=self.verbose)
        if rr_intervals_session is None:
            raise ValueError("Failed to compute RR intervals from BVP data. Please check the data quality.")
        
        # Correct RR intervals
        min_rr = 300  # ms, physiological limit for minimum RR interval
        max_rr = 2000  # ms, physiological limit for maximum RR interval

        corrected_rr_session = rr_intervals_session.copy()
        anomaly_ratio = 1.0

        while anomaly_ratio > 0.01:
            corrected_rr_session, anomaly_ratio_session = correct_rr_intervals(corrected_rr_session, min_rr=min_rr, max_rr=max_rr, verbose=self.verbose)
            anomaly_ratio = anomaly_ratio_session

        if self.verbose:
            print(f"\t\tExtracted {len(corrected_rr_session)} RR intervals from session BVP data.")

        self.bvp["processed"]["RR_Intervals"] = corrected_rr_session

        if self.verbose:
            print(f"\t\tComputed RR intervals for session BVP data with {len(corrected_rr_session)} intervals.")


    def epoch_time_serie_with_fixed_duration(self, signal_type: str, key: str, duration: int, overlap: int = 0) -> None:
        """
        Epoch time series into fixed-length segments with optional overlap.

        Args:
            signal_type (str): Type of signal to epoch ('eda', 'bvp', 'temperature', 'heartrate').
            key (str): Key for the specific time series to epoch.
            duration (int): Duration of each epoch in seconds.
            overlap (int, optional): Overlap between epochs in 
                seconds. Defaults to 0.
        Raises:
            ValueError: If data has not been processed, duration is not positive, or signal_type is invalid.
        """
        if self.verbose:
            print(f"\t\tEpoching {signal_type.upper()} time series for '{key}' with duration {duration}s and overlap {overlap}s.")

        if not self.data_processed:
            raise ValueError("Data has not been processed. Please process the data before epoching.")

        if duration <= 0:
            raise ValueError("Duration must be a positive integer.")

        if signal_type not in ["eda", "bvp", "temperature", "heartrate"]:
            raise ValueError("signal_type must be one of ['eda', 'bvp', 'temperature', 'heartrate].")

        # Retrieve the signal and sampling rate
        serie_session = self.__getattribute__(signal_type)["processed"][key]
        sampling_rate = self.__getattribute__(signal_type)["processed"]["sampling_rate"]

        # Convert the signal to numpy array
        signal_session = np.asarray(list(serie_session.values())).flatten()
        timestamps = np.arange(len(signal_session)) / sampling_rate  # Generate timestamps in seconds

        # Epoch the signal
        epochs = segment_signal_epochs(signal_session, timestamps, epoch_len=duration, epoch_overlap=overlap)

        self.__getattribute__(signal_type)["epochs"][key] = epochs
        
        if self.verbose:
            print(f"\t\tCreated {len(epochs)} epochs of {duration}s from {signal_type.upper()} '{key}' data.")
 
    def epoch_time_serie_with_fixed_number(self, signal_type: str, key: str, n_epochs: int) -> None:
        """
        Epoch time series into a fixed number of equal-length segments.
        Args:
            signal_type (str): Type of signal to epoch ('eda', 'bvp', 'temperature', 'heartrate').
            key (str): Key for the specific time series to epoch.
            n_epochs (int): Number of epochs to create.
        Raises:
            ValueError: If data has not been processed, n_epochs is not greater than 1, or signal_type is invalid.
        """
        if self.verbose:
            print(f"\t\tEpoching {signal_type.upper()} time series for '{key}' into {n_epochs} equal epochs...")

        if not self.data_processed:
            raise ValueError("Data has not been processed. Please process the data before epoching.")

        if n_epochs <= 1:
            raise ValueError("Number of epochs must be greater than 1.")

        if signal_type not in ["eda", "bvp", "temperature", "heartrate"]:
            raise ValueError("signal_type must be one of ['eda', 'bvp', 'temperature', 'heartrate].")

        # Retrieve the signal and sampling rate
        serie_session = self.__getattribute__(signal_type)["processed"][key]
        sampling_rate = self.__getattribute__(signal_type)["processed"]["sampling_rate"]

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
        self.__getattribute__(signal_type)["epochs"][key] = epochs

        if self.verbose:
            print(f"\t\tCreated {len(epochs)} epochs of equal length for {signal_type.upper()} '{key}' data.")


    def epoch_time_serie_with_sliding_window(self, signal_type: str, key: str, duration: float, step: float) -> None:
        """
        Epoch time series using a sliding window approach.
        Args:
            signal_type (str): Type of signal to epoch ('eda', 'bvp', 'temperature', 'heartrate').
            key (str): Key for the specific time series to epoch.
            duration (float): Duration of each epoch in seconds.
            step (float): Step size for the sliding window in seconds.
        Raises:
            ValueError: If data has not been processed, duration or step is not positive, or signal_type is invalid.
        """
        if self.verbose:
            print(f"\t\tSliding-window epoching {signal_type.upper()} time series for '{key}' with duration {duration}s and step {step}s...")

        if not self.data_processed:
            raise ValueError("Data has not been processed. Please process the data before epoching.")

        if duration <= 0 or step <= 0:
            raise ValueError("Duration and step must be positive.")

        if signal_type not in ["eda", "bvp", "temperature", "heartrate"]:
            raise ValueError("signal_type must be one of ['eda', 'bvp', 'temperature', 'heartrate].")

        # Retrieve the signal and sampling rate
        serie_session = self.__getattribute__(signal_type)["processed"][key]
        sampling_rate = self.__getattribute__(signal_type)["processed"]["sampling_rate"]

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
        self.__getattribute__(signal_type)["epochs"][key] = epochs

        if self.verbose:
            print(f"\t\tCreated {len(epochs)} sliding epochs of {duration}s every {step}s for {signal_type.upper()} '{key}'.")


    def epoch_intervals_serie_with_fixed_duration(self, signal_type: str, key: str, duration: float, overlap: float = 0.0) -> None:
        """
        Epoch interval series (e.g., RR, ISI) into fixed-length segments with optional overlap.
        Args:
            signal_type (str): Type of signal to epoch ('eda', 'bvp', 'temperature', 'heartrate').
            key (str): Key for the specific interval series to epoch.
            duration (float): Duration of each epoch in seconds.
            overlap (float, optional): Overlap between epochs in seconds. Defaults to 0.0.
        Raises:
            ValueError: If data has not been processed, duration is not positive, or signal_type is invalid.
        """
        if self.verbose:
            print(f"\t\tEpoching {signal_type.upper()} interval series for '{key}' with fixed duration {duration}s...")

        if not self.data_processed:
            raise ValueError("Data has not been processed. Please process the data before epoching.")

        if duration <= 0:
            raise ValueError("Duration must be a positive number.")

        duration_ms = duration * 1000  # convert to milliseconds
        overlap_ms = overlap * 1000  # convert to milliseconds

        if signal_type not in ["eda", "bvp", "temperature", "heartrate"]:
            raise ValueError("signal_type must be one of ['eda', 'bvp', 'temperature', 'heartrate'].")

        # Retrieve the signal and sampling rate
        interval_session = self.__getattribute__(signal_type)["processed"][key]
        sampling_rate = self.__getattribute__(signal_type)["processed"]["sampling_rate"]

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
        
        self.__getattribute__(signal_type)["epochs"][key] = segment_fixed(np.asarray(interval_session), duration_ms, overlap_ms)

        if self.verbose:
            print(f"\t\tCreated {len(getattr(self, signal_type)['epochs'][key])} epochs of {duration}s for {signal_type.upper()} '{key}' data.")

    def epoch_intervals_serie_with_fixed_number(self, signal_type: str, key: str, n_epochs: int) -> None:
        """
        Epoch interval series (e.g., RR, ISI) into a fixed number of equal-length segments.
        Args:
            signal_type (str): Type of signal to epoch ('eda', 'bvp', 'temperature', 'heartrate').
            key (str): Key for the specific interval series to epoch.
            n_epochs (int): Number of epochs to create.
        Raises:
            ValueError: If data has not been processed, n_epochs is not greater than 1, or signal_type is invalid.
        """
        if self.verbose:
            print(f"\t\tEpoching {signal_type.upper()} interval series for '{key}' into {n_epochs} equal parts...")

        if not self.data_processed:
            raise ValueError("Data has not been processed. Please process the data before epoching.")

        if n_epochs <= 1:
            raise ValueError("Number of epochs must be greater than 1.")

        if signal_type not in ["eda", "bvp", "temperature", "heartrate"]:
            raise ValueError("signal_type must be one of ['eda', 'bvp', 'temperature', 'heartrate].")

        # Retrieve the signal and sampling rate
        interval_session = self.__getattribute__(signal_type)["processed"][key]
        sampling_rate = self.__getattribute__(signal_type)["processed"]["sampling_rate"]

        def segment_equal_chunks(intervals, n):
            length = len(intervals)
            chunk_size = length // n
            segments = [intervals[i*chunk_size:(i+1)*chunk_size] for i in range(n - 1)]
            segments.append(intervals[(n - 1)*chunk_size:])
            return segments

        self.__getattribute__(signal_type)["epochs"][key] = segment_equal_chunks(np.asarray(interval_session), n_epochs)

        if self.verbose:
            print(f"\t\tCreated {len(getattr(self, signal_type)['epochs'][key])} epochs of equal length for {signal_type.upper()} '{key}' data.")

    def epoch_intervals_serie_with_sliding_window(self, signal_type: str, key: str, duration: float, step: float) -> None:
        """
        Epoch interval series (e.g., RR, ISI) using a sliding window approach.
        Args:
            signal_type (str): Type of signal to epoch ('eda', 'bvp', 'temperature', 'heartrate').
            key (str): Key for the specific interval series to epoch.
            duration (float): Duration of each epoch in seconds.
            step (float): Step size for the sliding window in seconds.
        Raises:
            ValueError: If data has not been processed, duration or step is not positive, or signal_type is invalid.
        """
        if self.verbose:
            print(f"\t\tSliding-window epoching {signal_type.upper()} interval series for '{key}' with duration {duration}s and step {step}s...")

        if not self.data_processed:
            raise ValueError("Data has not been processed. Please process the data before epoching.")

        if duration <= 0 or step <= 0:
            raise ValueError("Duration and step must be positive.")

        duration_ms = duration * 1000
        step_ms = step * 1000

        if signal_type not in ["eda", "bvp", "temperature", "heartrate"]:
            raise ValueError("signal_type must be one of ['eda', 'bvp', 'temperature', 'heartrate].")

        # Retrieve the signal and sampling rate
        interval_session = self.__getattribute__(signal_type)["processed"][key]
        sampling_rate = self.__getattribute__(signal_type)["processed"]["sampling_rate"]
        
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

        self.__getattribute__(signal_type)["epochs"][key] = segment_sliding(np.asarray(interval_session), duration_ms, step_ms)
        
        if self.verbose:
            print(f"\t\tCreated {len(getattr(self, signal_type)['epochs'][key])} sliding epochs of {duration}s every {step}s for {signal_type.upper()} '{key}' data.")

    def epoch_metric(self, signal_type: str, key: str, method: str, is_interval: bool = False, **kwargs: Dict[str, Any]) -> None:
        """
        Epoch a specific metric (e.g., EDA, BVP, Temperature, Heartrate) using the specified method.
        Args:
            signal_type (str): Type of signal to epoch ('eda', 'bvp', 'temperature', 'heartrate').
            key (str): Key for the specific metric to epoch.
            method (str): Method for epoching ('fixed_duration', 'fixed_number', 'sliding_window').
            is_interval (bool, optional): Whether the metric is an interval series (e.g., RR intervals). Defaults to False.
            **kwargs: Additional keyword arguments for the epoching method.
        Raises:
            ValueError: If data has not been processed, method is invalid, or signal_type is invalid.
        """
        if self.verbose:
            print(f"\t\tEpoching {signal_type.upper()} metric for '{key}' using {method}...")

        if method not in ["fixed_duration", "fixed_number", "sliding_window"]:
            raise ValueError("Method must be one of ['fixed_duration', 'fixed_number', 'sliding_window'].")

        if not self.data_processed:
            raise ValueError("Data has not been processed. Please process the data before epoching.")

        if signal_type not in ["eda", "bvp", "temperature", "heartrate"]:
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
        """
        Epoch the processed physiological signals using the specified method.
        Args:
            method (str): Method for epoching ('fixed_duration', 'fixed_number', 'sliding_window').
            **kwargs: Additional keyword arguments for the epoching method.
        Raises:
            ValueError: If data has not been processed, method is invalid, or if the method is not supported.
        """
        if not self.data_processed:
            raise ValueError("Data has not been processed. Please process the data before epoching.")
        if method not in ["fixed_duration", "fixed_number", "sliding_window"]:
            raise ValueError("Method must be one of ['fixed_duration', 'fixed_number', 'sliding_window'].")
        if self.verbose:
            print(f"\tEpoching signal for subject {self.subject_id} and session {self.session_id} using method '{method}'...")

        self.eda["epochs"] = {"method": method}
        self.bvp["epochs"] = {"method": method}
        self.temperature["epochs"] = {"method": method}
        self.heartrate["epochs"] = {"method": method}

        # Epoch EDA data
        if "EDA_Tonic" in self.eda["processed"].keys():
            self.epoch_metric("eda", "EDA_Tonic", method, **kwargs)
        if "EDA_Phasic" in self.eda["processed"].keys():
            self.epoch_metric("eda", "EDA_Phasic", method, **kwargs)
        # Epoch BVP data
        if "RR_Intervals" in self.bvp["processed"].keys():
            self.epoch_metric("bvp", "RR_Intervals", method, is_interval=True, **kwargs)
        if "Heartrate" in self.heartrate["processed"].keys():
            self.epoch_metric("heartrate", "Heartrate", method, **kwargs)
        # Epoch Temperature data

        self.data_epoched = True

        if self.verbose:
            print(f"\tEpoching complete for subject {self.subject_id} and session {self.session_id} using method '{method}'.")

 
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
            if len(rr_intervals) > 2 and i + 2 < len(rr_intervals):
                reference = np.median(rr_intervals[i-2:i+2])  # Use median of surrounding intervals
                if reference < min_rr or reference > max_rr:
                    if verbose:
                        print(f"\t\tWarning: Reference RR interval {reference} ms is outside physiological limits.")
                    reference = np.median(rr_intervals)
            else:
                reference = np.median(rr_intervals)
            
            num_splits = max(2, int(np.round(current_rr / reference)))
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
    
    return np.array(corrected_rr), anomaly_ratio
    

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
