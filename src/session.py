"""
Session class for managing physio recordings in a session.
"""


import sys
sys.path.append('../src/')

from physio_recording import PhysioRecording
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


class Session:
    """
    Class representing a session containing physio recordings.
    Attributes:
        session_id (int): Unique identifier for the session.
        family_id (int): Identifier for the family associated with the session.
        seance_id (int): Identifier for the seance associated with the session.
        verbose (bool): Flag to control verbosity of output.
        physio_recordings (List[PhysioRecording]): List of physio recordings in the session.
    """

    def __init__(
        self,
        session_id: int,
        family_id: int,
        seance_id: int,
        verbose: bool = True,
    ):
        """
        Initialize a Session instance.
        Args:
            session_id (int): Unique identifier for the session.
            family_id (int): Identifier for the family associated with the session.
            seance_id (int): Identifier for the seance associated with the session.
            verbose (bool): Flag to control verbosity of output.
        """
        if not isinstance(session_id, int) or session_id < 0:
            raise ValueError("session_id must be a positive integer")
        if not isinstance(family_id, int) or family_id < 0:
            raise ValueError("family_id must be a positive integer")
        if not isinstance(seance_id, int) or seance_id < 0:
            raise ValueError("seance_id must be a positive integer")
        if not isinstance(verbose, bool):
            raise ValueError("verbose must be a boolean value")
        if verbose:
            print(f"Initializing session {session_id} for family {family_id}, seance {seance_id}...")
        self.session_id = session_id
        self.family_id = family_id
        self.seance_id = seance_id
        self.verbose = verbose

        self.physio_recordings: List[PhysioRecording] = []
        if self.verbose:
            print(f"Session {self.session_id} initialized with family {self.family_id} and seance {self.seance_id}.")
        
    
    def add_physio_recording(self, physio_recording: PhysioRecording):
        """
        Add a physio recording to the session.
        Args:
            physio_recording (PhysioRecording): The physio recording to add.
        Raises:
            ValueError: If physio_recording is not an instance of PhysioRecording.
        """
        if not isinstance(physio_recording, PhysioRecording):
            raise ValueError("physio_recording must be an instance of PhysioRecording")
        if self.verbose:
            print(f"Adding physio recording for subject {physio_recording.subject_id} to session {self.session_id}, family {self.family_id}, seance {self.seance_id}...")
        self.physio_recordings.append(physio_recording)
        if self.verbose:
            print(f"Physio recording for subject {physio_recording.subject_id} added to session {self.session_id}.")

    def load_physio_recordings_data(self):
        """
        Load raw data from all physio recordings for the session.
        """
        if self.verbose:
            print(f"Loading physio recordings for session {self.session_id}, family {self.family_id}, seance {self.seance_id}...")
        for recording in self.physio_recordings:
            recording.load_raw_data()

    def process_physio_recordings(self):
        """
        Process all physio recordings for the session.
        """
        if self.verbose:
            print(f"Processing physio recordings for session {self.session_id}, family {self.family_id}, seance {self.seance_id}...")
        for recording in self.physio_recordings:
            recording.process_raw_data()

    def epoch_physio_recordings(self, method: str = 'fixed_duration', **kwargs):
        """
        Epoch physio recordings for the session based on the specified method.
        Args:
            method (str): The method to use for epoching. Default is 'fixed_duration'.
            Valid methods are 'fixed_duration', 'fixed_number', and 'sliding_window'.
            If method is 'fixed_duration', kwargs must include 'duration' (int) and 'overlap' (int).
            If method is 'fixed_number', kwargs must include 'number' (int) and 'overlap' (int).
            If method is 'sliding_window', kwargs must include 'window_size' (int) and 'overlap' (int).            **kwargs: Additional keyword arguments for the epoching method.
        Raises:
            ValueError: If method is not a valid epoching method.
        """
        if self.verbose:
            print(f"Epoching physio recordings for session {self.session_id}, family {self.family_id}, seance {self.seance_id}...")
        for recording in self.physio_recordings:
            recording.epoch_processed_signals(method=method, **kwargs)
        if self.verbose:
            print(f"Physio recordings for session {self.session_id} have been epoch processed using method '{method}'.")



    def plot_time_series(self, signal_type: str, key: str, min_y: float = None, max_y: float = None, return_fig: bool = False):
        """
        Plot time series for a specific signal type and key from the physio recordings in the session.
        Args:
            signal_type (str): The type of signal to plot. Valid options are 'eda', 'bvp', 'temperature', 'heartrate'.
            key (str): The key to plot from the processed data. Valid keys are 'rs' and 'session'.
        Returns:
            fig: The matplotlib figure object containing the plotted time series.
        Raises:
            ValueError: If signal_type is not one of the valid types or if key is not valid."""
        if self.verbose:
            print(f"Plotting time series for signal type '{signal_type}' and key '{key}' for session {self.session_id}, family {self.family_id}, seance {self.seance_id}...")

        if signal_type not in ['eda', 'bvp', 'temperature', 'heartrate']:
            raise ValueError(f"Invalid signal type '{signal_type}'. Valid types are 'eda', 'bvp', 'temperature', 'heartrate'.")
        
        # Create a figure with two columns and three rows
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(3, 2, figure=fig)
        gs.update(wspace=0.025, hspace=0.05)  # set the spacing between subplots

        for i, recording in enumerate(self.physio_recordings):
            if i >= 3:
                break
            ax1 = fig.add_subplot(gs[i, 0])
            ax2 = fig.add_subplot(gs[i, 1])
            # Plot the raw signal
            rs_data = recording.__getattribute__(signal_type)["processed"]["rs"][key]
            if isinstance(rs_data, dict):
                rs_data = list(rs_data.values())
            session_data = recording.__getattribute__(signal_type)["processed"]["session"][key]
            if isinstance(session_data, dict):
                session_data = list(session_data.values())
            if rs_data is not None:
                ax1.plot(rs_data, label=f"RS - Subject {recording.subject_id}", color='blue')
                ax1.set_title(f"{signal_type.upper()} - RS - Subject {recording.subject_id}")
                ax1.set_xlabel("Time")
                ax1.set_ylabel(key)
                ax1.legend()
                if min_y is not None and max_y is not None:
                    ax1.set_ylim(min_y, max_y)
                elif min_y is not None:
                    ax1.set_ylim(bottom=min_y)
                elif max_y is not None:
                    ax1.set_ylim(top=max_y)
            else:
                ax1.set_title(f"{signal_type.upper()} - RS - Subject {recording.subject_id} (No Data)")
            if session_data is not None:
                ax2.plot(session_data, label=f"Session - Subject {recording.subject_id}", color='orange')
                ax2.set_title(f"{signal_type.upper()} - Session - Subject {recording.subject_id}")
                ax2.set_xlabel("Time")
                ax2.set_ylabel(key)
                ax2.legend()
                if min_y is not None and max_y is not None:
                    ax2.set_ylim(min_y, max_y)
                elif min_y is not None:
                    ax2.set_ylim(bottom=min_y)
                elif max_y is not None:
                    ax2.set_ylim(top=max_y)
            else:
                ax2.set_title(f"{signal_type.upper()} - Session - Subject {recording.subject_id} (No Data)")
        
        plt.suptitle(f"Time Series for {signal_type.upper()} - Key: {key} - Session {self.session_id}, Family {self.family_id}, Seance {self.seance_id}")
        plt.show()
        if self.verbose:
            print(f"Time series for signal type '{signal_type}' and key '{key}' plotted for session {self.session_id}, family {self.family_id}, seance {self.seance_id}.")

        if return_fig:
            return fig
        return None
    

    def plot_poincare_map(self, return_fig: bool = False):
        """
        Plot Poincare maps for all physio recordings in the session.
        Returns:
            fig: The matplotlib figure object containing the Poincare maps.
        Raises:
            ValueError: If no physio recordings are available in the session.
        """
        if not self.physio_recordings:
            raise ValueError("No physio recordings available in the session to plot Poincare maps.")
        
        if self.verbose:
            print(f"Plotting Poincare maps for session {self.session_id}, family {self.family_id}, seance {self.seance_id}...")

        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(3, 2, figure=fig)
        gs.update(wspace=0.025, hspace=0.05)  # set the spacing between subplots

        for i, recording in enumerate(self.physio_recordings):
            if i >= 3:
                break

            ax1 = fig.add_subplot(gs[i, 0])
            ax2 = fig.add_subplot(gs[i, 1])

            rr_rs = recording.bvp["processed"]["rs"]["RR_Intervals"]
            rr_session = recording.bvp["processed"]["session"]["RR_Intervals"]
            if rr_rs is not None and len(rr_rs) >= 3:
                rr_rs = np.asarray(rr_rs, dtype=float).ravel()
                rr_t = rr_rs[1:]
                rr_tm = rr_rs[:-1]
                ax1.scatter(rr_t, rr_tm, label='RS', color='blue', alpha=0.5)
                ax1.set_title(f"RS - Subject {recording.subject_id}")
                ax1.set_ylim(bottom=0, top=2000)
                ax1.set_xlim(left=0, right=2000)
                ax1.set_xlabel("RR Interval (t)")
                ax1.set_ylabel("RR Interval (t-1)")
                ax1.axhline(0, color='black', lw=0.5, ls='--')
                ax1.axvline(0, color='black', lw=0.5, ls='--')
                ax1.set_xlim(left=0)
                ax1.set_ylim(bottom=0)
                ax1.legend()
            else:
                ax1.set_title(f"RS - Subject {recording.subject_id} (No Data)")
            if rr_session is not None and len(rr_session) >= 3:
                rr_session = np.asarray(rr_session, dtype=float).ravel()
                rr_t = rr_session[1:]
                rr_tm = rr_session[:-1]
                ax2.scatter(rr_t, rr_tm, label='Session', color='orange', alpha=0.5)
                ax2.set_title(f"Session - Subject {recording.subject_id}")
                ax2.set_ylim(bottom=0, top=2500)
                ax2.set_xlim(left=0, right=2500)
                ax2.set_xlabel("RR Interval (t)")
                ax2.set_ylabel("RR Interval (t-1)")
                ax2.axhline(0, color='black', lw=0.5, ls='--')
                ax2.axvline(0, color='black', lw=0.5, ls='--')
                ax2.set_xlim(left=0)
                ax2.set_ylim(bottom=0)
                ax2.legend()
            else:
                ax2.set_title(f"Session - Subject {recording.subject_id} (No Data)")
        
        plt.suptitle(f"Poincare Maps - Session {self.session_id}, Family {self.family_id}, Seance {self.seance_id}")
        plt.show()
        if self.verbose:
            print(f"Poincare maps plotted for session {self.session_id}, family {self.family_id}, seance {self.seance_id}.")

        if return_fig:
            return fig
        return None
    

