"""
Session class for managing physio recordings in a session.
"""


import sys
sys.path.append('../src/')

from physio_recording import PhysioRecording
from typing import List

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
        if not isinstance(session_id, int) or session_id <= 0:
            raise ValueError("session_id must be a positive integer")
        if not isinstance(family_id, int) or family_id <= 0:
            raise ValueError("family_id must be a positive integer")
        if not isinstance(seance_id, int) or seance_id <= 0:
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

    

