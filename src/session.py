import sys
sys.path.append('../src/')

from physio_recording import PhysioRecording
from typing import List
class Session:

    def __init__(
        self,
        session_id: int,
        family_id: int,
        seance_id: int,
        verbose: bool = True,
    ):
        self.session_id = session_id
        self.family_id = family_id
        self.seance_id = seance_id
        self.verbose = verbose

        self.physio_recordings: List[PhysioRecording] = []
        
    
    def add_physio_recording(self, physio_recording: PhysioRecording):
        """Add a PhysioRecording to the session."""
        self.physio_recordings.append(physio_recording)

    def load_physio_recordings_data(self):
        """Load all physio recordings for the session."""
        if self.verbose:
            print(f"Loading physio recordings for session {self.session_id}, family {self.family_id}, seance {self.seance_id}...")
        for recording in self.physio_recordings:
            recording.load_raw_data()

    def process_physio_recordings(self):
        """Process all physio recordings for the session."""
        if self.verbose:
            print(f"Processing physio recordings for session {self.session_id}, family {self.family_id}, seance {self.seance_id}...")
        for recording in self.physio_recordings:
            recording.process_raw_data()

    def epoch_physio_recordings(self, method: str = 'fixed_duration', **kwargs):
        """Epoch all physio recordings for the session."""
        if self.verbose:
            print(f"Epoching physio recordings for session {self.session_id}, family {self.family_id}, seance {self.seance_id}...")
        for recording in self.physio_recordings:
            recording.epoch_processed_signals(method=method, **kwargs)

    

