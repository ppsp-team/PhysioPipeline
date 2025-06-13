import sys
sys.path.append('../src/')

from physio_recording import PhysioRecording

class Session:

    seance_id: int
    family_id: int
    session_id: int
    verbose: bool = True
    physio_recordings: [PhysioRecording] = []


    def __init__(self, session_id: int, family_id: int, seance_id: int, verbose: bool = True):
        
        self.session_id = session_id
        self.seance_id = seance_id
        self.family_id = family_id
        self.verbose = verbose

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

    def extract_physio_features(self):
        """Extract physio features for all physio recordings in the session."""
        if self.verbose:
            print(f"Extracting features for physio recordings in session {self.session_id}, family {self.family_id}, seance {self.seance_id}...")
        for recording in self.physio_recordings:
            recording.extract_features()

    

