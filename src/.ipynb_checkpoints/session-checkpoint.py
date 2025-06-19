import sys
sys.path.append('../src/')

from physio_recording import PhysioRecording

class Session:

    seance_id: int
    family_id: int
    session_id: int
    physio_recordings: [PhysioRecording] = []


    def __init__(self, session_id: int, family_id: int, seance_id: int):
        self.session_id = session_id
        self.seance_id = seance_id
        self.family_id = family_id

    def add_physio_recording(self, physio_recording: PhysioRecording):
        """Add a PhysioRecording to the session."""
        self.physio_recordings.append(physio_recording)

    def load_physio_recordings_data(self):
        """Load all physio recordings for the session."""
        for recording in self.physio_recordings:
            recording.load_raw_data()

    def process_physio_recordings(self):
        """Process all physio recordings for the session."""
        for recording in self.physio_recordings:
            recording.process_data()

    

