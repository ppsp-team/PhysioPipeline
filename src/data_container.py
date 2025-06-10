import pandas as pd

class DataContainer():
    rs = None
    session = None
    def __init__(self):
        pass

    def setRSData(self, rs):
        if not isinstance(rs, pd.DataFrame):
            raise ValueError("rs must be a pandas DataFrame.")
        self.rs = rs

    def getRSData(self):
        return self.rs

    def setSessionData(self, session):
        if not isinstance(session, pd.DataFrame):
            raise ValueError("session must be a pandas DataFrame.")
        self.session = session

    def getSessionData(self):
        return self.session


# Data Classes
# ────────────────────────────────────────────────  
class PPGDataContainer(DataContainer):
    def __init__(self):
        super().__init__()

class ECGDataContainer(DataContainer):
    def __init__(self):
        super().__init__()

class EDADataContainer(DataContainer):
    def __init__(self):
        super().__init__()
