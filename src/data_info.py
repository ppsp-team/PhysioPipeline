
# This file contains classes to represent metadata and data information for various physiological signals.
# It includes classes for PPG, ECG, EDA, and transcript data, along with a data loader for reading from Excel and JSON files.

#Import necessary libraries

import json
import os
import pickle
import pandas as pd
from datetime import datetime
from pathlib import Path

# Sensor Data Information Class
# ────────────────────────────────────────────────
class SensorDataInfo:
    def __init__(self):
        self.nb_channels: int = 0
        self.sampling_rate: float = 0.0

        self.file_path: Path = None
        
        self.rs_nb_samples: int = 0
        self.rs_duration: int = 0

        self.session_nb_samples: int = 0
        self.session_duration: int = 0

    def set_rs_duration(self, duration: datetime):
        self.rs_duration = duration

    def set_session_duration(self, duration: datetime):
        self.session_duration = duration

    def get_rs_duration(self) -> datetime:
        return self.rs_duration

    def get_session_duration(self) -> datetime:
        return self.session_duration

    def set_sampling_rate(self, sampling_rate: float):
        self.sampling_rate = sampling_rate

    def get_sampling_rate(self) -> float:
        return self.sampling_rate

    def set_nb_channels(self, nb_channels: int):
        self.nb_channels = nb_channels

    def get_nb_channels(self) -> int:
        return self.nb_channels

    def set_rs_nb_samples(self, nb_samples: int):
        self.rs_nb_samples = nb_samples

    def set_session_nb_samples(self, nb_samples: int):
        self.session_nb_samples = nb_samples

    def get_rs_nb_samples(self) -> int:
        return self.rs_nb_samples

    def get_session_nb_samples(self) -> int:
        return self.session_nb_samples

    def set_file_path(self, file_path: Path):
        self.file_path = file_path

    def get_file_path(self) -> Path:
        return self.file_path

    def to_dict(self) -> dict:
        return {
            "nb_channels": self.nb_channels,
            "sampling_rate": self.sampling_rate,
            "file_path": str(self.file_path) if self.file_path else None,
            "rs_nb_samples": self.rs_nb_samples,
            "rs_duration": self.rs_duration,
            "session_nb_samples": self.session_nb_samples,
            "session_duration": self.session_duration
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])

    def __str__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"nb_channels={self.nb_channels}, "
                f"sampling_rate={self.sampling_rate}, "
                f"file_path={self.file_path}, "
                f"rs_nb_samples={self.rs_nb_samples}, "
                f"rs_duration={self.rs_duration}, "
                f"session_nb_samples={self.session_nb_samples}, "
                f"session_duration={self.session_duration})")

# Metadata Class
# ────────────────────────────────────────────────

class Metadata:
    date : datetime = None
    misc: str = ""

    def __init__(self, date: datetime = None, misc: str = ""):
        if date is None:
            date = datetime.now()
        self.date = date
        self.misc = misc

    def set_date(self, date: datetime):
        self.date = date

    def get_date(self) -> datetime:
        return self.date

    def set_misc(self, misc: str):
        self.misc = misc

    def get_misc(self) -> str:
        return self.misc
        
    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat() if self.date else None,
            "misc": self.misc
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])

    def __str__(self) -> str:
        return f"Metadata(date={self.date.isoformat() if self.date else None}, misc='{self.misc}')"

    def description(self) -> str:
        return f"""
        Metadata class represents the metadata for sensor data.
        Date: {self.date.isoformat() if self.date else None}
        Miscellaneous Information: {self.misc if self.misc else "No additional information provided"}
        """


# Individual Data Info Classes
# ────────────────────────────────────────────────

class PPGDataInfo:
    def __init__(self):
        
        super().__init__()

    def description(self) -> str:
	    return f"""
        PPGDataInfo class represents the metadata for PPG (Photoplethysmography) data.
        {super().description()}
        """

class ECGDataInfo:
    def __init__(self):
        
        super().__init__()

    def description(self) -> str:
	    return f"""
        ECGDataInfo class represents the metadata for PPG (Photoplethysmography) data.
        {super().description()}
        """


class EDADataInfo:
    def __init__(self):
        
        super().__init__()
        
    def description(self) -> str:
	    return f"""
        EDADataInfo class represents the metadata for EDA (Electrodermal Activity) data.
        {super().description()}
        """

class TranscriptDataInfo:
    def __init__(self, language: str = 'fr', roles: str = [], file_path: Path = None):
        if language not in ['fr', 'en']:
            raise ValueError("Language must be either 'fr' or 'en'.")
        self.language = language
        self.roles = roles if roles else []
        self.file_path = file_path

    def __str__(self) -> str:
        if not self.roles:
            roles_str = "No roles defined"
        else:
            roles_str = ", ".join(self.roles)
        if self.file_path is None:
            file_path_str = "No file path provided"
        else:
            file_path_str = str(self.file_path)
        return (f"TranscriptDataInfo(language={self.language}, "
                f"roles=[{roles_str}], "
                f"file_path={file_path_str})")

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "roles": self.roles,
            "file_path": str(self.file_path) if self.file_path else None
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])

    def set_language(self, language: str):
        if language not in ['fr', 'en']:
            raise ValueError("Language must be either 'fr' or 'en'.")
        self.language = language

    def get_language(self) -> str:
        return self.language

    def set_roles(self, roles: str):
        if not isinstance(roles, list):
            raise ValueError("Roles must be a list of strings.")
        self.roles = roles

    def get_roles(self) -> str:
        return self.roles

    def set_file_path(self, file_path: Path):
        if not isinstance(file_path, Path):
            raise ValueError("File path must be a Path object.")
        self.file_path = file_path

    def get_file_path(self) -> Path:
        return self.file_path if self.file_path else None

    def description(self) -> str:
        roles_str = ", ".join(self.roles) if self.roles else "No roles defined"
        return f"""
        TranscriptDataInfo class represents the metadata for transcript data.
        Language: {self.language}
        Roles: [{roles_str}]
        File path: {self.file_path}
        """

# Combined Data Info Handler
#────────────────────────────────────────────────

class DataInfo:
    def __init__(self):
        self.hasPPG = False
        self.hasECG = False
        self.hasEDA = False
        self.hasTranscript = False

        self.ppgDataInfo = PPGDataInfo()
        self.ecgDataInfo = ECGDataInfo()
        self.edaDataInfo = EDADataInfo()
        self.transcriptDataInfo = TranscriptDataInfo()

    def set_ppg_data_info(self, ppg_data_info: PPGDataInfo):
        if not isinstance(ppg_data_info, PPGDataInfo):
            raise ValueError("ppg_data_info must be an instance of PPGDataInfo.")
        self.ppgDataInfo = ppg_data_info
        self.hasPPG = True

    def set_ecg_data_info(self, ecg_data_info: ECGDataInfo):
        if not isinstance(ecg_data_info, ECGDataInfo):
            raise ValueError("ecg_data_info must be an instance of ECGDataInfo.")
        self.ecgDataInfo = ecg_data_info
        self.hasECG = True

    def set_eda_data_info(self, eda_data_info: EDADataInfo):
        if not isinstance(eda_data_info, EDADataInfo):
            raise ValueError("eda_data_info must be an instance of EDADataInfo.")
        self.edaDataInfo = eda_data_info
        self.hasEDA = True

    def set_transcript_data_info(self, transcript_data_info: TranscriptDataInfo):
        if not isinstance(transcript_data_info, TranscriptDataInfo):
            raise ValueError("transcript_data_info must be an instance of TranscriptDataInfo.")
        self.transcriptDataInfo = transcript_data_info
        self.hasTranscript = True

    def get_ppg_data_info(self) -> PPGDataInfo:
        return self.ppgDataInfo

    def get_ecg_data_info(self) -> ECGDataInfo:
        return self.ecgDataInfo

    def get_eda_data_info(self) -> EDADataInfo:
        return self.edaDataInfo

    def get_transcript_data_info(self) -> TranscriptDataInfo:
        return self.transcriptDataInfo

    def to_dict(self) -> dict:
        return self.get_data_info()

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)

    def to_dataframe(self) -> pd.DataFrame:
        data = {
            "hasPPG": self.hasPPG,
            "hasECG": self.hasECG,
            "hasEDA": self.hasEDA,
            "hasTranscript": self.hasTranscript,
            "ppgDataInfo": self.ppgDataInfo.to_dict(),
            "ecgDataInfo": self.ecgDataInfo.to_dict(),
            "edaDataInfo": self.edaDataInfo.to_dict(),
            "transcriptDataInfo": self.transcriptDataInfo.to_dict()
        }
        return pd.DataFrame([data])

    def description(self) -> str:
        return f"""
        DataInfo class contains metadata for various physiological signals.
        PPG Data: {self.hasPPG}
        ECG Data: {self.hasECG}
        EDA Data: {self.hasEDA}
        Transcript Data: {self.hasTranscript}
        
        PPG Data Info: {self.ppgDataInfo.description()}
        ECG Data Info: {self.ecgDataInfo.description()}
        EDA Data Info: {self.edaDataInfo.description()}
        Transcript Data Info: {self.transcriptDataInfo.description()}
        """
    

class TranscriptData :
    def __init__(self):
        self.data = None


# Example usage

def main():
    dataloader = Dataloader()
    dataloader.loadFromExcel("data.xlsx")
    
    print(dataloader.info.GetDataInfo())
    print(dataloader.PPGData().head(10))

    with open("output.pkl", "w") as f:
        dataloader.saveDataToFile(f)

if __name__ == "__main__":
    main()