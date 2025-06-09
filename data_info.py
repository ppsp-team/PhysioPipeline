
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
    def __init__(self, sampling_rate: float = 0, duration: datetime = None, nb_channels: int = 0, nb_samples: int = 0, file_path: Path = None):
        if duration is None:
            duration = datetime.now()
        self.duration = duration
        self.nb_channels = nb_channels
        self.nb_samples = nb_samples
        self.file_path = file_path

    def set_duration(self, duration: datetime):
        self.duration = duration

    def get_duration(self) -> datetime:
        return self.duration

    def set_sampling_rate(self, sampling_rate: float):
        self.sampling_rate = sampling_rate

    def get_sampling_rate(self) -> float:
        return self.sampling_rate

    def set_nb_channels(self, nb_channels: int):
        self.nb_channels = nb_channels

    def get_nb_channels(self) -> int:
        return self.nb_channels

    def set_nb_samples(self, nb_samples: int):
        self.nb_samples = nb_samples

    def get_nb_samples(self) -> int:
        return self.nb_samples

    def set_file_path(self, file_path: Path):
        self.file_path = file_path

    def get_file_path(self) -> Path:
        return self.file_path

    def to_dict(self) -> dict:
        return {
            "sampling_rate": self.sampling_rate,
            "duration": self.duration,
            "nb_channels": self.nb_channels,
            "nb_samples": self.nb_samples,
            "file_path": self.file_path,
            "description": self.description
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])

    def __str__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"sampling_rate={self.sampling_rate}, "
                f"duration={self.duration}, "
                f"nb_channels={self.nb_channels}, "
                f"nb_samples={self.nb_samples}, "
                f"file_path={self.file_path}, "
                f"description='{self.description}')")

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
    def __init__(self, sampling_rate: float = 0, duration: datetime = None, 
                 nb_channels: int = 0, nb_samples: int = 0, 
                 file_path: Path = None, date=None, description: str = ""):
        
        super().__init__(sampling_rate, duration, nb_channels, nb_samples, file_path)

    def description(self) -> str:
	    return f"""
        PPGDataInfo class represents the metadata for PPG (Photoplethysmography) data.
        Date: {self.date.isoformat() if self.date else None}
        Sampling Rate: {self.sampling_rate if self.sampling_rate else 0.0}
        Duration: {self.duration.isoformat() if self.duration else None}
        Number of Channels: {self.nb_channels if self.nb_channels else 0}
        Number of Samples: {self.nb_samples if self.nb_samples else 0}
        File Path: {self.file_path if self.file_path else "No file path provided"}
        Description: {self.description if self.description else "No description provided"}
        """

class ECGDataInfo:
    def __init__(self, sampling_rate: float = 0, duration: datetime = None, 
                 nb_channels: int = 0, nb_samples: int = 0, 
                 file_path: Path = None, date=None, description: str = ""):
        
        super().__init__(sampling_rate, duration, nb_channels, nb_samples, file_path)

    def description(self) -> str:
	    return f"""
        ECGDataInfo class represents the metadata for ECG (Electrocardiography) data.
        Date: {self.date.isoformat() if self.date else None}
        Sampling Rate: {self.sampling_rate if self.sampling_rate else 0.0}
        Duration: {self.duration.isoformat() if self.duration else None}
        Number of Channels: {self.nb_channels if self.nb_channels else 0}
        Number of Samples: {self.nb_samples if self.nb_samples else 0}
        File Path: {self.file_path if self.file_path else "No file path provided"}
        Description: {self.description if self.description else "No description provided"}
        """

class EDADataInfo:
    def __init__(self, sampling_rate: float = 0, duration: datetime = None, 
                 nb_channels: int = 0, nb_samples: int = 0, 
                 file_path: Path = None, date=None, description: str = ""):
        
        super().__init__(sampling_rate, duration, nb_channels, nb_samples, file_path)
        
    def description(self) -> str:
	    return f"""
        EDADataInfo class represents the metadata for EDA (Electrodermal Activity) data.
        Date: {self.date.isoformat() if self.date else None}
        Sampling Rate: {self.sampling_rate if self.sampling_rate else 0.0}
        Duration: {self.duration.isoformat() if self.duration else None}
        Number of Channels: {self.nb_channels if self.nb_channels else 0}
        Number of Samples: {self.nb_samples if self.nb_samples else 0}
        File Path: {self.file_path if self.file_path else "No file path provided"}
        Description: {self.description if self.description else "No description provided"}
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
    
# Data Classes
# ────────────────────────────────────────────────  
class PPGData :
    def __init__(self):
        self.data = None


class ECGData :
    def __init__(self):
        self.data = None

class EDAData :
    def __init__(self):
        self.data = None

class TranscriptData :
    def __init__(self):
        self.data = None

	
# Data Loader
# ────────────────────────────────────────────────

class Dataloader:
    def __init__(self): 
        self.info = DataInfo()
        self.ppgData = None
        self.ecgData = None
        self.edaData = None
        self.transcriptData = None

    def load_from_excel(self, file_path):
        df = pd.read_excel('example.xlsx')
        print(df)
        print("Data loaded from Excel file.")

    def load_from_json(self, file_path):
        df = pd.read_json(file_path)
        print(df)
        print("Data loaded from JSON file.")

    def load(self, file_type=None, file_path=None):
        if file_type == "excel":
            self.load_from_excel(file_path)
        elif file_type == "json":
            self.load_from_json(file_path)
        else:
            raise ValueError("Unsupported file type. Use 'excel' or 'json'.")

    def load_data_from_file(self, file_path):
        if file_path.endswith('.xlsx'):
            self.load_from_excel(file_path)
        elif file_path.endswith('.json'):
            self.load_from_json(file_path)
        else:
            raise ValueError("Unsupported file format. Please use .xlsx or .json files.")


    def save_data_to_file(self, file_path):
        if file_path.endswith('.xlsx'):
            self.info.to_excel(file_path)
            print(f"Data saved to {file_path}")
        elif file_path.endswith('.json'):
            self.info.to_json(file_path)
            print(f"Data saved to {file_path}")
        else:
            raise ValueError("Unsupported file format. Please use .xlsx or .json files.")

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