
# This file contains classes to represent metadata and data information for various physiological signals.
# It includes classes for PPG, ECG, EDA, and transcript data, along with a data loader for reading from Excel and JSON files.

#Import necessary libraries

import json
import os
import pickle
import pandas as pd
from datetime import datetime

# Sensor Data Information Class
# ────────────────────────────────────────────────
class SensorDataInfo:
    def __init__(self, sampling_rate=0, duration=None, nb_channels=0, nb_samples=0, file_path=None, description=""):
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.nb_channels = nb_channels
        self.nb_samples = nb_samples
        self.file_path = file_path
        self.description = description

    def set_duration(self, duration):
        self.duration = duration

    def get_duration(self):
        return self.duration

    def __str__(self):
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




# Individual Data Info Classes
# ────────────────────────────────────────────────

class PPGDataInfo:
    def __init__(SensorDataInfo):
    def description(self):
	    pass


class ECGDataInfo:
    def __init__(SensorDataInfo):
        pass

class EDADataInfo:
    def __init__(SensorDataInfo):
        pass
class TranscriptDataInfo:
    def __init__(self, language='en', format='text', file_path=None):
        self.language = language
        self.format = format
        self.file_path = file_path

    def __str__(self):
        return f"TranscriptDataInfo(language={self.language}, format={self.format})"
	

# Combined Data Info Handler
#────────────────────────────────────────────────

class DataInfo:
    def __init__(self, hasPPG=False, hasECG=False, hasEDA=False, hasTranscript=False):
        self.hasPPG = hasPPG
        self.hasECG = hasECG
        self.hasEDA = hasEDA
        self.hasTranscript = hasTranscript

        self.ppgDataInfo = PPGDataInfo()
        self.ecgDataInfo = ECGDataInfo()
        self.edaDataInfo = EDADataInfo()
        self.transcriptDataInfo = TranscriptDataInfo()

    def get_data_info(self):
        return {
            "hasPPG": self.hasPPG,
            "hasECG": self.hasECG,
            "hasEDA": self.hasEDA,
            "hasTranscript": self.hasTranscript,
            "ppgDataInfo": str(self.ppgDataInfo),
            "ecgDataInfo": str(self.ecgDataInfo),
            "edaDataInfo": str(self.edaDataInfo),
            "transcriptDataInfo": str(self.transcriptDataInfo)
        }
    

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