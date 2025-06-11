import pandas as pd
from .data_container import DataContainer


class Subject:
    def __init__(self, subject_id, subject_name,data_container=None,file_path=None):
        """
        Initializes a Subject instance.
        :param subject_id: Unique identifier for the subject.
        :param subject_name: Name of the subject.
        :param data_container: Optional data container for storing subject-related data.
        :param file_path: Optional file path for storing subject-related data.
        """
        self.data_container = data_container
        self.file_path = file_path
        self.subject_id = subject_id
        self.subject_name = subject_name
        
        if not isinstance(subject_id, int):
            raise TypeError("subject_id must be an integer")
        if not isinstance(subject_name, str):
            raise TypeError("subject_name must be a string")
        if data_container is not None and not isinstance(data_container, pd.DataFrame):
            raise TypeError("data_container must be a pandas DataFrame")
        if file_path is not None and not isinstance(file_path, str):
            raise TypeError("file_path must be a string")
        
    def __repr__(self):
        return f"Subject(id={self.subject_id}, name={self.subject_name}), data_container={self.data_container is not None}, file_path={self.file_path})"
    def __str__(self):
        return f"Subject ID: {self.subject_id}, Name: {self.subject_name}, Data Container: {self.data_container is not None}, File Path: {self.file_path}"

    

class Dyad:
    def __init__(self, subject1, subject2, subject3):
        """
        Initializes a Dyad instance.
        :param subject1: First subject in the dyad.
        :param subject2: Second subject in the dyad.
        :param subject3: Third subject in the dyad.
        """
        if not isinstance(subject1, Subject):
            raise TypeError("subject1 must be an instance of Subject")
        if not isinstance(subject2, Subject):
            raise TypeError("subject2 must be an instance of Subject")
        if not isinstance(subject3, Subject):
            raise TypeError("subject3 must be an instance of Subject")
        
        self.subject1 = subject1
        self.subject2 = subject2
        self.subject3 = subject3 
        self.subjects = [subject1, subject2, subject3]

    def __repr__(self):
        return f"Dyad(subject1={self.subject1}, subject2={self.subject2}, subject3={self.subject3})"
    def __str__(self):
        return f"Dyad: {self.subject1}, {self.subject2}, {self.subject3}"
    def get_subjects(self):
        return self.subjects
    def get_subject_by_id(self, subject_id):
        for subject in self.subjects:
            if subject.subject_id == subject_id:
                return subject
        raise ValueError(f"Subject with ID {subject_id} not found in the dyad.")
    

class Cohort:
    def __init__(self, cohort_name, dyads=None):
        """
        Initializes a Cohort instance.
        :param cohort_name: Name of the cohort.
        :param dyads: Optional list of Dyad instances in the cohort.
        """
        if not isinstance(cohort_name, str):
            raise TypeError("cohort_name must be a string")
        if dyads is not None and not all(isinstance(dyad, Dyad) for dyad in dyads):
            raise TypeError("dyads must be a list of Dyad instances")
        
        self.cohort_name = cohort_name
        self.dyads = dyads if dyads is not None else []

    def __repr__(self):
        return f"Cohort(name={self.cohort_name}, dyads_count={len(self.dyads)})"
    def __str__(self):
        return f"Cohort Name: {self.cohort_name}, Number of Dyads: {len(self.dyads)}"
    def get_dyads(self):
        return self.dyads
    def get_dyad_by_subject_id(self, subject_id):
        for dyad in self.dyads:
            if any(subject.subject_id == subject_id for subject in dyad.get_subjects()):
                return dyad
        raise ValueError(f"Dyad with subject ID {subject_id} not found in the cohort.")
 