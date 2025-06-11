import pandas as pd
from .data_container import DataContainer


class subject:
    def __init__(self, subject_id, subject_name, data_container=None, file_path=None):
        # Validate first, then assign to PRIVATE attributes
        if not isinstance(subject_id, int):
            raise TypeError("subject_id must be an integer")
        if not isinstance(subject_name, str):
            raise TypeError("subject_name must be a string")
        if data_container is not None and not isinstance(data_container, pd.DataFrame):
            raise TypeError("data_container must be a pandas DataFrame")
        if file_path is not None and not isinstance(file_path, str):
            raise TypeError("file_path must be a string")
            
        # Use private attributes to avoid recursion
        self._subject_id = subject_id
        self._subject_name = subject_name
        self._data_container = data_container
        self._file_path = file_path
    
    # Properties that work correctly
    @property
    def subject_id(self):
        return self._subject_id  
    
    @subject_id.setter  
    def subject_id(self, value):
        if not isinstance(value, int):
            raise TypeError("subject_id must be an integer")
        self._subject_id = value  # Set private attribute
    
    @property
    def subject_name(self):
        return self._subject_name
    
    @subject_name.setter 
    def subject_name(self, value):
        if not isinstance(value, str):
            raise TypeError("subject_name must be a string")
        self._subject_name = value
    
    @property
    def data_container(self):
        return self._data_container
    
    @data_container.setter 
    def data_container(self, value):
        if value is not None and not isinstance(value, pd.DataFrame):
            raise TypeError("data_container must be a pandas DataFrame")
        self._data_container = value
    
    @property
    def file_path(self):
        return self._file_path
    
    @file_path.setter 
    def file_path(self, value):
        if value is not None and not isinstance(value, str):
            raise TypeError("file_path must be a string")
        self._file_path = value
    
    def add_subject_data(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        
        if self._data_container is None:
            self._data_container = data.copy()
        else:
            self._data_container = pd.concat([self._data_container, data], ignore_index=True)
        
        if self._file_path is not None:
            self._data_container.to_csv(self._file_path, index=False)
        return self._data_container
    
    def remove_subject(self, row_id): 
        if not isinstance(row_id, int):
            raise TypeError("row_id must be an integer")
        if self._data_container is not None:
            self._data_container = self._data_container.drop(row_id, errors='ignore')
            if self._file_path is not None:
                self._data_container.to_csv(self._file_path, index=False)
        else:
            raise ValueError("No data container available to remove subject data.")
    
    def filter_subject_data(self, condition):
        if self._data_container is None:
            raise ValueError("No data container available for filtering.")
        if not callable(condition):
            raise TypeError("condition must be callable")
        return self._data_container[self._data_container.apply(condition, axis=1)]
    
    def load_subject_data(self, file_path):
        if not isinstance(file_path, str):
            raise TypeError("file_path must be a string")
        self._file_path = file_path
        self._data_container = pd.read_csv(file_path)
        return self._data_container
    
    def __repr__(self):
        return f"Subject(id={self._subject_id}, name='{self._subject_name}')"
    
    def __str__(self):
        data_info = f"{len(self._data_container)} rows" if self._data_container is not None else "No data"
        return f"Subject {self._subject_id}: {self._subject_name} ({data_info})"


class Dyad(subject):
    subjects = []
    
    def __init__(self, subject1, subject2,subject3=None, file_path=None,self_data_container=None):
        """
        Initializes a Dyad instance with two subjects.
        :param subject1: First subject, must be an instance of Subject.
        :param
 subject2: Second subject, must be an instance of Subject.
        :param subject3: Optional third subject, must be an instance of Subject.
        :param file_path: Optional file path for saving data.
        """ 
        if not isinstance(subject1, subject):
            raise TypeError("subject1 must be an instance of subject")
        if not isinstance(subject2, subject):
            raise TypeError("subject2 must be an instance of subject")
        if subject3 is not None and not isinstance(subject3, subject):
            raise TypeError("subject3 must be an instance of subject")
        if file_path is not None and not isinstance(file_path, str):
            raise TypeError("file_path must be a string")
        if subject3 is not None:
            self.subjects = [subject1, subject2, subject3]

        self.subjects = [subject1, subject2]
        self.file_path = file_path
        self.data_container = None
        if file_path is not None:
            self.data_container = pd.DataFrame()
        super().__init__(subject1.subject_id, subject1.subject_name, self.data_container, file_path)    

    @property
    def subject1(self):
        return self.subjects[0]
    @subject1.setter
    def subject1(self, value):
        if not isinstance(value, subject):
            raise TypeError("subject1 must be an instance of subject")
        self.subjects[0] = value

    @property
    def subject2(self):
        return self.subjects[1]
    @subject2.setter
    def subject2(self, value):
        if not isinstance(value, subject):
            raise TypeError("subject2 must be an instance of subject")
        self.subjects[1] = value

    @property
    def subject3(self):
        return self.subjects[2] if len(self.subjects) > 2 else None

    @subject3.setter
    def subject3(self, value):
        if not isinstance(value, subject):
            raise TypeError("subject3 must be an instance of Subject")
        if len(self.subjects) == 2:
            self.subjects.append(value)
        else:
            self.subjects[2] = value
    
    def add_dyad_data(self, data):
        """
        Adds data to the dyad's data container.
        :param data: Data to be added, must be a pandas DataFrame.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        
        if self.data_container is None:
            self.data_container = data.copy()
        else:
            self.data_container = pd.concat([self.data_container, data], ignore_index=True)
        
        if self.file_path is not None:
            self.data_container.to_csv(self.file_path, index=False)
        return self.data_container
    
    def remove_dyad(self, row_id):
        """
        Removes a row from the dyad's data container.
        :param row_id: Row index to be removed, must be an integer.
        """
        if not isinstance(row_id, int):
            raise TypeError("row_id must be an integer")
        if self.data_container is not None:
            self.data_container = self.data_container.drop(row_id, errors='ignore')
            if self.file_path is not None:
                self.data_container.to_csv(self.file_path, index=False)
        else:
            raise ValueError("No data container available to remove dyad data.")
        return self.data_container
    
    def __repr__(self):
        return f"Dyad(subject1={self.subject1.subject_id}, subject2={self.subject2.subject_id})"

    
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
 





















