import sys
sys.path.append('../src/')

from session import Session
import pandas as pd
from typing import Dict, Any
import numpy as np

class DPPA:

    def __init__(self, verbose: bool = True):
        if not isinstance(verbose, bool):
            raise ValueError("verbose must be a boolean value")

        self.session: Session = None
        self.features: pd.DataFrame = None
        self.clusters: Dict[str, Any] = {}

        self.features: Dict[str, Any] = {}
        self.verbose = verbose
        if self.verbose:
            print("DPPA initialized. Ready to set session and compute features.")

    def set_session(self, session: Session):
        """
        Set the session for DPPA and initialize features.
        Args:
            session (Session): The session to set.
        Raises:
            ValueError: If session is not an instance of Session or is None.
        """
        if not isinstance(session, Session):
            raise ValueError("session must be an instance of Session")
        if session is None:
            raise ValueError("session cannot be None")
        self.session = session

        self.features = {
            "subjects": {},
            "dyads": {},
        }

        for physio_recording1 in session.physio_recordings:
            self.features["subjects"][physio_recording1.subject_id] = {
                "RR_Intervals": {
                    "rs": physio_recording1.bvp["epochs"]["rs"]["RR_Intervals"],
                    "session": physio_recording1.bvp["epochs"]["session"]["RR_Intervals"]               
                }
            }
            for physio_recording2 in session.physio_recordings:
                if physio_recording1.subject_id < physio_recording2.subject_id:
                    self.features["dyads"][f"{physio_recording1.subject_id}_{physio_recording2.subject_id}"] = {
                        "ICD": {
                            "rs": _empty_metrics_like(physio_recording1.bvp["epochs"]["rs"]["RR_Intervals"]),
                            "session": _empty_metrics_like(physio_recording1.bvp["epochs"]["session"]["RR_Intervals"])
                        }
                    }
        if self.verbose:
            print(f"Session {session.session_id} set with {len(session.physio_recordings)} physio recordings. Features initialized.")

    def get_session(self) -> Session:
        """
        Get the current session.
        Returns:
            Session: The current session.
        Raises:
            ValueError: If session has not been set.
        """
        if self.session is None:
            raise ValueError("Session has not been set")
        return self.session
    
    def compute_individual_features(self):
        """
        Compute individual features for each subject in the session.
        Raises:
            ValueError: If session has not been set or if RR Intervals are not set for a subject.
        """
        if self.session is None:
            raise ValueError("Session has not been set")
        
        if self.verbose:
            print("Computing individual features for each subject...")

        for sid in self.features["subjects"]:
            rr_rs      = self.features["subjects"][sid]["RR_Intervals"]["rs"]
            rr_session = self.features["subjects"][sid]["RR_Intervals"]["session"]

            if rr_rs is None or rr_session is None:
                raise ValueError(f"RR Intervals for subject {sid} are not set")

            # Pre-allocate
            self.features["subjects"][sid]["Centroid"]      = {"rs": _empty_metrics_like(rr_rs), "session": _empty_metrics_like(rr_session)}
            self.features["subjects"][sid]["SD1"]           = {"rs": _empty_metrics_like(rr_rs), "session": _empty_metrics_like(rr_session)}
            self.features["subjects"][sid]["SD2"]           = {"rs": _empty_metrics_like(rr_rs), "session": _empty_metrics_like(rr_session)}
            self.features["subjects"][sid]["SD1_SD2_Ratio"] = {"rs": _empty_metrics_like(rr_rs), "session": _empty_metrics_like(rr_session)}

            # Compute metrics
            for step, rr_container in [("rs", rr_rs), ("session", rr_session)]:
                for epoch_id, rr_raw in _iter_epochs(rr_container):
                    rr = np.asarray(rr_raw, dtype=float).ravel()
                    if rr.size < 3:
                        raise ValueError(f"Need ≥3 RR intervals in epoch {epoch_id} (sid {sid})")

                    rr_t  = rr[1:]
                    rr_tm = rr[:-1]

                    centroid = [rr_t.mean(), rr_tm.mean()]
                    sd1 = np.std(rr_t - rr_tm, ddof=1) / np.sqrt(2.0)
                    sd2 = np.std(rr_t + rr_tm, ddof=1) / np.sqrt(2.0)
                    sd_ratio = sd1 / sd2 if sd2 else np.inf

                    self.features["subjects"][sid]["Centroid"][step][epoch_id]      = centroid
                    self.features["subjects"][sid]["SD1"][step][epoch_id]           = sd1
                    self.features["subjects"][sid]["SD2"][step][epoch_id]           = sd2
                    self.features["subjects"][sid]["SD1_SD2_Ratio"][step][epoch_id] = sd_ratio

            if self.verbose:
                print(f"\tComputed features for subject {sid}: Centroid, SD1, SD2, and SD1/SD2 Ratio.")
        
        if self.verbose:
            print("Individual features computed.")

    def compute_clusters(self, threshold: float = 50):
        """
        Compute clusters of subjects based on their Centroid features.
        Args:
            threshold (float): Distance threshold for clustering.
        Raises:
            ValueError: If session has not been set or if Centroid features are not set for subjects.
        """
        if self.session is None:
            raise ValueError("Session has not been set")
        if self.verbose:
            print("Computing clusters of subjects based on Centroid features...")

        clusters = {
            "2-clusters": {},
            "3-soft-clusters": {},
            "3-strong-clusters": {}
        }

        for dyad_id in self.features["dyads"]:
            clusters["2-clusters"][dyad_id] = 0
        clusters["3-soft-clusters"] = 0
        clusters["3-strong-clusters"] = 0

        n_epochs = len(next(iter(self.features["dyads"].values()))["ICD"]["session"])

        for n in range(n_epochs):
            dyads = list(self.features["dyads"].keys())

            is_cluster = [False] * len(dyads)
            for dyad in dyads:
                ICD = self.features["dyads"][dyad]["ICD"]["session"][n]
                if ICD < threshold:
                    is_cluster[dyads.index(dyad)] = True
            if sum(is_cluster) == 3:
                clusters["3-strong-clusters"] += 1
            elif sum(is_cluster) == 2:
                clusters["3-soft-clusters"] += 1
            elif sum(is_cluster) == 1:
                dyad = dyads[is_cluster.index(True)]
                clusters["2-clusters"][dyad] += 1

        # divide by number of epochs to get average counts
        for key in clusters:
            if isinstance(clusters[key], dict):
                for dyad in clusters[key]:
                    clusters[key][dyad] /= n_epochs
            else:
                clusters[key] /= n_epochs

        # Store clusters in features
        self.features["clusters"] = clusters

        if self.verbose:
            print("Clusters computed:")
            for key, value in clusters.items():
                print(f"\t{key}: {value}")
        


    def compute_icd(self):
        """
        Compute Inter-Centroid Distances (ICD) for all dyads in the session.
        Raises:
            ValueError: If session has not been set or if Centroid features are not set for dyads.
        """
        if self.session is None:
            raise ValueError("Session has not been set")
        if self.verbose:
            print("Computing Inter-Centroid Distances (ICD) for all dyads...")

        for sid1 in self.features["subjects"]:
            for sid2 in self.features["subjects"]:
                if sid1 < sid2:
                    for step in ["rs", "session"]:
                        for epoch_id, rr in _iter_epochs(self.features["subjects"][sid1]["Centroid"][step]):
                            c1 = self.features["subjects"][sid1]["Centroid"][step][epoch_id]
                            c2 = self.features["subjects"][sid2]["Centroid"][step][epoch_id]

                            icd = np.sqrt(np.sum((np.asarray(c1) - np.asarray(c2)) ** 2))
                            self.features["dyads"][f"{sid1}_{sid2}"]["ICD"][step][epoch_id] = icd
                    
                    if self.verbose:
                        print(f"\tComputed ICD for dyad ({sid1}, {sid2}) in both rs and session steps.")

        if self.verbose:
            print("ICD computed.")

    def run(self):
        """
        Run the full DPPA computation: set session, compute individual features, and compute ICD.
        Raises:
            ValueError: If session has not been set.
        """
        if self.session is None:
            raise ValueError("Session has not been set")
        
        if self.verbose:
            print("Running DPPA computation...")

        self.compute_individual_features()
        self.compute_icd()

        if self.verbose:
            print("DPPA computation completed.")

from typing import List, Tuple
                            
def _iter_epochs(container):
    """
    Yield (epoch_id, rr_array) pairs from either a dict or a list-like.
    Dict → keeps original keys; list-like → uses enumerate().
    """
    if isinstance(container, dict):
        return container.items()
    else:  # list, tuple, ndarray, pd.Series, etc.
        return enumerate(container)

def _empty_metrics_like(container):
    """Return an empty dict with matching hashable keys."""
    if isinstance(container, dict):
        return {k: None for k in container}
    else:
        return {i: None for i in range(len(container))}