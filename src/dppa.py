import sys
sys.path.append('../src/')

from session import Session
import pandas as pd
from typing import Dict, Any
import numpy as np
import os
from pathlib import Path

class DPPA:

    def __init__(self):
        self.session: Session = None
        self.features: pd.DataFrame = None

        self.features: Dict[str, Any] = {}
        pass

    def set_session(self, session: Session):
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

    def get_session(self) -> Session:
        if self.session is None:
            raise ValueError("Session has not been set")
        return self.session
    
    def compute_individual_features(self):
        if self.session is None:
            raise ValueError("Session has not been set")

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


    def compute_icd(self):

        for sid1 in self.features["subjects"]:
            for sid2 in self.features["subjects"]:
                if sid1 < sid2:
                    for step in ["rs", "session"]:
                        for epoch_id, rr in _iter_epochs(self.features["subjects"][sid1]["Centroid"][step]):
                            c1 = self.features["subjects"][sid1]["Centroid"][step][epoch_id]
                            c2 = self.features["subjects"][sid2]["Centroid"][step][epoch_id]

                            icd = np.sqrt(np.sum((np.asarray(c1) - np.asarray(c2)) ** 2))
                            self.features["dyads"][f"{sid1}_{sid2}"]["ICD"][step][epoch_id] = icd
                            
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