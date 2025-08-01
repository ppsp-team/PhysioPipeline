import sys
sys.path.append('../src/')

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter
import pandas as pd



class NSTE_Session_Analysis:
    """
    Efficient NSTE (Normalized Symbolic Transfer Entropy) analysis class.
    """
    
    def __init__(self, verbose: bool = False):
        if not isinstance(verbose, bool):
            raise ValueError("verbose must be a boolean value")
            
        self.session = None
        self.features: Dict[str, Any] = {}
        self.verbose = verbose
        
        # NSTE parameters
        self.fs: int = None
        self.win_size_sec: int = None
        self.win_step_sec: int = None
        self.dim: int = None
        self.tau: int = None
        self.num_iter: int = None
        
        print("NSTE_Session_Analysis initialized.") if self.verbose else None

    def set_session(self, session: Session):
        """Set the session for NSTE and initialize features structure."""
        if not isinstance(session, Session):
            raise ValueError("session must be an instance of Session")
        if session is None:
            raise ValueError("session cannot be None")
        
        self.session = session
        
        # Initialize features structure
        self.features = {
            "subjects": {},
            "dyads": {}
        }
        
        # Initialize individual subject features and extract EDA signals
        for physio_recording in session.physio_recordings:
            subject_id = physio_recording.subject_id
            self.features["subjects"][subject_id] = {
                "EDA_Signal": {
                    "rs": physio_recording.eda["epochs"]["rs"]["signal"] if "epochs" in physio_recording.eda else physio_recording.eda["signal"],
                    "session": physio_recording.eda["epochs"]["session"]["signal"] if "epochs" in physio_recording.eda else physio_recording.eda["signal"]
                },
                "EDA_Timestamps": {
                    "rs": physio_recording.eda["epochs"]["rs"]["timestamps"] if "epochs" in physio_recording.eda else physio_recording.eda["timestamps"],
                    "session": physio_recording.eda["epochs"]["session"]["timestamps"] if "epochs" in physio_recording.eda else physio_recording.eda["timestamps"]
                }
            }
        
        # Initialize dyadic features
        subject_ids = [rec.subject_id for rec in session.physio_recordings]
        for i in range(len(subject_ids)):
            for j in range(i + 1, len(subject_ids)):
                dyad_id = f"{subject_ids[i]}_{subject_ids[j]}"
                self.features["dyads"][dyad_id] = {
                    "NSTE_XY": {"rs": {}, "session": {}},
                    "NSTE_YX": {"rs": {}, "session": {}},
                    "STE_XY": {"rs": {}, "session": {}},
                    "STE_YX": {"rs": {}, "session": {}}
                }
        
        print(f"Session {session.session_id} set with {len(session.physio_recordings)} recordings.") if self.verbose else None

    def get_session(self):
        """Get the current session."""
        if self.session is None:
            raise ValueError("Session has not been set")
        return self.session

    def set_parameters(self, fs: int, win_size_sec: int, win_step_sec: int,
                      dim: int, tau: int, num_iter: int):
        """Set NSTE parameters."""
        if not all(isinstance(arg, int) and arg > 0 for arg in [fs, win_size_sec, win_step_sec, dim, tau, num_iter]):
            raise ValueError("All NSTE parameters must be positive integers.")
        
        self.fs = fs
        self.win_size_sec = win_size_sec
        self.win_step_sec = win_step_sec
        self.dim = dim
        self.tau = tau
        self.num_iter = num_iter
        
        print(f"Parameters set: fs={fs}, dim={dim}, tau={tau}") if self.verbose else None
    

    def _delay_reconstruction(self, data: np.ndarray, lag: int, dim: int) -> np.ndarray:
        """Reconstruct phase space using time delay embedding."""
        data = np.asarray(data).flatten()
        n_points = len(data)
        
        if n_points < lag * (dim - 1) + 1:
            return np.array([])
        
        max_epoch = n_points - lag * (dim - 1)
        embedded = np.zeros((max_epoch, dim))
        
        for j in range(dim):
            embedded[:, j] = data[j * lag:j * lag + max_epoch]
        
        return embedded

    def _symbolize(self, data: np.ndarray) -> np.ndarray:
        """Convert data to ordinal patterns."""
        if data.ndim != 2:
            return np.array([])
        
        n_points, dim = data.shape
        symbols = np.zeros(n_points)
        
        for i in range(n_points):
            ranks = np.argsort(np.argsort(data[i, :]))
            symbols[i] = sum(rank * (dim ** j) for j, rank in enumerate(ranks))
        
        return symbols.astype(int)

    def _estimate_probabilities(self, *symbol_arrays: np.ndarray) -> List[float]:
        """Estimate probabilities from symbol sequences."""
        if not symbol_arrays:
            return []
        
        min_len = min(len(arr) for arr in symbol_arrays)
        if min_len == 0:
            return []
            
        symbol_arrays = [arr[:min_len] for arr in symbol_arrays]
        joint_symbols = list(zip(*symbol_arrays))
        counts = Counter(joint_symbols)
        total = len(joint_symbols)
        
        return [count / total for count in counts.values()]

    def _shannon_entropy(self, probabilities: List[float]) -> float:
        """Calculate Shannon entropy."""
        probabilities = [p for p in probabilities if p > 0]
        if not probabilities:
            return 0.0
        return -sum(p * np.log2(p) for p in probabilities)

    def _calculate_nste_single_window(self, x_win: np.ndarray, y_win: np.ndarray) -> Tuple[float, float, float, float]:
        """Calculate NSTE for a single window pair."""
        x_win = np.asarray(x_win).flatten()
        y_win = np.asarray(y_win).flatten()
        
        min_required_len = self.tau * (self.dim - 1) + 1
        if len(x_win) < min_required_len or len(y_win) < min_required_len:
            return np.nan, np.nan, np.nan, np.nan
        
        # Delay reconstruction
        x_embedded = self._delay_reconstruction(x_win, self.tau, self.dim)
        y_embedded = self._delay_reconstruction(y_win, self.tau, self.dim)
        
        if x_embedded.size == 0 or y_embedded.size == 0:
            return np.nan, np.nan, np.nan, np.nan
        
        # Ensure equal lengths
        min_len_emb = min(len(x_embedded), len(y_embedded))
        x_embedded = x_embedded[:min_len_emb]
        y_embedded = y_embedded[:min_len_emb]
        
        # Symbolization
        sx = self._symbolize(x_embedded)
        sy = self._symbolize(y_embedded)
        
        if sx.size == 0 or sy.size == 0 or min(len(sx), len(sy)) < 2:
            return np.nan, np.nan, np.nan, np.nan
        
        # Prepare time series
        min_len_sym = min(len(sx), len(sy))
        sx_curr = sx[:min_len_sym-1]
        sx_next = sx[1:min_len_sym]
        sy_curr = sy[:min_len_sym-1]
        sy_next = sy[1:min_len_sym]
        
        # Calculate transfer entropy Y -> X
        p_xnext_x_y = self._estimate_probabilities(sx_next, sx_curr, sy_curr)
        p_xnext_x = self._estimate_probabilities(sx_next, sx_curr)
        p_x_y = self._estimate_probabilities(sx_curr, sy_curr)
        p_x = self._estimate_probabilities(sx_curr)
        
        h_xnext_x_y = self._shannon_entropy(p_xnext_x_y)
        h_xnext_x = self._shannon_entropy(p_xnext_x)
        h_x_y = self._shannon_entropy(p_x_y)
        h_x = self._shannon_entropy(p_x)
        
        ste_yx = h_xnext_x + h_x_y - h_xnext_x_y - h_x
        
        # Calculate transfer entropy X -> Y
        p_ynext_y_x = self._estimate_probabilities(sy_next, sy_curr, sx_curr)
        p_ynext_y = self._estimate_probabilities(sy_next, sy_curr)
        p_y_x = self._estimate_probabilities(sy_curr, sx_curr)
        p_y = self._estimate_probabilities(sy_curr)
        
        h_ynext_y_x = self._shannon_entropy(p_ynext_y_x)
        h_ynext_y = self._shannon_entropy(p_ynext_y)
        h_y_x = self._shannon_entropy(p_y_x)
        h_y = self._shannon_entropy(p_y)
        
        ste_xy = h_ynext_y + h_y_x - h_ynext_y_x - h_y
        
        # Normalize
        p_xnext = self._estimate_probabilities(sx_next)
        p_ynext = self._estimate_probabilities(sy_next)
        h_xnext = self._shannon_entropy(p_xnext)
        h_ynext = self._shannon_entropy(p_ynext)
        
        nste_yx = ste_yx / h_xnext if h_xnext > 0 else 0
        nste_xy = ste_xy / h_ynext if h_ynext > 0 else 0
        
        # Ensure non-negative
        return max(0, ste_yx), max(0, ste_xy), max(0, nste_yx), max(0, nste_xy)

    def compute_nste_pair(self, signal1: np.ndarray, signal2: np.ndarray) -> Dict[str, Any]:
        """Compute NSTE for a pair of signals over sliding windows."""
        if any(param is None for param in [self.fs, self.win_size_sec, self.win_step_sec, self.dim, self.tau]):
            raise ValueError("NSTE parameters must be set before computing.")

        win_size_samples = int(self.win_size_sec * self.fs)
        win_step_samples = int(self.win_step_sec * self.fs)
        n_samples = min(len(signal1), len(signal2))

        n_windows = (n_samples - win_size_samples) // win_step_samples + 1
        
        results = {
            'ste_yx': [], 'ste_xy': [], 'nste_yx': [], 'nste_xy': [], 'timestamps': []
        }

        for i in range(n_windows):
            start = i * win_step_samples
            end = start + win_size_samples

            x_win = signal1[start:end]
            y_win = signal2[start:end]

            ste_yx, ste_xy, nste_yx, nste_xy = self._calculate_nste_single_window(x_win, y_win)

            results['ste_yx'].append(ste_yx)
            results['ste_xy'].append(ste_xy)
            results['nste_yx'].append(nste_yx)
            results['nste_xy'].append(nste_xy)
            results['timestamps'].append(start / self.fs)

        return results

    def compute_dyadic_nste(self):
        """Compute NSTE features for all dyads in the session."""
        if self.session is None:
            raise ValueError("Session has not been set")
        
        if any(param is None for param in [self.fs, self.win_size_sec, self.dim, self.tau]):
            raise ValueError("NSTE parameters have not been set. Call set_parameters() first.")
        
        print("Computing dyadic NSTE features...") if self.verbose else None
        
        for dyad_id in self.features["dyads"]:
            print(f"Processing dyad {dyad_id}...") if self.verbose else None
            
            # Extract subject IDs
            subject_ids = dyad_id.split("_")
            sid1, sid2 = subject_ids[0], subject_ids[1]
            
            # Get EDA signals for both subjects
            eda1_rs = self.features["subjects"][sid1]["EDA_Signal"]["rs"]
            eda1_session = self.features["subjects"][sid1]["EDA_Signal"]["session"]
            eda2_rs = self.features["subjects"][sid2]["EDA_Signal"]["rs"]
            eda2_session = self.features["subjects"][sid2]["EDA_Signal"]["session"]
            
            # Process each epoch type (rs, session)
            for step, (eda1_container, eda2_container) in [("rs", (eda1_rs, eda2_rs)), ("session", (eda1_session, eda2_session))]:
                print(f"  Processing {step} epoch...") if self.verbose else None
                
                # Process each epoch within the step
                if isinstance(eda1_container, dict):
                    epochs_to_process = eda1_container.items()
                else:
                    epochs_to_process = [("single", eda1_container)]
                
                for epoch_id, eda1_data in epochs_to_process:
                    # Get corresponding data from second subject
                    if isinstance(eda2_container, dict):
                        eda2_data = eda2_container.get(epoch_id)
                    else:
                        eda2_data = eda2_container
                    
                    if eda1_data is None or eda2_data is None:
                        print(f"    Warning: Missing data for epoch {epoch_id}") if self.verbose else None
                        continue
                    
                    # Convert to numpy arrays
                    x_signal = np.asarray(eda1_data, dtype=float).flatten()
                    y_signal = np.asarray(eda2_data, dtype=float).flatten()
                    
                    # Ensure equal lengths
                    min_len = min(len(x_signal), len(y_signal))
                    x_signal = x_signal[:min_len]
                    y_signal = y_signal[:min_len]
                    
                    # Check if we have enough data
                    win_size_samples = self.win_size_sec * self.fs
                    if min_len < win_size_samples:
                        print(f"    Warning: Not enough data in epoch {epoch_id}") if self.verbose else None
                        self.features["dyads"][dyad_id]["NSTE_XY"][step][epoch_id] = np.nan
                        self.features["dyads"][dyad_id]["NSTE_YX"][step][epoch_id] = np.nan
                        self.features["dyads"][dyad_id]["STE_XY"][step][epoch_id] = np.nan
                        self.features["dyads"][dyad_id]["STE_YX"][step][epoch_id] = np.nan
                        continue
                    
                    # Calculate NSTE for the entire epoch (or use sliding windows if preferred)
                    ste_yx, ste_xy, nste_yx, nste_xy = self._calculate_nste_single_window(x_signal, y_signal)
                    
                    # Store results
                    self.features["dyads"][dyad_id]["NSTE_XY"][step][epoch_id] = nste_xy
                    self.features["dyads"][dyad_id]["NSTE_YX"][step][epoch_id] = nste_yx
                    self.features["dyads"][dyad_id]["STE_XY"][step][epoch_id] = ste_xy
                    self.features["dyads"][dyad_id]["STE_YX"][step][epoch_id] = ste_yx
                    
                    print(f"Epoch {epoch_id}: NSTE_XY={nste_xy:.4f}, NSTE_YX={nste_yx:.4f}") if self.verbose else None
        
        print("Dyadic NSTE features computed.") if self.verbose else None

    def get_results(self) -> Dict[str, Any]:
        """Return the computed NSTE features."""
        if not self.features:
            raise ValueError("NSTE features have not been computed yet. Call compute_dyadic_nste() first.")
        return self.features.copy()

    def get_dyad_results(self, dyad_id: str) -> Dict[str, Any]:
        """Get results for a specific dyad."""
        if dyad_id not in self.features.get("dyads", {}):
            raise ValueError(f"Dyad {dyad_id} not found in results")
        return self.features["dyads"][dyad_id].copy()

    def get_nste_summary(self) -> Dict[str, Any]:
        """Get summary statistics of NSTE results."""
        summary = {"dyads": {}, "overall": {}}
        all_metrics = {'nste_yx': [], 'nste_xy': [], 'asymmetry': []}
        
        for dyad_key, dyad_data in self.features["dyads"].items():
            summary["dyads"][dyad_key] = {}
            
            for condition in ["rs", "session"]:
                if condition in dyad_data["NSTE_YX"] and condition in dyad_data["NSTE_XY"]:
                    # Collect all valid NSTE values from epochs
                    valid_yx = []
                    valid_xy = []
                    
                    for epoch_id, nste_yx_val in dyad_data["NSTE_YX"][condition].items():
                        if not np.isnan(nste_yx_val):
                            valid_yx.append(nste_yx_val)
                    
                    for epoch_id, nste_xy_val in dyad_data["NSTE_XY"][condition].items():
                        if not np.isnan(nste_xy_val):
                            valid_xy.append(nste_xy_val)
                    
                    if valid_yx and valid_xy:
                        asymmetry = [y - x for y, x in zip(valid_yx, valid_xy)]
                        
                        summary["dyads"][dyad_key][condition] = {
                            "mean_nste_yx": np.mean(valid_yx),
                            "std_nste_yx": np.std(valid_yx),
                            "mean_nste_xy": np.mean(valid_xy),
                            "std_nste_xy": np.std(valid_xy),
                            "mean_asymmetry": np.mean(asymmetry),
                            "std_asymmetry": np.std(asymmetry)
                        }
                        
                        # Collect for overall summary
                        all_metrics['nste_yx'].extend(valid_yx)
                        all_metrics['nste_xy'].extend(valid_xy)
                        all_metrics['asymmetry'].extend(asymmetry)
        
        # Overall summary
        if all_metrics['nste_yx'] and all_metrics['nste_xy']:
            summary["overall"] = {
                "mean_nste_yx": np.mean(all_metrics['nste_yx']),
                "std_nste_yx": np.std(all_metrics['nste_yx']),
                "mean_nste_xy": np.mean(all_metrics['nste_xy']),
                "std_nste_xy": np.std(all_metrics['nste_xy']),
                "mean_asymmetry": np.mean(all_metrics['asymmetry']),
                "std_asymmetry": np.std(all_metrics['asymmetry'])
            }
        
        return summary
    
    def plot_nste_dynamics(asymmetry, nste_yx_list, nste_xy_list, win_step_sec):
        """
        Plot NSTE dynamics using given asymmetry and NSTE values.
        Parameters:
        - asymmetry: List of asymmetry values over time.
        - nste_yx_list: List of NSTE Y→X values over time.
        - nste_xy_list: List of NSTE X→Y values over time.
        - win_step_sec: The window step size in seconds for computing time array.
        """
        time = np.linspace(0, len(asymmetry) * win_step_sec, len(asymmetry))
    
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
        # Plot NSTE Asymmetry
        axes[0].plot(time, asymmetry, label='NSTE Asymmetry', color='purple', marker='o', linewidth=2)
        axes[0].axhline(0, linestyle='--', color='gray', linewidth=1)
        axes[0].set(title='NSTE Asymmetry Over Time', xlabel='Time (s)', ylabel='Asymmetry')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_facecolor('#fafafa')
    
        # Plot NSTE Dynamics (Y→X and X→Y)
        axes[1].plot(time, nste_yx_list, label='NSTE Y→X', color='green', marker='o', linewidth=2)
        axes[1].plot(time, nste_xy_list, label='NSTE X→Y', color='red', marker='o', linewidth=2)
        axes[1].set(title='NSTE Dynamics Over Time', xlabel='Time (s)', ylabel='NSTE Value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_facecolor('#fafafa')
    
        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))







