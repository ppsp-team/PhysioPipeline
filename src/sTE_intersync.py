import numpy as np
from collections import Counter
from scipy.signal import correlate
import logging as log

import pandas as pd
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt



class NSTE_Analysis:

    def __init__(self, verbose=True):
       
       if not isinstance(verbose, bool):
            raise ValueError("verbose must be a boolean value")
       
       # Data storage
       self.x_signal: np.ndarray = None
       self.y_signal: np.ndarray = None
       self.timestamps: np.ndarray = None

        # Parameters
       self.fs = None
       self.win_size_sec = None
       self.win_step_sec = None
       self.dim = None
       self.tau = None
       self.num_iter = None
        
        
        # Computed results
       self.results: Dict[str, Any] = {}
       self.verbose = verbose

       if self.verbose:
        print("NSTE_Analysis initialized.")


    def set_signals(self, x_signal, y_signal, timestamps=None):
     """
     Set the signals to analyze
     """
     self.x_signal = np.asarray(x_signal)
     self.y_signal = np.asarray(y_signal)
     self.timestamps = timestamps

    def set_parameters(self, fs: int, win_size_sec: int, win_step_sec: int,
                       dim: int, tau: int, num_iter: int):
        """
        Sets the parameters for NSTE calculation.
        """
        if not all(isinstance(arg, int) and arg > 0 for arg in [fs, win_size_sec, win_step_sec, dim, tau, num_iter]):
            raise ValueError("All NSTE parameters (fs, win_size_sec, win_step_sec, dim, tau, num_iter) must be positive integers.")
        
        self.fs = fs
        self.win_size_sec = win_size_sec
        self.win_step_sec = win_step_sec
        self.dim = dim
        self.tau = tau
        self.num_iter = num_iter

        if self.verbose:
            print(f"NSTE parameters set: fs={fs}, win_size_sec={win_size_sec}, win_step_sec={win_step_sec}, dim={dim}, tau={tau}, num_iter={num_iter}")



    def _delay_reconstruction(self, data: np.ndarray, lag: int, dim: int) -> np.ndarray:
        """
        Reconstruct phase space using time delay embedding.
        Adjusted to handle 1D input data for single channel embedding.
        """
        if data.ndim == 1:
            data = data[:, np.newaxis] # Make it 2D (n_points, 1)

        n_points, n_channels = data.shape
        if n_points < lag * (dim - 1) + 1:

            if self.verbose:
             print(f"DEBUG: n_points: {n_points}, n_channels: {n_channels}")

            if self.verbose:
                print(f"Warning (_delay_reconstruction): Not enough data points ({n_points}) for reconstruction with lag {lag} and dim {dim}. Expected at least {lag * (dim - 1) + 1}. Returning empty array.")
            return np.array([]) # Return empty array if not enough data

        max_epoch = n_points - lag * (dim - 1)
        embedded = np.zeros((max_epoch, dim, n_channels))
        
        if self.verbose:
            print(f"DEBUG: Created embedded array shape: {embedded.shape}")
            print(f"DEBUG: max_epoch: {max_epoch}")


        for c in range(n_channels):
            for j in range(dim):
                start_idx = j * lag
                end_idx = n_points - (dim - 1 - j) * lag
                embedded[:, j, c] = data[start_idx:end_idx, c]

        if self.verbose and c == 0:  # Only print for first channel 
                print(f"DEBUG: dim {j}: start_idx={start_idx}, end_idx={end_idx}, slice_len={end_idx-start_idx}")


        final_shape = embedded.squeeze().shape

        if self.verbose:
         print(f"DEBUG: Final output shape after squeeze: {final_shape}")

        return embedded.squeeze()

    def _symbolize(self, data: np.ndarray) -> np.ndarray:
        """
        Convert data to ordinal patterns (symbolic representation).
        Expected data shape: (n_points, dim)
        """
        if data.ndim == 1:
            if self.verbose:
                print("Warning (_symbolize): Cannot symbolize 1D data directly for ordinal patterns. Returning empty array.")
            return np.array([])
        
        n_points, dim = data.shape
        symbols = np.zeros(n_points)
        
        print(f"Symbols shape: {symbols.shape}")

        for i in range(n_points):
            ranks = np.argsort(np.argsort(data[i, :]))
            symbol_val = 0 
            for j, rank in enumerate(ranks):
                symbol_val += rank * (dim ** j)
            symbols[i] = symbol_val

        print(f"DEBUG: rank {j}: symbol_val={symbol_val}")

        return symbols.astype(int)

    def _estimate_probabilities(self, *symbol_arrays: np.ndarray) -> list[float]:
        """
        Estimate joint and marginal probabilities from symbol sequences.
        """
        if not symbol_arrays:
            return []
        
        lengths = [len(arr) for arr in symbol_arrays]
        if not all(l == lengths[0] for l in lengths):
            if self.verbose:
                print("Warning (_estimate_probabilities): Symbol arrays have different lengths. Taking minimum length.")
            min_len = min(lengths)
            symbol_arrays = [arr[:min_len] for arr in symbol_arrays]
            
        joint_symbols = list(zip(*symbol_arrays))
        
        counts = Counter(joint_symbols)
        total = len(joint_symbols)
        if total == 0: return []
        
        probabilities = {symbol : count / total for symbol , count in counts.items()}

        if self.verbose:
          for i, arr in enumerate(symbol_arrays):
            unique_symbols = np.unique(arr)
            symbol_counts = Counter(arr)
            print(f"DEBUG: Array {i} - unique symbols: {unique_symbols}, counts: {dict(symbol_counts)}")
    
        
        return list(probabilities.values())

    

    def _shannon_entropy(self, probabilities: list[float]) -> float:
        """Calculate Shannon entropy from a list of probabilities."""
        probabilities = [p for p in probabilities if p > 0]
        if not probabilities: return 0.0
        return -sum(p * np.log2(p) for p in probabilities)

    def _calculate_nste_single_window(self, x_win: np.ndarray, y_win: np.ndarray) -> tuple[float, float, float, float]:
        """
        Calculates Symbolic Transfer Entropy for a single window.
        This is the internal, specific calculation.
        """
        x_win = np.asarray(x_win).flatten()
        y_win = np.asarray(y_win).flatten()

        min_required_len = self.tau * (self.dim - 1) + 1
        if len(x_win) < min_required_len or len(y_win) < min_required_len:
            if self.verbose:
                print(f"Warning (_calculate_ste_single_window): Window too short ({len(x_win)}/{len(y_win)} samples) for embedding with dim={self.dim}, tau={self.tau}. Expected at least {min_required_len}. Returning NaN for STE.")
            return np.nan, np.nan, np.nan, np.nan

        x_embedded = self._delay_reconstruction(x_win, self.tau, self.dim)
        y_embedded = self._delay_reconstruction(y_win, self.tau, self.dim)
        
        if x_embedded.size == 0 or y_embedded.size == 0:
            return np.nan, np.nan, np.nan, np.nan

        min_len_emb = min(len(x_embedded), len(y_embedded))
        x_embedded = x_embedded[:min_len_emb]
        y_embedded = y_embedded[:min_len_emb]

        sx = self._symbolize(x_embedded)
        sy = self._symbolize(y_embedded)

        if sx.size == 0 or sy.size == 0:
            return np.nan, np.nan, np.nan, np.nan
        
        min_len_sym = min(len(sx), len(sy))
        sx = sx[:min_len_sym]
        sy = sy[:min_len_sym]
        
        if min_len_sym < (self.dim + 1):
             if self.verbose:
                 print(f"Warning (_calculate_ste_single_window): Not enough symbolic data points ({min_len_sym}) after symbolization. Expected at least {self.dim + 1}. Returning NaN.")
             return np.nan, np.nan, np.nan, np.nan

        min_len_joint = min(len(sx) - 1, len(sy) - 1) #maximum number of aligned "current-next" pairs symbols for timeries x and y 
        
        if min_len_joint <= 0:
            if self.verbose:
                print(f"Warning (_calculate_ste_single_window): Not enough points for joint probability estimation ({min_len_joint} after alignment). Returning NaN.")
            return np.nan, np.nan, np.nan, np.nan
            
        sx_next = sx[1 : min_len_joint + 1]
        sx_curr = sx[0 : min_len_joint]
        sy_curr = sy[0 : min_len_joint]

        print(f"  x_win length: {len(x_win)}")
        print(f"  y_win length: {len(y_win)}")
        print(f"  (Required min length for embedding with dim={self.dim}, tau={self.tau}: {min_len_emb})")

        # Probability calculation 
        sx_next = sx[1 : min_len_joint + 1]
        sx_curr = sx[0 : min_len_joint]
        sy_next = sy[1 : min_len_joint + 1]
        sy_curr = sy[0 : min_len_joint]
        
        # Y → X
        p_xnext_x_y = self._estimate_probabilities(sx_next, sx_curr, sy_curr)
        p_xnext_x   = self._estimate_probabilities(sx_next, sx_curr)
        p_x_y       = self._estimate_probabilities(sx_curr, sy_curr)
        p_x         = self._estimate_probabilities(sx_curr)
        
        # X → Y
        p_ynext_y_x = self._estimate_probabilities(sy_next, sy_curr, sx_curr)
        p_ynext_y   = self._estimate_probabilities(sy_next, sy_curr)
        p_y_x       = self._estimate_probabilities(sy_curr, sx_curr)
        p_y         = self._estimate_probabilities(sy_curr)
        
        # Entropies
        h_xnext_x_y = self._shannon_entropy(p_xnext_x_y)
        h_xnext_x   = self._shannon_entropy(p_xnext_x)
        h_x_y       = self._shannon_entropy(p_x_y)
        h_x         = self._shannon_entropy(p_x)
        
        h_ynext_y_x = self._shannon_entropy(p_ynext_y_x)
        h_ynext_y   = self._shannon_entropy(p_ynext_y)
        h_y_x       = self._shannon_entropy(p_y_x)
        h_y         = self._shannon_entropy(p_y)
        
        # STE & NSTE
        ste_yx = max(0, h_xnext_x + h_x_y - h_xnext_x_y - h_x)
        ste_xy = max(0, h_ynext_y + h_y_x - h_ynext_y_x - h_y)
        
        h_xnext = self._shannon_entropy(self._estimate_probabilities(sx_next))
        h_ynext = self._shannon_entropy(self._estimate_probabilities(sy_next))
        
        nste_yx = max(0, ste_yx / h_xnext if h_xnext > 0 else 0)
        nste_xy = max(0, ste_xy / h_ynext if h_ynext > 0 else 0)

        print("\n--- Results from _calculate_nste_single_window ---")
        print(f"STE Y->X: {ste_yx:.4f}")
        print(f"STE X->Y: {ste_xy:.4f}")
        print(f"NSTE Y->X: {nste_yx:.4f}")
        print(f"NSTE X->Y: {nste_xy:.4f}")
        
    
        return ste_yx, ste_xy, nste_yx,nste_xy
    
    def compute_nste(self):
        """
        Computes Normalized Symbolic Transfer Entropy (NSTE) over multiple 
        sliding windows of the input signals. Results are stored in 
        self.results['nste'] and corresponding timestamps in self.results['timestamps'].
        """
        # Check that all required parameters are set
        required_params = [
        self.x_signal, self.y_signal, self.fs,
        self.win_size_sec, self.win_step_sec,
        self.dim, self.tau, self.num_iter
        ]
        if any(param is None for param in required_params):
         raise ValueError("Ensure all signals and parameters are set before computing NSTE.")

        win_size_samples = int(self.win_size_sec * self.fs)
        win_step_samples = int(self.win_step_sec * self.fs)
        n_samples = min(len(self.x_signal), len(self.y_signal))

        n_windows = (n_samples - win_size_samples) // win_step_samples + 1
 
        if self.verbose:
            print(f"Computing NSTE over {n_windows} windows...")

        nste_values = []
        timestamps = []

        for i in range(n_windows):
            start = i * win_step_samples
            end = start + win_size_samples

            x_win = self.x_signal[start:end]
            y_win = self.y_signal[start:end]

            try:
                nste_val = self._calculate_nste_single_window(x_win, y_win)
            except Exception as e:
                nste_val = np.nan
                if self.verbose:
                    print(f"Window {i}: Error - {e}")

            nste_values.append(nste_val)
            timestamps.append(start / self.fs)

            if self.verbose:
               print(f"Window {i + 1}/{n_windows}: NSTE = {nste_val}")

        self.results['nste'] = nste_values
        self.results['timestamps'] = timestamps


    def run(self):

        """
        Run the full NSTE analysis: prepare windows, compute NSTE values, and compile results.
        Raises:
            ValueError: If parameters have not been set.
    """
        
        if self.verbose:
            print("\nRunning NSTE analysis...")

        self.results = {} # Clear previous results on a new run

        
        

        # Step 2: Compute observed and shuffled NSTE values, and p-values
        self.compute_nste(self)
        
        


    def get_results(self) -> Dict[str, Any]:
        """
        Returns the computed NSTE results.
        Returns:
            Dict[str, Any]: A dictionary containing the NSTE results.
        Raises:
            ValueError: If results have not been computed yet.
        """
        if not self.results:
            raise ValueError("NSTE results have not been computed yet. Call run_analysis() first.")
        return self.results.copy()
    
    import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def unix_to_minutes(unix_timestamps: np.ndarray) -> np.ndarray:
    """Convert unix timestamps to minutes from start."""
    times = [datetime.fromtimestamp(ts) for ts in unix_timestamps]
    start = times[0]
    return np.array([(t - start).total_seconds() / 60 for t in times])

def plot_time_series(nste_time, nste_yx, nste_xy, asym_ave, figsize=(12, 6)):
    """Plot NSTE and Asymmetry metrics."""
    t = unix_to_minutes(nste_time)
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    fig, axes = plt.subplots(1, 2, figsize=(figsize[0]*2, figsize[1]))

    # Plot NSTE
    axes[0].plot(t, nste_yx, label='NSTE Y→X', color=colors[0])
    axes[0].plot(t, nste_xy, label='NSTE X→Y', color=colors[1])
    axes[0].set(title='NSTE', xlabel='Time (min)', ylabel='Value')
    axes[0].legend()

    # Plot Asymmetry (sampled)
    axes[1].plot(t[::5], asym_ave[::5], label='Asymmetry', color=colors[2], marker='o', linewidth=2)
    axes[1].axhline(0, linestyle='--', color='gray', linewidth=1)
    axes[1].set(title='Asymmetry (X-Y)', xlabel='Time (min)', ylabel='Value')

    for ax in axes:
        ax.grid(True)
        ax.set_facecolor('#fafafa')

    plt.tight_layout()
    plt.show()
    return fig






        

# --- ADAPTED EXAMPLE USAGE (using randomly generated numbers with simplified NSTE_Analysis) ---
if __name__ == "__main__":
    # Generate test data
    num_samples = 1000
    X_data = np.random.randn(num_samples)
    Y_data = np.random.randn(num_samples)
    
    # Test different tau values
    for tau in [1, 2, 3]:
        print(f"\n--- Testing tau = {tau} ---")
        
        # Create analyzer
        analyzer = NSTE_Analysis(verbose=True)
        
        # Set signals and parameters
        analyzer.set_signals(X_data, Y_data)  # You'd need to add this method
        analyzer.set_parameters(fs=10, win_size_sec=10,win_step_sec=2, dim=3, tau=tau, num_iter=1)
        
        # Test single window calculation
        ste_yx, ste_xy, nste_yx, nste_xy = analyzer._calculate_nste_single_window(X_data, Y_data)
        
        print(f"STE Y->X: {ste_yx:.4f}")
        print(f"NSTE Y->X: {nste_yx:.4f}")
        print(f"STE X->Y: {ste_xy:.4f}")
        print(f"NSTE X->Y: {nste_xy:.4f}")

    n = 1000
    t0 = 1640995200
    nste_time = np.linspace(t0, t0 + 3600, num_samples)
    x = np.linspace(0, 4*np.pi, num_samples)
    nste_yx = 0.3 + 0.2*np.sin(x) + 0.1*np.random.randn(num_samples)
    nste_xy = 0.25 + 0.15*np.cos(x*1.2) + 0.1*np.random.randn(num_samples)
    asym_ave = nste_yx - nste_xy + 0.05*np.random.randn(num_samples)

    plot_time_series(nste_time, nste_yx, nste_xy, asym_ave)