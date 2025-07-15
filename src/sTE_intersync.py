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
        
        # NSTE parameters
       self.fs: int = None
       self.win_size_sec: int = None
       self.win_step_sec: int = None
       self.dim: int = None
       self.tau: int = None
       self.num_iter: int = None
        
        # Computed results
       self.results: Dict[str, Any] = {}
       self.verbose = verbose

       if self.verbose:
        print("NSTE_Analysis initialized.")

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
             print(f"DEBUG: n_points: {n_points}, n_channels: {n_channels}, min_required: {min_required}")

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

        print(f"  x_win length: {len(x_sample_window)}")
        print(f"  y_win length: {len(y_sample_window)}")
        print(f"  (Required min length for embedding with dim={test_dim}, tau={test_tau}: {min_len_needed})")

        # Probability calculation 

        p_xnext_x_y = self._estimate_probabilities(sx_next, sx_curr, sy_curr)
        p_xnext_x = self._estimate_probabilities(sx_next, sx_curr)
        p_x_y = self._estimate_probabilities(sx_curr, sy_curr)
        p_x = self._estimate_probabilities(sx_curr)

        py_next = sy[1 : min_len_joint + 1]
        py_curr = sy[0 : min_len_joint]
        px_curr = sx[0 : min_len_joint]

        p_ynext_y_x = self._estimate_probabilities(py_next, py_curr, px_curr)
        p_ynext_y = self._estimate_probabilities(py_next, py_curr)
        p_y_x = self._estimate_probabilities(py_curr, px_curr)
        p_y = self._estimate_probabilities(py_curr)
        
        h_xnext_x_y = self._shannon_entropy(p_xnext_x_y)
        h_xnext_x = self._shannon_entropy(p_xnext_x)
        h_x_y = self._shannon_entropy(p_x_y)
        h_x = self._shannon_entropy(p_x)

        h_ynext_y_x = self._shannon_entropy(p_ynext_y_x)
        h_ynext_y = self._shannon_entropy(p_ynext_y)
        h_y_x = self._shannon_entropy(p_y_x)
        h_y = self._shannon_entropy(p_y)

        ste_yx = h_xnext_x + h_x_y - h_xnext_x_y - h_x
        ste_xy = h_ynext_y + h_y_x - h_ynext_y_x - h_y

        p_xnext = self._estimate_probabilities(sx_next)
        h_xnext = self._shannon_entropy(p_xnext)
        
        p_ynext = self._estimate_probabilities(py_next)
        h_ynext = self._shannon_entropy(p_ynext)

        nste_yx = ste_yx / h_xnext if h_xnext > 0 else 0
        nste_xy = ste_xy / h_ynext if h_ynext > 0 else 0
        
        ste_yx = max(0, ste_yx)
        ste_xy = max(0, ste_xy)
        nste_yx = max(0, nste_yx)
        nste_xy = max(0, nste_xy)

        print("\n--- Results from _calculate_nste_single_window ---")
        print(f"STE Y->X: {ste_yx:.4f}")
        print(f"STE X->Y: {ste_xy:.4f}")
        print(f"NSTE Y->X: {nste_yx:.4f}")
        print(f"NSTE X->Y: {nste_xy:.4f}")
        
    
        return ste_yx, ste_xy, nste_yx,nste_xy


    def run(self):

        """
        Run the full NSTE analysis: prepare windows, compute NSTE values, and compile results.
        Raises:
            ValueError: If parameters have not been set.
    """
        
        if self.verbose:
            print("\nRunning NSTE analysis...")

        self.results = {} # Clear previous results on a new run

        # Step 1: Validate parameters and prepare data windows
        try:
            self._validate_and_prepare_windows()
            if not self._analysis_successful:
                # If window preparation failed (e.g., signal too short, no windows formed),
                # _analysis_successful will be False and results already set with an error.
                if self.verbose:
                    print("NSTE run terminated early due to windowing issues.")
                return 
        except ValueError as e:
            raise ValueError(f"Parameter or data validation error: {e}") from e

        # Step 2: Compute observed and shuffled NSTE values, and p-values
        self._compute_nste_values()
        
        # Step 3: Compile all results into the final dictionary
        self._compile_final_results()

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



  



        

# --- ADAPTED EXAMPLE USAGE (using randomly generated numbers with simplified NSTE_Analysis) ---
if __name__ == "__main__":
    # 1. Generate random time series data
    num_samples = 1000
    t = np.linspace(1, 100, num_samples) 
    fs = int(num_samples / (t[-1] - t[0])) 

    X_data = np.random.randn(num_samples)
    Y_data = np.random.randn(num_samples)
    
    print(f"Generated random data: X (len={len(X_data)}), Y (len={len(Y_data)}), fs={fs}Hz")

    lags = [1, 2, 3, 4, 5]

    max_nste_yx = -np.inf
    max_nste_xy = -np.inf
    best_tau_yx = None
    best_tau_xy = None
    
    fixed_dim = 3
    num_shuffles = 100

    full_signal_duration = t[-1] - t[0] + (1/fs) 
    win_size_for_full_signal = int(np.ceil(full_signal_duration)) 
    
    for current_tau in lags:
        print(f"\n--- Analyzing for tau = {current_tau} ---")
        
        # MODIFIED: Create NSTE_Analysis instance, passing data directly
        # verbose=True here to see the new debug prints within the class
        engine = NSTE_Analysis(
            x_signal=X_data, 
            y_signal=Y_data, 
            timestamps=t, 
            window_size=1, # Dummy value, will be set by set_parameters
            lag=1,         # Dummy value, will be set by set_parameters
            verbose=True # Set to True for more detailed internal debugging messages
        ) 
        
        # 4. Set parameters for a single large window
        engine.set_parameters(
            fs=fs,
            win_size_sec=win_size_for_full_signal, 
            win_step_sec=win_size_for_full_signal, 
            dim=fixed_dim,
            tau=current_tau,
            num_iter=num_shuffles
        )
        
        # 5. Run the NSTE Analysis (now called 'run')
        try:
            engine.run() # Call the new 'run' method
            
            results = engine.get_results()
            
            if 'error' in results:
                print(f"  Skipping NSTE for tau {current_tau} due to error: {results['error']}")
                continue 

            # Accessing results
            ste_yx_val = results['ste_yx'][0]
            ste_xy_val = results['ste_xy'][0]
            nste_yx_val = results['nste_yx'][0]
            nste_xy_val = results['nste_xy'][0]
            
            pval_yx_val = results['pval_YX'][0]
            pval_xy_val = results['pval_XY'][0]
            
            print(f"  STE Y->X for tau {current_tau}: {ste_yx_val:.4f}")
            print(f"  NSTE Y->X for tau {current_tau}: {nste_yx_val:.4f} (p={pval_yx_val:.4f})")
            print(f"  STE X->Y for tau {current_tau}: {ste_xy_val:.4f}")
            print(f"  NSTE X->Y for tau {current_xy_val:.4f} (p={pval_xy_val:.4f})")

            if nste_yx_val > max_nste_yx:
                max_nste_yx = nste_yx_val
                best_tau_yx = current_tau
            if nste_xy_val > max_nste_xy:
                max_nste_xy = nste_xy_val
                best_tau_xy = current_tau

        except ValueError as e:
            print(f"  Analysis Error (ValueError) for tau {current_tau}: {e}")
        except Exception as e:
            print(f"  An unexpected error occurred for tau {current_tau}: {type(e).__name__}: {e}")

    print("\n--- Summary of Maximum NSTE across Lags ---")
    print(f"Maximum NSTE Y->X found: {max_nste_yx:.4f} (at tau={best_tau_yx})")
    print(f"Maximum NSTE X->Y found: {max_nste_xy:.4f} (at tau={best_tau_xy})")






    # Initialize the NSTE_Analysis object
nste_analyzer = NSTE_Analysis(verbose=True)

# Manually mock the internal structure expected after set_session
nste_analyzer.features = {
    "subjects": {
        "A": {
            "EDA_Signal": {
                "rs": np.random.rand(1000),  # simulated signal
                "session": np.random.rand(1000)
            },
            "EDA_Timestamps": {
                "rs": np.linspace(0, 100, 1000),
                "session": np.linspace(0, 100, 1000)
            }
        },
        "B": {
            "EDA_Signal": {
                "rs": np.random.rand(1000),
                "session": np.random.rand(1000)
            },
            "EDA_Timestamps": {
                "rs": np.linspace(0, 100, 1000),
                "session": np.linspace(0, 100, 1000)
            }
        }
    },
    "dyads": {
        "A_B": {
            "NSTE_XY": {"rs": {}, "session": {}},
            "NSTE_YX": {"rs": {}, "session": {}},
            "STE_XY": {"rs": {}, "session": {}},
            "STE_YX": {"rs": {}, "session": {}}
        }
    }
}

# Set NSTE parameters (you can modify these for experiments)
nste_analyzer.set_parameters(
    fs=10,              # sampling frequency
    win_size_sec=10,    # window size in seconds
    win_step_sec=5,     # window step size
    dim=3,              # embedding dimension
    tau=2,              # time delay
    num_iter=1          # number of iterations (for shuffling, if implemented)
)

# Manually invoke compute_dyadic_nste()
nste_analyzer.compute_dyadic_nste()

# Retrieve and print results
results = nste_analyzer.get_results()
dyad_results = nste_analyzer.get_dyad_results("A_B")

print("\n--- NSTE Results for A_B ---")
print("NSTE_XY (rs):", dyad_results["NSTE_XY"]["rs"])
print("NSTE_YX (rs):", dyad_results["NSTE_YX"]["rs"])
print("STE_XY (rs):", dyad_results["STE_XY"]["rs"])
print("STE_YX (rs):", dyad_results["STE_YX"]["rs"])