import sys
sys.path.append('../src/')

from session import Session
import pandas as pd
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt


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
        self.clusters = clusters

        if self.verbose:
            print("Clusters computed:")
            for key, value in clusters.items():
                print(f"\t{key}: {value}")

    def plot_feature(self, feature_name: str, y_min: int = 0, y_max: int = 100, return_fig: bool = False):
        """
        Plot a feature for each subject in the session.
        Args:
            feature_name (str): The name of the feature to plot (e.g., "Centroid", "SD1", "SD2", "SD1_SD2_Ratio").
        Raises:
            ValueError: If session has not been set or if the feature is not set for subjects.
        """
        if self.session is None:
            raise ValueError("Session has not been set")
        
        if self.verbose:
            print(f"Plotting feature '{feature_name}' for all subjects in the session...")
        if feature_name not in ["Centroid", "SD1", "SD2", "SD1_SD2_Ratio"]:
            raise ValueError(f"Feature {feature_name} is not recognized. Available features: Centroid, SD1, SD2, SD1_SD2_Ratio.")

        for sid in self.features["subjects"]:

            feature_rs = self.features["subjects"][sid][feature_name]["rs"]
            if isinstance(feature_rs, dict):
                feature_rs = [feature_rs[k] for k in sorted(feature_rs.keys())]
            if isinstance(feature_rs, np.ndarray):
                feature_rs = feature_rs.tolist()

            feature_session = self.features["subjects"][sid][feature_name]["session"]
            if isinstance(feature_session, dict):
                feature_session = [feature_session[k] for k in sorted(feature_session.keys())]
            if isinstance(feature_session, np.ndarray):
                feature_session = feature_session.tolist()

            if feature_name == "Centroid":
                feature_rs = [np.linalg.norm(np.asarray(c)) for c in feature_rs]
                feature_session = [np.linalg.norm(np.asarray(c)) for c in feature_session]

            if feature_rs is None or feature_session is None:
                raise ValueError(f"Feature {feature_name} for subject {sid} is not set")

            plt.figure(figsize=(10, 5))

            if len(feature_rs) == 1:
                plt.axhline(y=feature_rs[0], color='r', linestyle='--', label=f'{feature_name} for Subject {sid} RS')
            else:
                plt.axhline(y=np.mean(feature_rs), color='r', linestyle='--', label='Mean Feature RS')

            plt.plot(feature_session, label=f'{feature_name} for Subject {sid} Session ', marker='x')

            # add a trend line for session SD1
            z = np.polyfit(range(len(feature_session)), feature_session, 1)
            p = np.poly1d(z)
            plt.plot(range(len(feature_session)), p(range(len(feature_session))), color='orange', linestyle='--', label=f'Trend Line {feature_name} Session')

            plt.ylim(y_min, y_max)
            plt.xlim(0, len(feature_session) - 1)
            plt.title(f'{feature_name} for Subject {sid}')
            plt.xlabel('Epoch ID')
            plt.ylabel(f'{feature_name} Value')
            plt.legend()
            plt.grid()
            plt.show()

        if self.verbose:
            print(f"Feature {feature_name} plotted for all subjects in the session.")

        if return_fig:
            return plt.gcf()
        else:
            return None
        

    def plot_sd1(self):
        """
        Plot SD1 for each subject in the session.
        Raises:
            ValueError: If session has not been set or if SD1 is not set for subjects.
        """
        return self.plot_feature("SD1", y_min=0, y_max=1000)
    
    def plot_sd2(self):
        """
        Plot SD2 for each subject in the session.
        Raises:
            ValueError: If session has not been set or if SD2 is not set for subjects.
        """
        return self.plot_feature("SD2", y_min=0, y_max=1000)
    
    def plot_sd1_sd2_ratio(self):
        """
        Plot SD1/SD2 Ratio for each subject in the session.
        Raises:
            ValueError: If session has not been set or if SD1/SD2 Ratio is not set for subjects.
        """
        return self.plot_feature("SD1_SD2_Ratio", y_min=0, y_max=5)
    
    def plot_centroid(self):
        """
        Plot Centroid for each subject in the session.
        Raises:
            ValueError: If session has not been set or if Centroid is not set for subjects.
        """
        return self.plot_feature("Centroid", y_min=0, y_max=2000)
    

    def plot_icd(self, return_fig: bool = False):
        """
        Plot Inter-Centroid Distances (ICD) for all dyads in the session.
        Args:
            return_fig (bool): If True, return the matplotlib figure object.
        Raises:
            ValueError: If session has not been set or if ICD is not set for dyads.
        """
        if self.session is None:
            raise ValueError("Session has not been set")
        
        if self.verbose:
            print("Plotting Inter-Centroid Distances (ICD) for all dyads...")


        for dyad_id, dyad_data in self.features["dyads"].items():
            icd_session = dyad_data["ICD"]["session"]
            if isinstance(icd_session, dict):
                icd_session = [icd_session[k] for k in sorted(icd_session.keys())]
            if isinstance(icd_session, np.ndarray):
                icd_session = icd_session.tolist()

            plt.figure(figsize=(10, 5))
            plt.plot(icd_session, label=f'Dyad {dyad_id}')

            plt.title(f'Inter-Centroid Distances (ICD) for dyad {dyad_id}')
            plt.xlabel('Epoch ID')
            plt.ylabel('ICD Value')
            plt.ylim(0, 1200)
            plt.xlim(0, len(icd_session) - 1)
            plt.axhline(y=np.mean(icd_session), color='r', linestyle='--', label='Mean ICD')
            # add a trend line
            z = np.polyfit(range(len(icd_session)), icd_session, 1)
            p = np.poly1d(z)
            plt.plot(range(len(icd_session)), p(range(len(icd_session))), color='orange', linestyle='--', label='Trend Line ICD')
            plt.legend()
            plt.grid()

        plt.show()

        if self.verbose:
            print("ICD plotted for all dyads in the session.")

        if return_fig:
            return plt.gcf()
        else:
            return None

    '''  
    def plot_participant_connection_bars(
        self,
        participant_ratios: pd.DataFrame,
        pair_ratios: pd.DataFrame,
        participants: list = ["0", "1", "2"],  # Par défaut : chaînes
        figsize: tuple = (5, 3)
    ):
        """
        Plot stacked bar chart for participant connection ratios.

        Parameters
        ----------
        participant_ratios : pd.DataFrame
        pair_ratios : pd.DataFrame
        participants : list of str or int
        figsize : tuple
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # Toujours travailler en str pour les IDs participants
        participants_str = [str(pid) for pid in participants]
        participant_ratios.index = participant_ratios.index.map(str)
        pair_ratios.index = pair_ratios.index.map(str)

        def p_ratio(pid, col):
            pid_str = str(pid)
            for key in (pid_str, f"P{pid_str}"):
                if key in participant_ratios.index:
                    return float(participant_ratios.loc[key, col])
            return 0.0

        def pair_ratio(i, j):
            i_str, j_str = str(i), str(j)
            key = f"{min(i_str, j_str)}-{max(i_str, j_str)}"
            return float(pair_ratios.loc[key, "2_cluster_ratio"]) if key in pair_ratios.index else 0.0

        pastel = {
            "blue":   "#A9D0F5",
            "green":  "#B7E1B1",
            "red":    "#F4A1A1",
            "orange": "#FFD7B5",
            "gold":   "#FFE69B",
        }
        color_map = {
            ("0", "hard"): pastel["gold"],   ("0", "soft"): pastel["orange"],
            ("0", "P1"):   pastel["green"],  ("0", "P2"):   pastel["red"],
            ("1", "hard"): pastel["gold"],   ("1", "soft"): pastel["orange"],
            ("1", "P0"):   pastel["blue"],   ("1", "P2"):   pastel["red"],
            ("2", "hard"): pastel["gold"],   ("2", "soft"): pastel["orange"],
            ("2", "P0"):   pastel["blue"],   ("2", "P1"):   pastel["green"],
        }
        stack_order = ["hard", "soft", "P0", "P1", "P2"]
        x = np.arange(len(participants_str))
        bottoms = np.zeros(len(participants_str))

        plot_data = {}
        for pid in participants_str:
            plot_data[pid] = {
                "hard": p_ratio(pid, "strong_3_ratio"),
                "soft": p_ratio(pid, "soft_3_ratio"),
                "P0":   pair_ratio(pid, "0") if pid != "0" else 0.0,
                "P1":   pair_ratio(pid, "1") if pid != "1" else 0.0,
                "P2":   pair_ratio(pid, "2") if pid != "2" else 0.0,
            }

        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        for metric in stack_order:
            heights = [plot_data[pid].get(metric, 0.0) for pid in participants_str]
            colors  = [color_map.get((pid, metric), "#CCCCCC") for pid in participants_str]
            bars = ax.bar(x, heights, 0.6, bottom=bottoms,
                        color=colors, edgecolor="white")

            # add percentages
            for rect, h in zip(bars, heights):
                if h >= 0.02:  # label slices ≥ 2 %
                    ax.text(rect.get_x() + rect.get_width()/2,
                            rect.get_y() + h/2,
                            f"{h*100:.0f}%",
                            ha="center", va="center", fontsize=7)

            bottoms += np.array(heights)

        ax.set_xticks(x)
        ax.set_xticklabels([f"P{p}" for p in participants_str])
        ax.set_ylabel("Ratio of epochs")
        ax.set_ylim(0, 1)
        ax.set_title("Connection ratios per participant")

        legend_items = [
            plt.Line2D([0], [0], lw=6, color=pastel["gold"],   label="Hard 3-cluster"),
            plt.Line2D([0], [0], lw=6, color=pastel["orange"], label="Soft 3-cluster"),
            plt.Line2D([0], [0], lw=6, color=pastel["blue"],   label="2-cluster with P0"),
            plt.Line2D([0], [0], lw=6, color=pastel["green"],  label="2-cluster with P1"),
            plt.Line2D([0], [0], lw=6, color=pastel["red"],    label="2-cluster with P2"),
        ]
        ax.legend(handles=legend_items, frameon=False, fontsize=7, ncol=2)
        fig.tight_layout()
        from IPython.display import display
        display(fig)
        plt.close(fig)

    '''

    def plot_connection_heatmap(self, return_fig: bool = False) -> plt.figure:
        """
        Plot a heatmap of connections between participants based on dyad clusters.
        Args:
            return_fig (bool): If True, return the matplotlib figure object.
        Returns:
            matplotlib.figure.Figure: The matplotlib figure object if return_fig is True.
        Raises:
            ValueError: If session has not been set or if clusters are not computed.
        """

        sids = list(self.features["subjects"].keys())

        n = len(sids)
        cols = [f"S{s}" for s in sids] + ["soft", "hard", "Total"]
        mat = np.full((n+1, len(cols)), np.nan)

        # group-level soft / hard ratios (use first participant as ref)
        soft_r = self.clusters["3-soft-clusters"]
        hard_r = self.clusters["3-strong-clusters"]

        pair_sum_total = 0

        for dyad in self.features["dyads"]:
            s1, s2 = dyad.split("_")
            s1, s2 = int(s1), int(s2)
            value = self.clusters["2-clusters"][dyad]
            mat[s1, s2] = value
            mat[s2, s1] = value
            pair_sum_total += value

        # bottom row: sums
        mat[n, :len(cols)] = 0.0
        mat[n, :n] = np.nansum(mat[:n, :n], axis=0)
        mat[0:n,n] = soft_r
        mat[0:n,n+1] = hard_r
        mat[n, n] = soft_r
        mat[n, n+1] = hard_r
        for i in range(n):
            mat[i, n+2] = np.nansum(mat[i, :n]) + soft_r + hard_r
        mat[n, n+2] = pair_sum_total + soft_r + hard_r

        cmap = plt.cm.get_cmap("coolwarm", 256)

        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=300)
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1)

        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])

        for i in range(n+1):
            for j in range(len(cols)):
                if np.isnan(mat[i, j]): continue
                ax.text(j, i, f"{mat[i, j]*100:.0f}%",
                        ha="center", va="center", fontsize=7)

        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right")
        ax.set_yticks(range(n+1))
        ax.set_yticklabels([f"P{p}" for p in sids] + ["Total"])

        ax.set_title("Connection heatmap (% of epochs)")
        cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
        cbar.ax.set_ylabel("% of epochs", rotation=270, labelpad=15)
        plt.show()

        if return_fig:
            return fig
        else:
            return None

    def plot_weighted_connectivity_graph(self, return_fig: bool = False) -> plt.figure:
        """
        Plot a weighted connectivity graph based on dyad clusters.
        Args:
            return_fig (bool): If True, return the matplotlib figure object.
        Returns:
            matplotlib.figure.Figure: The matplotlib figure object if return_fig is True.
        Raises:
            ValueError: If session has not been set or if clusters are not computed.
        """

        values = {}
        edges = {}

        for dyad in self.features["dyads"]:
            values[dyad] = self.clusters["2-clusters"][dyad]
            values[dyad] += self.clusters["3-soft-clusters"]
            values[dyad] += self.clusters["3-strong-clusters"]
            edges[(dyad[0], dyad[2])] = values[dyad]

        pos = {"0": (0, 0), "1": (1, 0), "2": (0.5, 0.866)}   # equilateral triangle

        plt.figure(figsize=(10, 5))
        cmap = plt.cm.get_cmap("coolwarm", 256)

        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        for (i, j), w in edges.items():
            (x1, y1), (x2, y2) = pos[i], pos[j]
            color = cmap(w)
            ax.plot([x1, x2], [y1, y2], lw=8, color=color, solid_capstyle='round')
            mid = ((x1 + x2)/2, (y1 + y2)/2)
            ax.text(*mid, f"{w*100:.0f}%", ha="center", va="center", fontsize=8, color="black")

        for p, (x, y) in pos.items():
            ax.scatter(x, y, s=300, color="white", edgecolor="black", zorder=3)
            ax.text(x, y, f"S{p}", ha="center", va="center", fontsize=10, weight="bold")

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1)
        ax.axis("off")
        ax.set_title("Connectivity graph")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
        cbar.ax.set_ylabel("% of epochs", rotation=270, labelpad=15)
        plt.show()

        if return_fig:
            return fig
        else:
            return None

                            
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