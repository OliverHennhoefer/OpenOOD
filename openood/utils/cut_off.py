import itertools

import numpy as np
import polars as pl
import polars.selectors as cs
from scipy.special import softmax

from tqdm import tqdm
from scipy import stats
from kneed import KneeLocator
from sklearn.neighbors import KernelDensity
from typing import Literal, List, Dict, Tuple, Optional, Any

class ScreeCutoffSelector:
    """
    Analyzes a scree plot derived from a Polars DataFrame's column
    and proposes an integer cutoff point (number of items to keep).

    Includes Kneedle elbow detection, percentage of total value,
    and two-lines piecewise regression methods.
    """

    def __init__(self, sensitivity: float = 1.0):
        """
        Initializes the selector.

        Args:
            sensitivity (float): Sensitivity parameter 'S' for the Kneedle algorithm.
                                 Lower values are less sensitive -> later cutoff.
                                 Defaults to 1.0.
        """
        if sensitivity <= 0:
            raise ValueError("Sensitivity 'S' must be positive.")
        self.sensitivity = sensitivity

    def _find_cutoff_index_kneedle(self, values: np.ndarray) -> int | None:
        # (Implementation remains the same - finds max curvature elbow)
        n_points = len(values)
        if n_points < 3:
            return None
        x = np.arange(n_points)
        y = values
        try:
            kneedle = KneeLocator(
                x,
                y,
                S=self.sensitivity,
                curve="convex",
                direction="decreasing",
                online=False,
            )
            if kneedle.elbow is None:
                print(f"Kneedle (S={self.sensitivity}) did not detect an elbow index.")
            return kneedle.elbow
        except Exception as e:
            print(f"Error during Kneedle elbow detection (S={self.sensitivity}): {e}")
            return None

    @staticmethod
    def _find_cutoff_index_percentage(
        values: np.ndarray, threshold: float = 0.90
    ) -> int | None:
        # (Implementation remains the same - finds cumulative threshold)
        if not (0 < threshold <= 1):
            raise ValueError("Percentage threshold must be > 0 and <= 1.")
        if len(values) == 0:
            return None
        if np.any(values < 0):
            print("Warning: Negative values found in percentage calculation.")

        total_sum = np.sum(values)
        if total_sum <= 0:
            print("Warning: Total sum is non-positive for percentage calculation.")
            return None

        cumulative_sum = np.cumsum(values)
        target_sum = total_sum * threshold
        indices_meeting_threshold = np.where(cumulative_sum >= target_sum)[0]

        if len(indices_meeting_threshold) > 0:
            return indices_meeting_threshold[0]  # noqa
        else:
            print(f"Warning: Cumulative sum never reached {threshold*100}%.")
            return len(values) - 1

    @staticmethod
    def _find_cutoff_index_two_lines(values: np.ndarray) -> int | None:
        """
        Internal helper: Finds the best breakpoint for fitting two lines.

        Args:
            values (np.ndarray): 1D numpy array of y-values, sorted descending.

        Returns:
            int | None: The 0-based index `k` that is the optimal breakpoint,
                        or None if calculation fails.
        """
        n_points = len(values)
        if n_points < 4:  # Need at least 2 points per line
            print("Warning: Too few points (<4) for two-lines method.")
            return None

        x = np.arange(n_points)
        y = values
        total_errors = np.full(n_points, np.inf)

        # Iterate through potential breakpoints k
        # Breakpoint k means line 1: 0..k, line 2: k+1..n-1
        # Need at least 2 points for each line fit
        for k in range(1, n_points - 2):
            # Segment 1: indices 0 to k (inclusive, k+1 points)
            x1 = x[0 : k + 1]
            y1 = y[0 : k + 1]
            # Segment 2: indices k+1 to n-1 (inclusive, n-(k+1) points)
            x2 = x[k + 1 : n_points]
            y2 = y[k + 1 : n_points]

            # Fit lines using linear regression (polyfit deg=1 is equivalent)
            try:
                res1 = stats.linregress(x1, y1)
                res2 = stats.linregress(x2, y2)

                # Calculate squared errors for each segment
                err1 = np.sum((y1 - (res1.slope * x1 + res1.intercept)) ** 2)
                err2 = np.sum((y2 - (res2.slope * x2 + res2.intercept)) ** 2)

                # Store total error for this breakpoint
                total_errors[k] = err1 + err2
            except ValueError as e:
                # Handle cases where linregress fails (e.g., vertical line - unlikely here)
                print(f"Linregress failed at k={k}: {e}")
                total_errors[k] = np.inf

        # Find the breakpoint k that minimizes the total error
        best_k = np.argmin(total_errors)

        if np.isinf(total_errors[best_k]):
            print(
                "Warning: Two-lines method failed to find a valid minimum error breakpoint."
            )
            return None

        # The best breakpoint is index `best_k`.
        return best_k

    def propose_cutoff(
        self,
        df: pl.DataFrame,
        value_col: str = "averaged_divergence",
        method: str = "kneedle",  # Default method
        percentage_threshold: float = 0.90,
    ) -> int:
        """
        Analyzes the scree plot and proposes a cutoff point (number of items to keep).

        Args:
            df (pl.DataFrame): Input Polars DataFrame.
            value_col (str): Column name for scree plot values.
            method (str): Cutoff method: 'kneedle', 'percentage', or 'two_lines'.
            percentage_threshold (float): Threshold for 'percentage' method (0 < threshold <= 1).

        Returns:
            int: Proposed number of items to keep. Returns total rows if no
                 cutoff determined or on error.

        Raises:
            ValueError, TypeError for invalid inputs.
        """
        # --- Input Validation ---
        if not isinstance(df, pl.DataFrame):
            raise TypeError("Input 'df' must be a Polars DataFrame.")
        if value_col not in df.columns:
            raise ValueError(f"Column '{value_col}' not found.")
        if method not in ["kneedle", "percentage", "two_lines"]:
            raise ValueError("Method must be 'kneedle', 'percentage', or 'two_lines'.")
        if df.height == 0:
            return 0

        # --- Data Preparation ---
        df_sorted = df.sort(value_col, descending=True)
        values_np = df_sorted[value_col].to_numpy()

        # --- Cutoff Detection ---
        cutoff_index = None  # 0-based index
        print(f"--- Applying method: {method} ---")  # Added print
        if method == "kneedle":
            cutoff_index = self._find_cutoff_index_kneedle(values_np)
        elif method == "percentage":
            cutoff_index = self._find_cutoff_index_percentage(
                values_np, threshold=percentage_threshold
            )
        elif method == "two_lines":
            cutoff_index = self._find_cutoff_index_two_lines(values_np)

        # --- Determine Result ---
        if cutoff_index is not None:
            num_items_to_keep = cutoff_index + 1
            # Provide context based on method
            context = (
                f"S={self.sensitivity}"
                if method == "kneedle"
                else f"%Thr={percentage_threshold}" if method == "percentage" else ""
            )
            print(
                f"Method '{method}' ({context}) found cutoff index {cutoff_index}."
                f"Proposing to keep {num_items_to_keep} items."
            )
            return num_items_to_keep
        else:
            # Fallback
            num_items_total = df.height
            print(
                f"Method '{method}' did not determine a specific cutoff point."
                f"Proposing to keep all {num_items_total} items."
            )
            return num_items_total
