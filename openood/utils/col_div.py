import itertools

import numpy as np
import polars as pl
import polars.selectors as cs

from tqdm import tqdm
from typing import Literal, List, Dict, Tuple, Optional


class ColumnarDistributionDivergence:
    """
    Calculates the average pairwise divergence (Hellinger distance or
    Jensen-Shannon divergence) between class distributions within specified
    columns of a Polars DataFrame.

    Distributions are estimated using histograms.
    """

    def __init__(self, data: pl.DataFrame):
        """
        Initializes the calculator with the input DataFrame.

        Args:
            data: A Polars DataFrame containing feature columns and a label column.
        """
        if not isinstance(data, pl.DataFrame):
            raise TypeError("Input data must be a Polars DataFrame.")
        self.data: pl.DataFrame = data
        self._label_col: Optional[str] = None
        self._labels: Optional[List[any]] = None

    @staticmethod
    def _calculate_bins_fd(data: np.ndarray, data_range: float) -> Optional[int]:
        """Calculates number of bins using Freedman-Diaconis rule."""
        n = len(data)
        if n < 2:
            return None  # Cannot compute IQR
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        if iqr > 0:
            bin_width = 2 * iqr / (n ** (1 / 3))
            # Ensure bin_width is not zero before division
            if bin_width > 0 and data_range > 0:
                num_bins = int(np.ceil(data_range / bin_width))
                return max(1, num_bins)  # Ensure at least 1 bin
            # Fall through if bin_width is 0 or data_range is 0
        # Fallback (if IQR is 0 or calculation failed) - use Sturges'
        if n > 0:
            return max(1, int(np.ceil(np.log2(n) + 1)))
        return None

    @staticmethod
    def _calculate_bins_sturges(data: np.ndarray) -> Optional[int]:
        """Calculates number of bins using Sturges' formula."""
        n = len(data)
        if n == 0:
            return None
        return max(1, int(np.ceil(np.log2(n) + 1)))

    def _get_feature_columns(self, like: str) -> List[str]:
        """Identifies feature columns based on a 'like' pattern."""
        if self._label_col is None:
            raise RuntimeError(
                "Label column must be set before getting feature columns."
            )

        # Use Polars selectors for more robust matching
        try:
            # Select columns matching 'like' pattern but exclude the label column
            selected_cols = self.data.select(
                cs.contains(like).exclude(self._label_col)
            ).columns
            feature_columns = selected_cols
        except pl.exceptions.ColumnNotFoundError:
            # Handle case where pattern doesn't match anything gracefully
            feature_columns = []
        except Exception as e:
            # Catch other potential errors during selection
            print(
                f"Warning: Could not select columns using pattern '{like}'. Error: {e}"
            )
            feature_columns = []  # Fallback to empty list

        # Ensure only numeric columns are selected if possible (optional check)
        # final_feature_columns = []
        # for col in feature_columns:
        #      if self.data[col].dtype.is_numeric():
        #           final_feature_columns.append(col)
        #      else:
        #           print(f"Warning: Column '{col}' matched but is not numeric. Skipping.")
        # return final_feature_columns
        return feature_columns  # Keep it simple unless non-numeric is confirmed issue

    @staticmethod
    def _calculate_normalized_histogram(
        series: pl.Series, bins: int, value_range: Tuple[float, float]
    ) -> Optional[np.ndarray]:
        """Calculates a normalized histogram for a Polars Series."""
        valid_series = series.drop_nulls()
        if len(valid_series) == 0:
            return None

        # np.histogram handles range where min==max if density=False
        hist, _ = np.histogram(
            valid_series.to_numpy(),
            bins=bins,
            range=value_range,
            density=False,  # Get counts first
        )
        hist_sum = hist.sum()
        if hist_sum > 0:
            # Normalize to probability distribution
            prob_dist = hist / hist_sum
        else:
            # Handle case where all values fall outside range (unlikely if range is from data)
            # Or if input series was empty after drop_nulls
            prob_dist = np.zeros(bins, dtype=float)  # Return zeros if no data in range
        return prob_dist

    @staticmethod
    def _hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
        """Calculates Hellinger distance between two probability distributions."""
        # Ensure non-negativity before sqrt
        p_sqrt = np.sqrt(np.maximum(p, 0))
        q_sqrt = np.sqrt(np.maximum(q, 0))
        return np.sqrt(np.sum((p_sqrt - q_sqrt) ** 2)) / np.sqrt(2.0)

    def _get_column_range(self, col: str) -> Optional[Tuple[float, float]]:
        """Get the min/max range for a column, ignoring potential nulls."""
        col_stats = self.data.select(
            pl.col(col).min().alias("min"), pl.col(col).max().alias("max")
        ).row(
            0
        )  # Use row(0) to get tuple directly
        min_val, max_val = col_stats
        if min_val is None or max_val is None:
            return None  # Column is all null or data is empty
        # Add small buffer to range if min == max to avoid issues with some histogram impls
        # although numpy should handle it. Let's skip buffer for now.
        return float(min_val), float(max_val)

    def transform(
        self,
        like: str | list[str],
        label_col: str,
        method: Literal["hellinger"],
        # Add parameter for bin selection method
        bin_selection_method: Literal["fixed", "fd", "sturges"] = "fd",  # Default to FD
        num_bins: int = 50,  # Used only if bin_selection_method is 'fixed'
    ) -> pl.DataFrame:
        self._label_col = label_col
        if self._label_col not in self.data.columns:
            raise ValueError(
                f"Label column '{self._label_col}' not found in DataFrame."
            )

        feature_cols = self._get_feature_columns(like)
        if not feature_cols:
            print(
                f"Warning: No numeric feature columns found matching pattern '{like}'."
            )
            return pl.DataFrame(
                {
                    "column": pl.Series(dtype=pl.Utf8),
                    "average_divergence": pl.Series(dtype=pl.Float64),
                }
            )

        self._labels = self.data[self._label_col].unique(maintain_order=True).to_list()
        if len(self._labels) < 2:
            print(
                "Warning: Fewer than 2 classes found. Cannot calculate pairwise distances."
            )
            return pl.DataFrame(
                {
                    "column": pl.Series(feature_cols, dtype=pl.Utf8),
                    "average_divergence": pl.Series(
                        [None] * len(feature_cols), dtype=pl.Float64
                    ),
                }
            )

        results: List[Dict[str, Optional[float]]] = []

        for col in tqdm(feature_cols):
            col_range = self._get_column_range(col)
            if col_range is None:
                print(
                    f"Warning: Skipping column '{col}' as it contains only nulls or is empty."
                )
                results.append({"column": col, "average_divergence": None})
                continue

            # --- Bin Calculation Logic ---
            if bin_selection_method == "fixed":
                current_num_bins = num_bins
            else:
                # Get all non-null data for the current column to apply rule
                col_data_np = (
                    self.data.select(pl.col(col).drop_nulls()).to_series().to_numpy()
                )
                data_range_val = col_range[1] - col_range[0]

                if len(col_data_np) < 2:
                    print(
                        f"Warning: Using fixed num_bins for column '{col}'"
                        f"due to insufficient data points ({len(col_data_np)})."
                    )
                    current_num_bins = num_bins
                elif data_range_val == 0:
                    current_num_bins = (
                        1  # Only one bin needed if all values are identical
                    )
                else:
                    if bin_selection_method == "fd":
                        current_num_bins = self._calculate_bins_fd(
                            col_data_np, data_range_val
                        )
                    elif bin_selection_method == "sturges":
                        current_num_bins = self._calculate_bins_sturges(col_data_np)
                    # Add elif blocks for 'sqrt', 'rice', 'scott' if implemented
                    else:
                        # Should not happen with Literal, but safeguard
                        raise ValueError(
                            f"Unknown bin_selection_method: {bin_selection_method}"
                        )

                # Fallback if rule calculation failed (returned None)
                if current_num_bins is None:
                    print(
                        f"Warning: Bin calculation failed for column '{col}'."
                        f"Falling back to fixed num_bins={num_bins}."
                    )
                    current_num_bins = num_bins

            # --- End Bin Calculation Logic ---

            # Check if bin calculation resulted in a valid number
            if current_num_bins is None or current_num_bins <= 0:
                print(
                    f"Error: Invalid number of bins ({current_num_bins}) calculated for column '{col}'. Skipping."
                )
                results.append({"column": col, "average_divergence": None})
                continue

            histograms: Dict[any, np.ndarray] = {}
            valid_labels_for_col: List[any] = []
            for label in self._labels:
                series = (
                    self.data.filter(pl.col(self._label_col) == label)
                    .select(pl.col(col))
                    .to_series()
                )
                # Use the calculated number of bins
                hist = self._calculate_normalized_histogram(
                    series, current_num_bins, col_range
                )
                if hist is not None:
                    histograms[label] = hist
                    valid_labels_for_col.append(label)

            # ... (rest of the divergence calculation logic remains the same) ...
            if len(valid_labels_for_col) < 2:
                results.append({"column": col, "average_divergence": None})
                continue

            pairwise_divergences = []
            for label1, label2 in itertools.combinations(valid_labels_for_col, 2):
                # ... (calculate divergence p vs q) ...
                # (Code for calculating divergence using p and q is unchanged)
                p = histograms[label1]
                q = histograms[label2]
                epsilon = 1e-10
                p_smooth = p + epsilon
                q_smooth = q + epsilon
                p_smooth /= p_smooth.sum()
                q_smooth /= q_smooth.sum()
                if method == "hellinger":
                    divergence = self._hellinger_distance(p_smooth, q_smooth)
                else:  # Already checked earlier, but for safety
                    raise ValueError(f"Invalid method '{method}'.")
                pairwise_divergences.append(divergence)

            avg_divergence = (
                float(np.mean(pairwise_divergences)) if pairwise_divergences else None
            )
            results.append({"column": col, "average_divergence": avg_divergence})
            # ... (end of loop) ...

        schema = {"column": pl.Utf8, "average_divergence": pl.Float64}
        return pl.DataFrame(results, schema=schema)
