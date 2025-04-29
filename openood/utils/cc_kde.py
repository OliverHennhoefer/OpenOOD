import numpy as np
import polars as pl
import polars.selectors as cs

from tqdm import tqdm
from scipy.special import softmax
from sklearn.neighbors import KernelDensity
from typing import Literal, List, Dict, Optional, Any


class DimensionWiseKdeOOD:
    """
    Performs Out-of-Distribution (OOD) detection based on comparing instance
    likelihoods across dimension-wise Kernel Density Estimates (KDEs) fitted
    for known normal classes.

    Attributes:
        kdes_ (dict): Stores the fitted sklearn KernelDensity objects.
                     Structure: {column_name: {class_label: kde_object | None}}
        fitted_columns_ (list): List of column names used for fitting.
        normal_labels_ (list): List of unique class labels seen during fitting.
        bandwidth_method_ (str): Bandwidth selection method used.
    """

    def __init__(self):
        """Initializes the DimensionWiseKdeOOD object."""
        self.kdes_: Optional[Dict[str, Dict[Any, Optional[KernelDensity]]]] = None
        self.fitted_columns_: Optional[List[str]] = None
        self.normal_labels_: Optional[List[Any]] = None
        self.bandwidth_method_: Optional[str] = None
        # Define a large negative value to represent log(0) for missing KDEs
        self._log_zero_penalty: float = -1e30
        # Add attributes for Mahalanobis if needed later (set during fit/transform on ID data)
        self.likelihood_mean_vector_: Optional[np.ndarray] = None
        self.likelihood_inv_covariance_: Optional[np.ndarray] = None

    @staticmethod
    def _calculate_bandwidth(data: np.ndarray, method: str) -> float:
        """
        Calculates bandwidth for KDE using Silverman's or Scott's rule.

        Args:
            data: 1D NumPy array of data points for a single class/column.
            method: 'silverman' or 'scott'.

        Returns:
            The calculated bandwidth (float). Returns a small default value
            if calculation is not possible (e.g., std dev is zero).
        """
        n = len(data)
        std_dev = np.std(data)

        # Handle cases with insufficient data or zero standard deviation
        if n <= 1 or std_dev == 0:
            # print(f"Warning: Using small default bandwidth for data with n={n}, std={std_dev:.2e}")
            return 0.01  # Heuristic small default bandwidth

        # Calculate bandwidth factor based on n^(-1/5) for KDE
        n_factor = n ** (-0.2)  # n^(-1/5)

        if method == "silverman":
            factor = 0.9 * std_dev * n_factor  # noqa
        elif method == "scott":
            factor = 1.06 * std_dev * n_factor  # noqa
        else:
            print(
                f"Warning: Unknown bandwidth method '{method}', defaulting to 'silverman'."
            )
            factor = 0.9 * std_dev * n_factor  # noqa

        return max(factor, 1e-6)

    def fit(
        self,
        data: pl.DataFrame,
        like: str | list[str],
        label_col: str,
        bandwidth_method: Literal["silverman", "scott"] = "silverman",
        kernel: str = "gaussian",
        min_samples_for_kde: int = 5,
        calculate_mahalanobis_params: bool = False,  # New flag
    ):
        """
        Fits 1D KDEs for each class in each specified column. Optionally
        calculates parameters for Mahalanobis distance in likelihood space.

        Args:
            data: Training Polars DataFrame with normal class data and labels.
            like: Polars selector pattern (e.g., "^feature_.*$", "col_*")
                  to select feature columns.
            label_col: Name of the column containing class labels.
            bandwidth_method: Method to estimate KDE bandwidth ('silverman' or 'scott').
            kernel: Kernel function for KDE (passed to sklearn.neighbors.KernelDensity).
            min_samples_for_kde: Minimum number of non-null samples required
                                 per class/column to attempt fitting a KDE.
            calculate_mahalanobis_params: If True, calculate and store the mean
                                         vector and inverse covariance of the
                                         log-likelihood vectors from the training data.
        """
        # --- [Existing fit code from previous steps remains unchanged] ---
        if not isinstance(data, pl.DataFrame):
            raise TypeError("Input data must be a Polars DataFrame.")
        if label_col not in data.columns:
            raise ValueError(f"Label column '{label_col}' not found in DataFrame.")

        self.bandwidth_method_ = bandwidth_method

        try:
            potential_cols = data.select(cs.contains(like).exclude(label_col)).columns
            self.fitted_columns_ = [
                col for col in potential_cols if data[col].dtype.is_numeric()
            ]
            if len(potential_cols) > len(self.fitted_columns_):
                non_numeric = set(potential_cols) - set(self.fitted_columns_)
                print(
                    f"Warning: Non-numeric columns matched pattern and were excluded: {non_numeric}"
                )

        except Exception as e:
            raise ValueError(
                f"Could not select numeric columns using pattern '{like}'. Original error: {e}"
            )

        if not self.fitted_columns_:
            raise ValueError(
                f"No numeric feature columns found matching pattern '{like}'."
            )

        self.normal_labels_ = data[label_col].unique(maintain_order=True).to_list()
        if not self.normal_labels_:
            raise ValueError(f"No labels found in label column '{label_col}'.")

        self.kdes_ = {col: {} for col in self.fitted_columns_}
        print(
            f"Fitting KDEs for {len(self.normal_labels_)} classes and {len(self.fitted_columns_)} columns..."
        )

        for col in tqdm(self.fitted_columns_, desc="Fitting KDEs"):
            for label in self.normal_labels_:
                class_col_data = (
                    data.filter(pl.col(label_col) == label)
                    .select(pl.col(col).drop_nulls())
                    .to_series()
                    .to_numpy()
                )

                if len(class_col_data) < min_samples_for_kde:
                    self.kdes_[col][label] = None
                    continue

                data_reshaped = class_col_data.reshape(-1, 1)

                try:
                    bw = self._calculate_bandwidth(
                        class_col_data, self.bandwidth_method_
                    )
                    kde = KernelDensity(kernel=kernel, bandwidth=bw)
                    kde.fit(data_reshaped)
                    self.kdes_[col][label] = kde
                except Exception as e:
                    print(
                        f"Error fitting KDE for col='{col}', label='{label}'. Error: {e}. Storing None."
                    )
                    self.kdes_[col][label] = None

        fitted_kde_count = sum(
            kde is not None
            for col_kdes in self.kdes_.values()
            for kde in col_kdes.values()
        )
        print(f"Fitting complete. Successfully fitted {fitted_kde_count} KDE models.")
        if fitted_kde_count == 0:
            print(
                "Warning: No KDEs were successfully fitted. Check data and parameters."
            )
        # --- [End of existing fit code] ---

        # --- Optional: Calculate Mahalanobis Parameters ---
        self.likelihood_mean_vector_ = None
        self.likelihood_inv_covariance_ = None
        if calculate_mahalanobis_params:
            print("Calculating Mahalanobis parameters on training likelihoods...")
            if fitted_kde_count == 0:
                print(
                    "Warning: Cannot calculate Mahalanobis params as no KDEs were fitted."
                )
                return self

            # Get likelihoods for the training data itself
            train_likelihood_df = self.transform(
                data, calculate_mahalanobis_score=False
            )  # Avoid recursion
            likelihood_cols = [
                f"total_loglik_class_{lbl}" for lbl in self.normal_labels_
            ]
            train_likelihood_vectors = train_likelihood_df.select(
                likelihood_cols
            ).to_numpy()

            # Handle potential NaNs/Infs in training likelihoods before calculating cov/mean
            train_likelihood_vectors = np.nan_to_num(
                train_likelihood_vectors,
                nan=self._log_zero_penalty,  # Or another strategy?
                posinf=0,
                neginf=self._log_zero_penalty,
            )

            if (
                train_likelihood_vectors.shape[0] < 2  # noqa
                or train_likelihood_vectors.shape[1] < 1  # noqa
            ):
                print(
                    "Warning: Insufficient data or classes to calculate Mahalanobis parameters reliably."
                )
                return self

            try:
                self.likelihood_mean_vector_ = np.mean(train_likelihood_vectors, axis=0)
                # Use pseudo-inverse for robustness if covariance is singular/ill-conditioned
                covariance = np.cov(train_likelihood_vectors, rowvar=False)
                self.likelihood_inv_covariance_ = np.linalg.pinv(covariance)
                print("Mahalanobis parameters calculated.")
            except Exception as e:
                print(
                    f"Error calculating Mahalanobis parameters: {e}. Parameters not stored."
                )
                self.likelihood_mean_vector_ = None
                self.likelihood_inv_covariance_ = None

        return self

    def transform(
        self, inference_data: pl.DataFrame, calculate_mahalanobis_score: bool = True
    ) -> pl.DataFrame:
        """
        Scores instances in the inference data based on likelihood against fitted KDEs.

        Args:
            inference_data: Polars DataFrame with instances to score (labels not needed).
                            Must contain columns matching those used in `fit`.
            calculate_mahalanobis_score: If True and parameters were calculated during fit,
                                         compute the Mahalanobis distance score.

        Returns:
            A Polars DataFrame with OOD scores for each input instance:
            - total_loglik_class_{label}: Summed log-likelihood across columns for each normal class.
            - max_log_likelihood: Maximum log-likelihood across all normal classes.
            - mean_log_likelihood: Mean log-likelihood across all normal classes.
            - median_log_likelihood: Median log-likelihood across all normal classes.
            - ood_score_neg_loglik: Negative of max_log_likelihood (higher = more OOD).
            - ood_score_neg_mean_loglik: Negative of mean_log_likelihood (higher = more OOD).
            - ood_score_neg_median_loglik: Negative of median_log_likelihood (higher = more OOD).
            - ood_score_likelihood_difference: Negative of (max_log_likelihood - mean_log_likelihood) (higher ~ more OOD).
            - ood_score_median_difference: Negative of (max_log_likelihood - median_log_likelihood) (higher ~ more OOD).
            - ood_score_std_max_mean_diff: (max_log_likelihood - mean_log_likelihood) /std_dev(log_likelihoods) (higher ~ more specific fit).
            - max_softmax_prob: Maximum probability from softmax applied to class log-likelihoods (lower = more OOD).
            - ood_score_msp: 1.0 - max_softmax_prob (higher = more OOD).
            - ood_score_mahalanobis: Mahalanobis distance in likelihood space (if calculated, higher = more OOD).
        """
        if (
            self.kdes_ is None
            or self.fitted_columns_ is None
            or self.normal_labels_ is None
        ):
            raise RuntimeError(
                "The '.fit()' method must be called before '.transform()'."
            )
        if not isinstance(inference_data, pl.DataFrame):
            raise TypeError("Input inference_data must be a Polars DataFrame.")

        missing_cols = [
            col for col in self.fitted_columns_ if col not in inference_data.columns
        ]
        if missing_cols:
            raise ValueError(
                f"Inference data is missing required columns: {missing_cols}"
            )

        n_instances = len(inference_data)
        n_classes = len(self.normal_labels_)
        label_to_index = {label: i for i, label in enumerate(self.normal_labels_)}

        total_log_likelihoods = np.full((n_instances, n_classes), 0.0, dtype=np.float64)

        # --- [Existing likelihood calculation loop remains unchanged] ---
        for col in tqdm(self.fitted_columns_, desc="Scoring Instances", leave=False):
            try:
                inference_col_data_np = (
                    inference_data.select(pl.col(col).cast(pl.Float64, strict=False))
                    .to_series()
                    .to_numpy()
                )
                inference_col_data_np_reshaped = inference_col_data_np.reshape(-1, 1)
            except Exception as e:
                print(
                    f"Error processing inference data for column '{col}': {e}. Skipping column."
                )
                total_log_likelihoods += self._log_zero_penalty
                continue

            for label in self.normal_labels_:
                kde = self.kdes_[col].get(label)
                class_index = label_to_index[label]

                if kde is not None:
                    try:
                        mask_finite = np.isfinite(
                            inference_col_data_np_reshaped
                        ).flatten()
                        log_likes = kde.score_samples(inference_col_data_np_reshaped)
                        log_likes_clean = np.nan_to_num(
                            log_likes,
                            nan=self._log_zero_penalty,
                            posinf=0,
                            neginf=self._log_zero_penalty,
                        )
                        log_likes_clean[~mask_finite] = self._log_zero_penalty
                        total_log_likelihoods[:, class_index] += log_likes_clean
                    except ValueError as e:
                        print(
                            f"Warning: Scoring failed for col='{col}', label='{label}'."
                            f" Error: {e}. Applying penalty."
                        )  # Reduced verbosity
                        total_log_likelihoods[:, class_index] += self._log_zero_penalty
                else:
                    total_log_likelihoods[:, class_index] += self._log_zero_penalty
        # --- [End of likelihood calculation loop] ---

        # --- Calculate final metrics ---
        results_data = {}
        # Add per-class log-likelihoods first
        likelihood_cols_order = [
            f"total_loglik_class_{lbl}" for lbl in self.normal_labels_
        ]  # Define order
        for i, lbl in enumerate(self.normal_labels_):
            results_data[likelihood_cols_order[i]] = total_log_likelihoods[:, i]

        # Prepare cleaned likelihoods for stats (replace -inf with NaN)
        logliks_for_stats = np.where(
            np.isneginf(total_log_likelihoods), np.nan, total_log_likelihoods
        )

        with np.errstate(all="ignore"):  # Suppress warnings for all-nan slices etc.

            # --- Max-based ---
            max_log_likelihood = np.nanmax(
                logliks_for_stats, axis=1
            )  # nanmax ignores NaN
            ood_score_neg_loglik = -max_log_likelihood
            results_data["max_log_likelihood"] = max_log_likelihood
            results_data["ood_score_neg_loglik"] = ood_score_neg_loglik

            # --- Mean-based ---
            mean_log_likelihood = np.nanmean(logliks_for_stats, axis=1)
            ood_score_neg_mean_loglik = -mean_log_likelihood
            results_data["mean_log_likelihood"] = mean_log_likelihood
            results_data["ood_score_neg_mean_loglik"] = ood_score_neg_mean_loglik

            # --- Median-based (NEW) ---
            median_log_likelihood = np.nanmedian(logliks_for_stats, axis=1)
            ood_score_neg_median_loglik = -median_log_likelihood
            results_data["median_log_likelihood"] = median_log_likelihood
            results_data["ood_score_neg_median_loglik"] = ood_score_neg_median_loglik

            # --- Difference: Max vs Mean ---
            loglik_diff_max_avg = max_log_likelihood - mean_log_likelihood
            ood_score_likelihood_difference = -loglik_diff_max_avg
            results_data["ood_score_likelihood_difference"] = (
                ood_score_likelihood_difference
            )

            # --- Difference: Max vs Median (NEW) ---
            loglik_diff_max_median = max_log_likelihood - median_log_likelihood
            ood_score_median_difference = -loglik_diff_max_median
            results_data["ood_score_median_difference"] = ood_score_median_difference

            # --- Standardized Difference: Max vs Mean (NEW) ---
            std_dev_log_likelihood = np.nanstd(logliks_for_stats, axis=1, ddof=0)
            # Handle cases with zero std dev (e.g., single class, or all likelihoods identical)
            # Avoid division by zero/NaN; if std=0, the concept is less meaningful, assign 0? or NaN? Let's use NaN.
            ood_score_std_max_mean_diff = np.full(n_instances, np.nan)
            valid_std_mask = (
                std_dev_log_likelihood > 1e-9
            )  # Check for non-trivial std dev
            if np.any(valid_std_mask):
                ood_score_std_max_mean_diff[valid_std_mask] = (
                    loglik_diff_max_avg[valid_std_mask]
                    / std_dev_log_likelihood[valid_std_mask]
                )
            results_data["ood_score_std_max_mean_diff"] = ood_score_std_max_mean_diff

            # --- Softmax-based ---
            probabilities = np.full_like(total_log_likelihoods, np.nan)
            valid_rows_mask = np.any(np.isfinite(total_log_likelihoods), axis=1)
            if np.any(valid_rows_mask):
                probabilities[valid_rows_mask] = softmax(
                    total_log_likelihoods[valid_rows_mask], axis=1
                )
            max_softmax_prob = np.nanmax(probabilities, axis=1)
            max_softmax_prob[~valid_rows_mask] = np.nan  # Ensure consistency
            ood_score_msp = 1.0 - max_softmax_prob
            results_data["max_softmax_prob"] = max_softmax_prob
            results_data["ood_score_msp"] = ood_score_msp

            # --- Mahalanobis Distance (Optional, NEW) ---
            ood_score_mahalanobis = np.full(n_instances, np.nan)  # Initialize with NaN
            if (
                calculate_mahalanobis_score
                and self.likelihood_mean_vector_ is not None
                and self.likelihood_inv_covariance_ is not None
            ):
                # Use the original likelihoods (not nan-replaced ones for stats)
                # but maybe handle NaNs/Infs before calculating distance?
                likelihood_vectors_clean = np.nan_to_num(
                    total_log_likelihoods,
                    nan=self._log_zero_penalty,  # Or impute with mean? Needs consideration.
                    posinf=0,
                    neginf=self._log_zero_penalty,
                )

                delta = likelihood_vectors_clean - self.likelihood_mean_vector_
                # Calculate squared Mahalanobis distance
                # D^2 = delta @ inv_cov @ delta^T (element-wise for each row)
                # Efficient calculation: sum( (delta @ inv_cov) * delta, axis=1)
                term1 = delta @ self.likelihood_inv_covariance_
                ood_score_mahalanobis = np.sum(term1 * delta, axis=1)
            results_data["ood_score_mahalanobis"] = ood_score_mahalanobis

        # Create result DataFrame
        results_df = pl.DataFrame(results_data)

        # Optional: Join results back to original inference_data index if needed
        # return inference_data.hstack(results_df)
        return results_df