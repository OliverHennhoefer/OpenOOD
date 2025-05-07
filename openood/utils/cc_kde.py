import numpy as np
import polars as pl
import polars.selectors as cs

from tqdm import tqdm
from scipy.special import softmax
from sklearn.neighbors import KernelDensity
from typing import Literal, List, Dict, Optional, Any
import warnings


class DimensionWiseKdeOOD:
    """
    Performs Out-of-Distribution (OOD) detection based on comparing instance
    likelihoods across dimension-wise Kernel Density Estimates (KDEs) fitted
    for known normal classes.
    """

    def __init__(self):
        """Initializes the DimensionWiseKdeOOD object."""
        self.kdes_: Optional[Dict[str, Dict[Any, Optional[KernelDensity]]]] = None
        self.fitted_columns_: Optional[List[str]] = None
        self.normal_labels_: Optional[List[Any]] = None
        self.bandwidth_method_: Optional[str] = None
        self._log_zero_penalty: float = -1e30  # Large negative value for log(0)
        self.likelihood_mean_vector_: Optional[np.ndarray] = None
        self.likelihood_inv_covariance_: Optional[np.ndarray] = None

    @staticmethod
    def _calculate_bandwidth(data: np.ndarray, method: str) -> float:
        """
        Calculates bandwidth for KDE using Silverman's or Scott's rule.
        """
        n = len(data)

        if n == 0:  # Should ideally not be called with n=0 if min_samples_for_kde > 0
            return 0.01

        std_dev = np.std(data)

        if n == 1:  # KDE on 1 point is ill-defined; std_dev is 0.0.
            return 0.01  # Small, non-zero bandwidth

        if std_dev == 0:  # Multiple identical points
            # KDE is problematic. A very small bandwidth makes it a sharp spike.
            return 1e-6

            # For 1D data, exponent in n_factor is 1/(d+4) = 1/5 = 0.2
        n_factor = n ** (-0.2)
        if method == "silverman":
            factor = 0.9 * std_dev * n_factor
        elif method == "scott":
            factor = 1.06 * std_dev * n_factor
        else:
            # This case should ideally be caught by Literal type hint or earlier validation
            warnings.warn(f"Unknown bandwidth method '{method}', defaulting to 'silverman'.", UserWarning)
            factor = 0.9 * std_dev * n_factor

        return max(factor, 1e-6)  # Ensure bandwidth is positive and non-zero

    def fit(
            self,
            data: pl.DataFrame,
            like: str | list[str],  # Python 3.10+ for | union type
            label_col: str,
            bandwidth_method: Literal["silverman", "scott"] = "silverman",
            kernel: str = "gaussian",
            min_samples_for_kde: int = 5,
            calculate_mahalanobis_params: bool = False,
    ):
        """
        Fits 1D KDEs for each class in each specified column. Optionally
        calculates parameters for Mahalanobis distance in likelihood space.
        """
        if not isinstance(data, pl.DataFrame):
            raise TypeError("Input data must be a Polars DataFrame.")
        if label_col not in data.columns:
            raise ValueError(f"Label column '{label_col}' not found in DataFrame.")

        self.bandwidth_method_ = bandwidth_method

        try:
            potential_cols_selector = cs.contains(like)
            if label_col in data.select(potential_cols_selector).columns:  # Avoid excluding if not matched
                potential_cols_selector = potential_cols_selector.exclude(label_col)

            potential_cols_df = data.select(potential_cols_selector)
            self.fitted_columns_ = potential_cols_df.select(cs.numeric()).columns

            non_numeric_excluded = set(potential_cols_df.columns) - set(self.fitted_columns_)
            if non_numeric_excluded:
                warnings.warn(
                    f"Non-numeric columns matching pattern '{like}' were excluded: {list(non_numeric_excluded)}",
                    UserWarning
                )
        except pl.exceptions.ColumnNotFoundError:  # `cs.contains(like)` found no columns
            self.fitted_columns_ = []
        except Exception as e:
            raise ValueError(f"Error selecting feature columns using pattern '{like}'. Original error: {e}")

        if not self.fitted_columns_:
            raise ValueError(
                f"No numeric feature columns found matching pattern '{like}' (excluding label column '{label_col}')."
            )

        self.normal_labels_ = data[label_col].unique(maintain_order=True).to_list()
        if not self.normal_labels_:
            raise ValueError(f"No labels found in label column '{label_col}'.")

        self.kdes_ = {
            col: {label: None for label in self.normal_labels_}
            for col in self.fitted_columns_
        }

        for col in tqdm(self.fitted_columns_, desc="Fitting KDEs (by feature)", unit="feature", leave=False):
            try:
                # Select feature and label columns, aliasing to avoid name conflicts if 'col' is same as 'label_col'
                # (though 'label_col' should be excluded from 'fitted_columns_')
                # Filter out rows where the current feature column is null.
                feature_and_labels_df = data.select(
                    pl.col(col).alias("feature_value"),
                    pl.col(label_col).alias("label_value")  # Use a distinct alias for clarity
                ).filter(pl.col("feature_value").is_not_null())

                if feature_and_labels_df.is_empty():
                    continue
            except pl.exceptions.PolarsError as e:
                warnings.warn(f"Error preparing data for column '{col}': {e}. Skipping this column.", UserWarning)
                continue

            for label in self.normal_labels_:
                class_feature_data_series = feature_and_labels_df.filter(
                    pl.col("label_value") == label
                ).select("feature_value").to_series()

                if len(class_feature_data_series) < min_samples_for_kde:
                    continue

                class_col_data_np = class_feature_data_series.to_numpy()
                data_reshaped = class_col_data_np.reshape(-1, 1)

                try:
                    bw = self._calculate_bandwidth(class_col_data_np, self.bandwidth_method_)
                    if bw < 1e-7:  # Effective zero bandwidth after calculation
                        warnings.warn(
                            f"Calculated bandwidth for col='{col}', label='{label}' is too small ({bw:.2e}). Skipping KDE.",
                            UserWarning)
                        continue

                    kde = KernelDensity(kernel=kernel, bandwidth=bw)
                    kde.fit(data_reshaped)
                    self.kdes_[col][label] = kde
                except Exception as e:  # Catch any error during KDE fitting for a specific slice
                    warnings.warn(f"Error fitting KDE for col='{col}', label='{label}'. Error: {e}. Storing None.",
                                  UserWarning)
                    # self.kdes_[col][label] remains None

        fitted_kde_count = sum(kde is not None for col_kdes in self.kdes_.values() for kde in col_kdes.values())
        if fitted_kde_count == 0:
            warnings.warn(
                "No KDEs were successfully fitted. Check data (e.g., constant features, too few samples per class) and parameters.",
                UserWarning)

        self.likelihood_mean_vector_ = None
        self.likelihood_inv_covariance_ = None
        if calculate_mahalanobis_params:
            if fitted_kde_count == 0:
                warnings.warn("Cannot calculate Mahalanobis params as no KDEs were fitted.", UserWarning)
                return self

            # Use original data to calculate likelihoods for Mahalanobis params
            train_likelihood_df = self.transform(data, calculate_mahalanobis_score=False)

            likelihood_cols_for_maha = [f"total_loglik_class_{lbl}" for lbl in self.normal_labels_]

            # Check if all expected likelihood columns are present
            actual_cols = train_likelihood_df.columns
            missing_maha_cols = [lc for lc in likelihood_cols_for_maha if lc not in actual_cols]
            if missing_maha_cols:
                warnings.warn(
                    f"Missing likelihood columns for Mahalanobis: {missing_maha_cols}. Cannot calculate params.",
                    UserWarning)
                return self

            train_likelihood_vectors = train_likelihood_df.select(likelihood_cols_for_maha).to_numpy()
            train_likelihood_vectors = np.nan_to_num(
                train_likelihood_vectors, nan=self._log_zero_penalty,
                posinf=self._log_zero_penalty, neginf=self._log_zero_penalty
            )

            if train_likelihood_vectors.shape[0] < 2 or train_likelihood_vectors.shape[1] == 0:
                warnings.warn(
                    "Insufficient data (rows or classes after likelihood calculation) "
                    "to calculate Mahalanobis parameters reliably.", UserWarning)
                return self

            try:
                self.likelihood_mean_vector_ = np.mean(train_likelihood_vectors, axis=0)
                # Ensure rowvar=False for (samples, features) layout
                covariance = np.cov(train_likelihood_vectors, rowvar=False)

                # Handle scalar covariance for single feature (single class likelihood vector)
                if covariance.ndim == 0:
                    covariance = np.array([[covariance]])

                    # Add ridge regularization for stability
                ridge_epsilon = 1e-6
                covariance_reg = covariance + np.eye(covariance.shape[0]) * ridge_epsilon
                self.likelihood_inv_covariance_ = np.linalg.pinv(covariance_reg)
            except Exception as e:
                warnings.warn(f"Error calculating Mahalanobis parameters: {e}. Parameters not stored.", UserWarning)
                self.likelihood_mean_vector_ = None
                self.likelihood_inv_covariance_ = None
        return self

    def transform(
            self, inference_data: pl.DataFrame, calculate_mahalanobis_score: bool = True
    ) -> pl.DataFrame:
        if not all(hasattr(self, attr) and getattr(self, attr) is not None for attr in
                   ['kdes_', 'fitted_columns_', 'normal_labels_']):
            raise RuntimeError("The '.fit()' method must be called before '.transform()'.")

        if not isinstance(inference_data, pl.DataFrame):
            raise TypeError("Input inference_data must be a Polars DataFrame.")

        missing_cols = [col for col in self.fitted_columns_ if col not in inference_data.columns]
        if missing_cols:
            raise ValueError(f"Inference data is missing required (fitted) columns: {missing_cols}")

        n_instances = len(inference_data)
        n_classes = len(self.normal_labels_)

        # Define expected output columns for schema consistency, even if empty
        likelihood_col_names = [f"total_loglik_class_{lbl}" for lbl in self.normal_labels_]
        metric_col_names = [
            "max_log_likelihood", "mean_log_likelihood",
            "ood_score_likelihood_difference", "ood_score_gap",
            "ood_score_entropy", "ood_score_mahalanobis"
        ]
        all_output_cols = likelihood_col_names + metric_col_names

        if n_instances == 0:
            schema = {col_name: pl.Float64 for col_name in all_output_cols}
            return pl.DataFrame(schema=schema)

        # Pre-extract relevant feature data to NumPy array
        if not self.fitted_columns_:  # No features were fitted
            inference_features_np = np.empty((n_instances, 0), dtype=np.float64)
        else:
            select_exprs = [pl.col(col).cast(pl.Float64, strict=False) for col in self.fitted_columns_]
            inference_features_np = inference_data.select(select_exprs).to_numpy()

        total_log_likelihoods = np.full((n_instances, n_classes), 0.0, dtype=np.float64)

        if not self.fitted_columns_ or n_classes == 0:  # No features or no classes to score against
            # Apply penalty if there are classes but no features to score them on
            if n_classes > 0:  # if n_classes is 0, total_log_likelihoods is (N,0) which is fine
                total_log_likelihoods[:] = self._log_zero_penalty * (
                    len(self.fitted_columns_) if self.fitted_columns_ else 1)
        else:
            for col_idx, col_name in enumerate(
                    tqdm(self.fitted_columns_, desc="Scoring Instances", unit="feature", leave=False,
                         dynamic_ncols=True)):

                current_feature_values = inference_features_np[:, col_idx]
                current_feature_values_reshaped = current_feature_values.reshape(-1, 1)
                mask_finite_input = np.isfinite(current_feature_values)

                for class_idx, label in enumerate(self.normal_labels_):
                    kde = self.kdes_[col_name].get(label)

                    if kde is not None:
                        log_likes_for_feature_class = np.full(n_instances, self._log_zero_penalty, dtype=float)
                        finite_values_to_score = current_feature_values_reshaped[mask_finite_input]

                        if finite_values_to_score.size > 0:
                            # Assuming controlled environment, score_samples should work if KDE is valid and input is shaped correctly.
                            # Errors here would indicate more fundamental issues than typically handled by try-except-pass.
                            scored_finite_values = kde.score_samples(finite_values_to_score)
                            scored_finite_values_clean = np.nan_to_num(
                                scored_finite_values, nan=self._log_zero_penalty,
                                posinf=self._log_zero_penalty, neginf=self._log_zero_penalty
                            )
                            log_likes_for_feature_class[mask_finite_input] = scored_finite_values_clean

                        total_log_likelihoods[:, class_idx] += log_likes_for_feature_class
                    else:  # KDE not available for this feature-class pair
                        total_log_likelihoods[:, class_idx] += self._log_zero_penalty

        # --- Prepare results DataFrame ---
        results_data = {}
        if self.normal_labels_:  # Check if there are any labels
            for i, lbl_col_name in enumerate(likelihood_col_names):
                results_data[lbl_col_name] = total_log_likelihoods[:, i]

        # For stats, replace penalties with NaN to ignore them in nanmax/nanmean
        logliks_for_stats = np.where(total_log_likelihoods <= (self._log_zero_penalty + 1e-9),
                                     np.nan, total_log_likelihoods)

        if n_classes == 0:  # Ensure logliks_for_stats is (N,0) if no classes
            logliks_for_stats = np.empty((n_instances, 0))

        # --- Metric Calculations (Optimized with NumPy) ---
        with np.errstate(invalid='ignore', divide='ignore'):  # Suppress warnings from all-NaN slices or log(0)
            if n_classes > 0:
                max_log_likelihood = np.nanmax(logliks_for_stats, axis=1)
                mean_log_likelihood = np.nanmean(logliks_for_stats, axis=1)
            else:
                max_log_likelihood = np.full(n_instances, np.nan)
                mean_log_likelihood = np.full(n_instances, np.nan)

            max_log_likelihood = np.nan_to_num(max_log_likelihood, nan=self._log_zero_penalty)
            mean_log_likelihood = np.nan_to_num(mean_log_likelihood, nan=self._log_zero_penalty)
            results_data["max_log_likelihood"] = max_log_likelihood
            results_data["mean_log_likelihood"] = mean_log_likelihood

            ood_score_likelihood_difference = -(max_log_likelihood - mean_log_likelihood)
            results_data["ood_score_likelihood_difference"] = ood_score_likelihood_difference

            ood_score_gap = np.full(n_instances, 0.0)
            if n_classes >= 2:
                sorted_logliks = np.sort(logliks_for_stats, axis=1)
                gap = sorted_logliks[:, -1] - sorted_logliks[:, -2]
                ood_score_gap = np.nan_to_num(-gap, nan=0.0)  # OOD if gap small/negative; 0 if not computable
            results_data["ood_score_gap"] = ood_score_gap

            ood_score_entropy = np.full(n_instances, 0.0)  # Default to 0 entropy if no classes
            if n_classes > 0:
                # Mask rows where all likelihoods are effectively _log_zero_penalty
                mask_meaningful_ll_rows = np.any(logliks_for_stats > (self._log_zero_penalty + 1e-9), axis=1)

                # Initialize probabilities to uniform for rows that are all penalties, or if n_classes=0
                probabilities = np.full_like(total_log_likelihoods, 1.0 / n_classes if n_classes > 0 else 0.0)

                if np.any(mask_meaningful_ll_rows):
                    meaningful_logliks = total_log_likelihoods[mask_meaningful_ll_rows, :]
                    probabilities[mask_meaningful_ll_rows, :] = softmax(meaningful_logliks, axis=1)

                # Add epsilon for log stability
                log_probabilities = np.log(probabilities + 1e-30)
                entropy_values = -np.sum(probabilities * log_probabilities, axis=1)

                default_entropy_val = np.log(n_classes) if n_classes > 1 else 0.0  # Max entropy for uniform
                ood_score_entropy = np.nan_to_num(entropy_values, nan=default_entropy_val)
            results_data["ood_score_entropy"] = ood_score_entropy

            ood_score_mahalanobis = np.full(n_instances, 0.0)  # Default to 0.0 (not OOD)
            if (
                    calculate_mahalanobis_score and
                    self.likelihood_mean_vector_ is not None and
                    self.likelihood_inv_covariance_ is not None and
                    n_classes > 0 and
                    self.likelihood_mean_vector_.shape[0] == n_classes  # Ensure consistency
            ):
                likelihood_vectors_clean = np.nan_to_num(
                    total_log_likelihoods, nan=self._log_zero_penalty,
                    posinf=self._log_zero_penalty, neginf=self._log_zero_penalty
                )
                if likelihood_vectors_clean.shape[1] == self.likelihood_mean_vector_.shape[0]:
                    delta = likelihood_vectors_clean - self.likelihood_mean_vector_
                    term1 = delta @ self.likelihood_inv_covariance_
                    m_dist_sq = np.sum(term1 * delta, axis=1)
                    ood_score_mahalanobis = np.maximum(m_dist_sq, 0)  # Ensure non-negativity
                else:  # Should not happen if checks pass
                    warnings.warn("Dimension mismatch for Mahalanobis calculation despite checks. Skipping.",
                                  UserWarning)

            results_data["ood_score_mahalanobis"] = np.nan_to_num(ood_score_mahalanobis, nan=0.0)

        results_df = pl.DataFrame(results_data)

        # Ensure correct column order, selecting only columns present in results_df
        ordered_cols_present = [col for col in all_output_cols if col in results_df.columns]
        return results_df.select(ordered_cols_present)