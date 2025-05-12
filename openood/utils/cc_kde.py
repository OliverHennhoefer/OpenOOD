import numpy as np
import polars as pl
import polars.selectors as cs

from tqdm import tqdm
from scipy.special import softmax, expit as sigmoid
from scipy.stats import mode
from sklearn.neighbors import KernelDensity
from typing import Literal, List, Dict, Optional, Any
import warnings


class DimensionWiseKdeOOD:
    """
    Performs Out-of-Distribution (OOD) detection based on comparing instance
    likelihoods across dimension-wise Kernel Density Estimates (KDEs) fitted
    for known normal classes.
    """
    # --- Metric Definitions and Dependencies ---
    ALWAYS_CALCULATED_METRIC_NAMES = [
        "max_log_likelihood", "mean_log_likelihood",
        "ood_score_likelihood_difference"
    ]
    ORDERED_OPTIONAL_METRIC_NAMES = [
        "ood_score_gap", "ood_score_entropy", "ood_score_mahalanobis",
        "ood_metric1_avg_support_c_star", "ood_metric2_feature_agreement",
        "ood_metric3_avg_feature_entropy", "ood_metric4_variance_support_c_star",
        "ood_metric5_robust_dissent_c_star", "ood_metric5_adapted_robust_dissent_c_star",
        "ood_metric6_q1_margin_c_star",  # Idea 1
        "ood_metric7_entropy_weighted_dissent_c_star",  # Idea 2
        "ood_metric8_prop_clear_dissent_c_star"  # Idea 3
    ]
    KNOWN_OPTIONAL_METRIC_NAMES_SET = set(ORDERED_OPTIONAL_METRIC_NAMES)

    # Define dependencies for efficient computation
    # These sets determine if certain intermediate arrays need to be computed.
    REQUIRES_C_STAR_INDICES_METRICS = {  # If any of these are active, c_star_indices is needed
        "ood_metric1_avg_support_c_star", "ood_metric4_variance_support_c_star",
        "ood_metric5_robust_dissent_c_star", "ood_metric5_adapted_robust_dissent_c_star",
        "ood_metric6_q1_margin_c_star", "ood_metric7_entropy_weighted_dissent_c_star",
        "ood_metric8_prop_clear_dissent_c_star"
    }
    REQUIRES_P_IFC_METRICS = {  # If any of these are active, P_ifc (softmax of feature_log_likelihoods) is needed
        "ood_metric1_avg_support_c_star", "ood_metric2_feature_agreement",
        "ood_metric3_avg_feature_entropy", "ood_metric4_variance_support_c_star",
        "ood_metric5_robust_dissent_c_star", "ood_metric5_adapted_robust_dissent_c_star",
        "ood_metric7_entropy_weighted_dissent_c_star"  # Needs P_ifc for p_support_c_star and per-feature entropy
    }
    # p_support_c_star_per_feature and its direct aggregates (mean, std, q1_prob, q1_logit)
    REQUIRES_P_SUPPORT_C_STAR_AGGREGATES_METRICS = {
        "ood_metric1_avg_support_c_star", "ood_metric4_variance_support_c_star",
        "ood_metric5_robust_dissent_c_star", "ood_metric5_adapted_robust_dissent_c_star",
        "ood_metric7_entropy_weighted_dissent_c_star"  # Needs base p_support for weighted calculation
    }
    # Per-feature entropy (normalized_entropy_if)
    REQUIRES_PER_FEATURE_ENTROPY_METRICS = {
        "ood_metric3_avg_feature_entropy", "ood_metric7_entropy_weighted_dissent_c_star"
    }

    # --- End Metric Definitions ---

    def __init__(self, dissent_margin_threshold: float = 1.0):
        """
        Initializes the DimensionWiseKdeOOD object.
        Args:
            dissent_margin_threshold (float): The log-likelihood margin required for a feature
                                              to be considered as "clearly dissenting" in
                                              ood_metric8_prop_clear_dissent_c_star.
        """
        self.kdes_: Optional[Dict[str, Dict[Any, Optional[KernelDensity]]]] = None
        self.fitted_columns_: Optional[List[str]] = None
        self.normal_labels_: Optional[List[Any]] = None
        self.bandwidth_method_: Optional[str] = None
        self._log_zero_penalty: float = -1e30  # Large negative value for log(0)
        self.likelihood_mean_vector_: Optional[np.ndarray] = None
        self.likelihood_inv_covariance_: Optional[np.ndarray] = None
        self._epsilon: float = 1e-30  # Small constant for numerical stability
        self.dissent_margin_threshold: float = dissent_margin_threshold

    @staticmethod
    def _calculate_bandwidth(data: np.ndarray, method: str) -> float:
        """
        Calculates bandwidth for KDE using Silverman's or Scott's rule.
        """
        n = len(data)
        if n == 0: return 0.01  # Default small bandwidth
        std_dev = np.std(data)
        if n == 1: return 0.01  # Default small bandwidth for single point
        if std_dev == 0: return 1e-6  # Very small bandwidth if no variance, to avoid errors

        n_factor = n ** (-0.2)  # n^(-1/5)
        if method == "silverman":
            factor = 0.9 * std_dev * n_factor
        elif method == "scott":
            factor = 1.06 * std_dev * n_factor
        else:
            warnings.warn(f"Unknown bandwidth method '{method}', defaulting to 'silverman'.", UserWarning)
            factor = 0.9 * std_dev * n_factor
        return max(factor, 1e-6)  # Ensure bandwidth is not excessively small

    def fit(
            self,
            data: pl.DataFrame,
            like: str | list[str],
            label_col: str,
            bandwidth_method: Literal["silverman", "scott"] = "silverman",
            kernel: str = "gaussian",
            min_samples_for_kde: int = 5,
            calculate_mahalanobis_params: bool = False,
    ):
        """
        Fits dimension-wise KDEs for each class and feature.
        Optionally calculates Mahalanobis distance parameters on total log-likelihood vectors.
        """
        if not isinstance(data, pl.DataFrame):
            raise TypeError("Input data must be a Polars DataFrame.")
        if label_col not in data.columns:
            raise ValueError(f"Label column '{label_col}' not found in DataFrame.")

        self.bandwidth_method_ = bandwidth_method

        # Select feature columns (numeric, excluding label_col)
        try:
            potential_cols_selector = cs.contains(like)
            potential_cols_df = data.select(potential_cols_selector)
            feature_cols_from_like = [col for col in potential_cols_df.columns if col != label_col]
            self.fitted_columns_ = data.select(feature_cols_from_like).select(cs.numeric()).columns
            non_numeric_excluded = set(feature_cols_from_like) - set(self.fitted_columns_)
            if non_numeric_excluded:
                warnings.warn(
                    f"Non-numeric columns matching pattern '{like}' (and not label_col) were excluded: {list(non_numeric_excluded)}",
                    UserWarning)
        except pl.exceptions.ColumnNotFoundError:  # If 'like' pattern matches no columns initially
            self.fitted_columns_ = []
        except Exception as e:  # Catch other potential Polars errors during selection
            raise ValueError(f"Error selecting feature columns using pattern '{like}'. Original error: {e}")

        if not self.fitted_columns_:
            raise ValueError(
                f"No numeric feature columns found matching pattern '{like}' (excluding label column '{label_col}').")

        self.normal_labels_ = data[label_col].unique(maintain_order=True).to_list()
        if not self.normal_labels_:
            raise ValueError(f"No labels found in label column '{label_col}'.")

        # Initialize KDE storage
        self.kdes_ = {
            col: {label: None for label in self.normal_labels_}
            for col in self.fitted_columns_
        }

        # Fit KDEs
        for col in tqdm(self.fitted_columns_, desc="Fitting KDEs (by feature)", unit="feature", leave=False):
            try:
                # Select feature and label values, filter out nulls for the current feature
                feature_and_labels_df = data.select(
                    pl.col(col).alias("feature_value"),
                    pl.col(label_col).alias("label_value")
                ).filter(pl.col("feature_value").is_not_null())

                if feature_and_labels_df.is_empty():
                    continue  # Skip if no non-null data for this feature
            except pl.exceptions.PolarsError as e:  # Catch errors during data prep for a column
                warnings.warn(f"Error preparing data for column '{col}': {e}. Skipping this column.", UserWarning)
                continue

            for label in self.normal_labels_:
                class_feature_data_series = feature_and_labels_df.filter(
                    pl.col("label_value") == label
                ).select("feature_value").to_series()

                if len(class_feature_data_series) < min_samples_for_kde:
                    continue  # Not enough samples for this class-feature pair

                class_col_data_np = class_feature_data_series.to_numpy()
                data_reshaped = class_col_data_np.reshape(-1, 1)  # KDE expects 2D array

                try:
                    bw = self._calculate_bandwidth(class_col_data_np, self.bandwidth_method_)
                    if bw < 1e-7:  # Extremely small bandwidth can cause issues
                        warnings.warn(
                            f"Calculated bandwidth for col='{col}', label='{label}' is too small ({bw:.2e}). Skipping KDE.",
                            UserWarning)
                        continue

                    kde = KernelDensity(kernel=kernel, bandwidth=bw)
                    kde.fit(data_reshaped)
                    self.kdes_[col][label] = kde
                except Exception as e:  # Catch errors during KDE fitting
                    warnings.warn(f"Error fitting KDE for col='{col}', label='{label}'. Error: {e}. Storing None.",
                                  UserWarning)

        fitted_kde_count = sum(kde is not None for col_kdes in self.kdes_.values() for kde in col_kdes.values())
        if fitted_kde_count == 0:
            warnings.warn("No KDEs were successfully fitted. Check data, parameters, and minimum sample counts.",
                          UserWarning)

        # Calculate Mahalanobis parameters if requested
        self.likelihood_mean_vector_ = None
        self.likelihood_inv_covariance_ = None
        if calculate_mahalanobis_params:
            if fitted_kde_count == 0:
                warnings.warn("Cannot calculate Mahalanobis parameters as no KDEs were fitted.", UserWarning)
                return self

            # Call transform with minimal metrics to get total_log_likelihoods for training data
            train_likelihood_df = self.transform(data, metrics_to_compute=[])

            likelihood_cols_for_maha = [f"total_loglik_class_{lbl}" for lbl in self.normal_labels_]
            actual_cols = train_likelihood_df.columns
            missing_maha_cols = [lc for lc in likelihood_cols_for_maha if lc not in actual_cols]

            if missing_maha_cols:
                warnings.warn(
                    f"Missing likelihood columns for Mahalanobis: {missing_maha_cols}. Parameters not calculated.",
                    UserWarning)
                return self
            if not likelihood_cols_for_maha:  # Should be caught by self.normal_labels_ check earlier
                warnings.warn("No normal labels found, cannot calculate Mahalanobis parameters.", UserWarning)
                return self

            train_likelihood_vectors = train_likelihood_df.select(likelihood_cols_for_maha).to_numpy()
            train_likelihood_vectors = np.nan_to_num(  # Clean up any potential NaNs/Infs from likelihoods
                train_likelihood_vectors, nan=self._log_zero_penalty,
                posinf=self._log_zero_penalty, neginf=self._log_zero_penalty
            )

            if train_likelihood_vectors.shape[0] < 2 or train_likelihood_vectors.shape[1] == 0:
                warnings.warn("Insufficient data for Mahalanobis parameters (need at least 2 samples and 1 class).",
                              UserWarning)
                return self
            try:
                self.likelihood_mean_vector_ = np.mean(train_likelihood_vectors, axis=0)
                covariance = np.cov(train_likelihood_vectors, rowvar=False)
                if covariance.ndim == 0:  # Handle 1-class case where cov is scalar
                    covariance = np.array([[covariance]])
                # Add ridge regularization for stability
                self.likelihood_inv_covariance_ = np.linalg.pinv(covariance + np.eye(covariance.shape[0]) * 1e-6)
            except Exception as e:
                warnings.warn(f"Error calculating Mahalanobis parameters: {e}. Parameters not stored.", UserWarning)
                self.likelihood_mean_vector_ = None
                self.likelihood_inv_covariance_ = None
        return self

    def transform(
            self,
            inference_data: pl.DataFrame,
            metrics_to_compute: Optional[List[str]] = None  # User specifies which optional metrics
    ) -> pl.DataFrame:
        """
        Transforms inference data by calculating log-likelihoods and specified OOD metrics.
        """
        # --- Initial Checks and Setup ---
        if not all(hasattr(self, attr) and getattr(self, attr) is not None for attr in
                   ['kdes_', 'fitted_columns_', 'normal_labels_']):
            raise RuntimeError("The '.fit()' method must be called before '.transform()'.")
        if not isinstance(inference_data, pl.DataFrame):
            raise TypeError("Input inference_data must be a Polars DataFrame.")
        if self.fitted_columns_ is None: self.fitted_columns_ = []  # Should be caught by RuntimeError

        missing_cols = [col for col in self.fitted_columns_ if col not in inference_data.columns]
        if missing_cols:
            raise ValueError(f"Inference data is missing required (fitted) columns: {missing_cols}")

        n_instances = len(inference_data)
        n_classes = len(self.normal_labels_)
        n_features = len(self.fitted_columns_)

        # --- Determine Active Metrics and Output Columns ---
        likelihood_col_names = [f"total_loglik_class_{lbl}" for lbl in self.normal_labels_]

        active_metric_col_names = list(self.ALWAYS_CALCULATED_METRIC_NAMES)
        runnable_optional_metrics_set = set()  # Set of optional metrics that will actually be computed

        if metrics_to_compute is None:  # Default: attempt all known optional metrics
            runnable_optional_metrics_set.update(self.KNOWN_OPTIONAL_METRIC_NAMES_SET)
        else:
            for m_name in metrics_to_compute:
                if m_name in self.KNOWN_OPTIONAL_METRIC_NAMES_SET:
                    runnable_optional_metrics_set.add(m_name)
                elif m_name not in self.ALWAYS_CALCULATED_METRIC_NAMES:  # Allow explicit request of always-on
                    warnings.warn(
                        f"Requested metric '{m_name}' is unknown or not an optional metric. It will be ignored.",
                        UserWarning)

        # Add feasible optional metrics to the final list, respecting order
        final_active_optional_metric_names = []
        for m_name in self.ORDERED_OPTIONAL_METRIC_NAMES:
            if m_name in runnable_optional_metrics_set:
                if m_name == "ood_score_mahalanobis":
                    if self.likelihood_mean_vector_ is not None:  # Check if params were fitted
                        final_active_optional_metric_names.append(m_name)
                    elif metrics_to_compute is not None and m_name in metrics_to_compute:  # Explicitly requested but not available
                        warnings.warn(
                            f"Metric '{m_name}' explicitly requested but its parameters were not fitted. It will not be included in the output.",
                            UserWarning)
                else:
                    final_active_optional_metric_names.append(m_name)
        active_metric_col_names.extend(final_active_optional_metric_names)

        all_output_cols = likelihood_col_names + active_metric_col_names

        if n_instances == 0:  # Handle empty input DataFrame
            return pl.DataFrame(schema={col_name: pl.Float64 for col_name in all_output_cols})

        # --- Feature Log-Likelihood Calculation ---
        inference_features_np = np.empty((n_instances, 0), dtype=np.float64)
        if self.fitted_columns_:  # Only select if there are fitted columns
            select_exprs = [pl.col(col).cast(pl.Float64, strict=False) for col in self.fitted_columns_]
            inference_features_np = inference_data.select(select_exprs).to_numpy()

        # Stores log P(x_f | class_c, kde_fc) for each instance i, feature f, class c
        feature_log_likelihoods_per_instance = np.full(
            (n_instances, n_features, n_classes), self._log_zero_penalty, dtype=np.float64
        )
        # Stores sum over f of log P(x_f | class_c, kde_fc) for each instance i, class c
        total_log_likelihoods = np.full((n_instances, n_classes), 0.0, dtype=np.float64)

        if n_features == 0 or n_classes == 0:  # No features or no classes to score against
            if n_classes > 0:  # If there are classes, but no features, assign penalty
                penalty_multiplier = n_features if n_features > 0 else 1  # Should be 1 if n_features is 0
                total_log_likelihoods[:] = self._log_zero_penalty * penalty_multiplier
        else:
            for col_idx, col_name in enumerate(
                    tqdm(self.fitted_columns_, desc="Scoring Instances", unit="feature", leave=False,
                         dynamic_ncols=True)):
                current_feature_values = inference_features_np[:, col_idx]
                current_feature_values_reshaped = current_feature_values.reshape(-1, 1)
                mask_finite_input = np.isfinite(current_feature_values)  # Score only finite values

                for class_idx, label in enumerate(self.normal_labels_):
                    kde = self.kdes_[col_name].get(label)
                    log_likes_this_feature_this_class = np.full(n_instances, self._log_zero_penalty, dtype=float)
                    if kde is not None:
                        finite_values_to_score = current_feature_values_reshaped[mask_finite_input]
                        if finite_values_to_score.size > 0:
                            scored_finite_values = kde.score_samples(finite_values_to_score)
                            # Clean NaNs/Infs from KDE scores (can happen with extreme values or tiny bandwidths)
                            cleaned_scored_values = np.nan_to_num(
                                scored_finite_values, nan=self._log_zero_penalty,
                                posinf=self._log_zero_penalty, neginf=self._log_zero_penalty
                            )
                            log_likes_this_feature_this_class[mask_finite_input] = cleaned_scored_values

                    feature_log_likelihoods_per_instance[:, col_idx, class_idx] = log_likes_this_feature_this_class
                    total_log_likelihoods[:, class_idx] += log_likes_this_feature_this_class

        # --- Prepare results_data dictionary ---
        results_data = {}
        if self.normal_labels_:  # Add total log-likelihoods per class
            for i, lbl_col_name in enumerate(likelihood_col_names):
                results_data[lbl_col_name] = total_log_likelihoods[:, i]

        # For stats, replace penalty values with NaN to ignore them in nanmean/nanmax
        logliks_for_stats = np.where(np.abs(total_log_likelihoods - self._log_zero_penalty) < 1e-9,
                                     np.nan, total_log_likelihoods)
        if n_classes == 0: logliks_for_stats = np.empty((n_instances, 0))

        with np.errstate(invalid='ignore', divide='ignore'):  # Suppress NaN warnings during numpy ops
            # --- Always Calculated Metrics ---
            if n_classes > 0:
                max_log_likelihood_val = np.nanmax(logliks_for_stats, axis=1)
                mean_log_likelihood_val = np.nanmean(logliks_for_stats, axis=1)
            else:  # No classes, so max/mean are undefined (NaN)
                max_log_likelihood_val = np.full(n_instances, np.nan)
                mean_log_likelihood_val = np.full(n_instances, np.nan)

            results_data["max_log_likelihood"] = np.nan_to_num(max_log_likelihood_val, nan=self._log_zero_penalty)
            results_data["mean_log_likelihood"] = np.nan_to_num(mean_log_likelihood_val, nan=self._log_zero_penalty)
            results_data["ood_score_likelihood_difference"] = -(
                    results_data["max_log_likelihood"] - results_data["mean_log_likelihood"])

            # --- Intermediate Computations for Optional Metrics (Conditional) ---
            # Determine if specific intermediate arrays are needed based on active optional metrics
            current_runnable_optional_metrics = set(final_active_optional_metric_names)  # Use the feasible list

            compute_c_star_indices = any(
                m in current_runnable_optional_metrics for m in self.REQUIRES_C_STAR_INDICES_METRICS)
            compute_P_ifc = any(m in current_runnable_optional_metrics for m in self.REQUIRES_P_IFC_METRICS)
            compute_p_support_c_star_aggregates = any(
                m in current_runnable_optional_metrics for m in self.REQUIRES_P_SUPPORT_C_STAR_AGGREGATES_METRICS)
            compute_per_feature_entropy = any(
                m in current_runnable_optional_metrics for m in self.REQUIRES_PER_FEATURE_ENTROPY_METRICS)

            c_star_indices = None  # Index of the class with max total_log_likelihood for each instance
            if compute_c_star_indices and n_classes > 0 and np.any(
                    np.abs(total_log_likelihoods - self._log_zero_penalty) > 1e-9):
                c_star_indices = np.argmax(total_log_likelihoods, axis=1)
            elif compute_c_star_indices:  # Handles n_classes=0 or all penalty LLs
                c_star_indices = np.full(n_instances, -1, dtype=int)  # Placeholder if no valid c_star

            P_ifc = None  # Softmax probabilities: P(class_c | x_f, kde_fc)
            if compute_P_ifc:
                P_ifc = np.empty((n_instances, n_features, n_classes), dtype=np.float64)
                if n_features > 0 and n_classes > 0:
                    P_ifc = softmax(feature_log_likelihoods_per_instance, axis=2)
                elif n_features > 0:  # n_classes is 0
                    P_ifc = np.empty((n_instances, n_features, 0), dtype=np.float64)
                # if n_features == 0, P_ifc is (n_instances, 0, n_classes), which is fine for downstream logic

            # Initialize aggregates for p_support_c_star_per_feature
            mean_p_support_c_star_agg = np.full(n_instances, 0.0)
            std_dev_support_c_star_agg = np.full(n_instances, 0.5)  # Max possible std dev for probs in [0,1]
            q1_support_c_star_agg = np.full(n_instances, 0.0)
            default_q1_logit = np.log(self._epsilon / (1.0 - self._epsilon)) if (
                                                                                            1.0 - self._epsilon) > 0 else -70.0  # Approx logit(epsilon)
            q1_logit_support_c_star_agg_adapted = np.full(n_instances, default_q1_logit)
            p_support_c_star_per_feature = None  # P_ifc[c_star_indices]

            if compute_p_support_c_star_aggregates and P_ifc is not None and c_star_indices is not None and n_features > 0 and n_classes > 0:
                # Ensure c_star_indices are valid before using them for indexing
                valid_c_star_mask = (c_star_indices != -1)
                if np.any(valid_c_star_mask):
                    idx_i, idx_j = np.arange(n_instances)[valid_c_star_mask, None], np.arange(n_features)[None, :]
                    idx_k_c_star = c_star_indices[valid_c_star_mask, None]

                    p_support_c_star_per_feature_valid = P_ifc[idx_i, idx_j, idx_k_c_star]

                    # Initialize full-size arrays for results
                    p_support_c_star_per_feature = np.full((n_instances, n_features), np.nan)  # For Metric 7
                    if p_support_c_star_per_feature_valid.size > 0:  # if any valid c_star
                        p_support_c_star_per_feature[valid_c_star_mask, :] = p_support_c_star_per_feature_valid

                    mean_p_support_c_star_agg[valid_c_star_mask] = np.nan_to_num(
                        np.nanmean(p_support_c_star_per_feature_valid, axis=1), nan=0.0)
                    std_dev_support_c_star_agg[valid_c_star_mask] = np.nan_to_num(
                        np.nanstd(p_support_c_star_per_feature_valid, axis=1), nan=0.5)
                    q1_support_c_star_agg[valid_c_star_mask] = np.nan_to_num(
                        np.nanpercentile(p_support_c_star_per_feature_valid, 25, axis=1), nan=0.0)

                    p_clipped_valid = np.clip(p_support_c_star_per_feature_valid, self._epsilon, 1.0 - self._epsilon)
                    logit_p_support_valid = np.log(p_clipped_valid) - np.log(1.0 - p_clipped_valid)
                    q1_logit_support_c_star_agg_adapted[valid_c_star_mask] = np.nan_to_num(
                        np.nanpercentile(logit_p_support_valid, 25, axis=1), nan=default_q1_logit)

            normalized_entropy_if = None  # Per-feature normalized entropy: H(P(class | x_f)) / H_max
            if compute_per_feature_entropy and P_ifc is not None and n_features > 0:
                if n_classes > 1:
                    log2_P_ifc = np.log2(P_ifc + self._epsilon)  # Add epsilon for log stability
                    entropy_if = -np.sum(P_ifc * log2_P_ifc, axis=2)  # Shape (n_instances, n_features)
                    max_ent = np.log2(n_classes)  # Max possible entropy
                    normalized_entropy_if = entropy_if / max_ent if max_ent > 0 else np.zeros_like(entropy_if)
                else:  # n_classes <= 1, entropy is 0
                    normalized_entropy_if = np.zeros((n_instances, n_features))

            # --- Optional Metrics Calculations (using precomputed intermediates) ---
            if "ood_score_gap" in final_active_optional_metric_names:
                ood_score_gap_val = np.full(n_instances, 0.0)
                if n_classes >= 2:
                    sorted_logliks = np.sort(logliks_for_stats, axis=1)
                    valid_gaps = ~np.isnan(sorted_logliks[:, -1]) & ~np.isnan(sorted_logliks[:, -2])
                    gap = np.full(n_instances, np.nan)
                    gap[valid_gaps] = sorted_logliks[valid_gaps, -1] - sorted_logliks[valid_gaps, -2]
                    ood_score_gap_val = np.nan_to_num(-gap, nan=0.0)
                results_data["ood_score_gap"] = ood_score_gap_val

            if "ood_score_entropy" in final_active_optional_metric_names:
                ood_score_entropy_val = np.full(n_instances, 0.0)
                if n_classes > 0:
                    mask_meaningful = np.any(np.abs(total_log_likelihoods - self._log_zero_penalty) > 1e-9, axis=1)
                    probs = np.full_like(total_log_likelihoods, 1.0 / n_classes if n_classes > 0 else 0.0)
                    if np.any(mask_meaningful):
                        probs[mask_meaningful, :] = softmax(total_log_likelihoods[mask_meaningful, :], axis=1)
                    log_probs = np.log(probs + self._epsilon)
                    entropy = -np.sum(probs * log_probs, axis=1)
                    default_ent = np.log(n_classes) if n_classes > 1 else 0.0
                    ood_score_entropy_val = np.nan_to_num(entropy, nan=default_ent)
                results_data["ood_score_entropy"] = ood_score_entropy_val

            if "ood_score_mahalanobis" in final_active_optional_metric_names:
                ood_score_mahalanobis_val = np.full(n_instances, 0.0)
                if (self.likelihood_mean_vector_ is not None and
                        self.likelihood_inv_covariance_ is not None and
                        n_classes > 0 and self.likelihood_mean_vector_.shape[0] == n_classes):
                    ll_vectors_clean = np.nan_to_num(total_log_likelihoods, nan=self._log_zero_penalty,
                                                     posinf=self._log_zero_penalty, neginf=self._log_zero_penalty)
                    if ll_vectors_clean.shape[1] == self.likelihood_mean_vector_.shape[0]:  # Check dim match
                        delta = ll_vectors_clean - self.likelihood_mean_vector_
                        m_dist_sq = np.sum((delta @ self.likelihood_inv_covariance_) * delta, axis=1)
                        ood_score_mahalanobis_val = np.maximum(m_dist_sq, 0)  # Distance must be non-negative
                results_data["ood_score_mahalanobis"] = np.nan_to_num(ood_score_mahalanobis_val, nan=0.0)

            if "ood_metric1_avg_support_c_star" in final_active_optional_metric_names:
                results_data["ood_metric1_avg_support_c_star"] = 1.0 - mean_p_support_c_star_agg

            if "ood_metric2_feature_agreement" in final_active_optional_metric_names:
                ood_metric2_val = np.full(n_instances, 1.0)  # Default to max OOD (no agreement)
                if P_ifc is not None and n_features > 0 and n_classes > 0:
                    winner_if = np.argmax(P_ifc, axis=2)  # Feature's preferred class index
                    confidence_if = np.max(P_ifc, axis=2)  # Confidence in that preferred class
                    combined_strength = np.zeros(n_instances)
                    for i in range(n_instances):
                        if n_features == 0: continue  # Should be caught by outer if
                        mode_res = mode(winner_if[i, :], keepdims=False)  # Find most agreed-upon class by features
                        c_agree_i, count_agree_i = mode_res.mode, mode_res.count

                        # Handle mode output (can be array if multimodal)
                        if isinstance(c_agree_i, np.ndarray) and c_agree_i.size > 0:
                            c_agree_i, count_agree_i = c_agree_i[0], count_agree_i[0]  # Take first mode
                        elif not isinstance(c_agree_i, (int, float, np.number)):  # If mode is empty or unexpected
                            c_agree_i, count_agree_i = -1, 0  # No agreement

                        if count_agree_i > 0 and n_features > 0:
                            agreement_ratio_i = count_agree_i / n_features
                            mask_agreed_features = (winner_if[i, :] == c_agree_i)
                            avg_conf_for_agreed_class_i = np.mean(confidence_if[i, mask_agreed_features]) if np.sum(
                                mask_agreed_features) > 0 else 0.0
                            combined_strength[i] = agreement_ratio_i * avg_conf_for_agreed_class_i
                    ood_metric2_val = 1.0 - combined_strength
                results_data["ood_metric2_feature_agreement"] = ood_metric2_val

            if "ood_metric3_avg_feature_entropy" in final_active_optional_metric_names:
                ood_metric3_val = np.full(n_instances,
                                          1.0 if n_classes > 1 else 0.0)  # Default: max entropy if multiple classes, 0 otherwise
                if normalized_entropy_if is not None and n_features > 0:
                    if n_classes > 1:
                        ood_metric3_val = np.nan_to_num(np.nanmean(normalized_entropy_if, axis=1),
                                                        nan=1.0)  # Avg per-feature normalized entropy
                    elif n_classes == 1:  # Entropy is 0 for a single class
                        ood_metric3_val = np.full(n_instances, 0.0)
                    # if n_classes == 0, normalized_entropy_if is None or all zeros, default 0.0
                elif n_features == 0:  # No features, max entropy if multiple classes
                    ood_metric3_val = np.full(n_instances, 1.0 if n_classes > 1 else 0.0)
                # else n_classes == 0 and n_features > 0, default 0.0
                results_data["ood_metric3_avg_feature_entropy"] = ood_metric3_val

            if "ood_metric4_variance_support_c_star" in final_active_optional_metric_names:
                weight_std_dev = 0.5
                max_possible_std_dev_m4 = 0.5  # Approx max std dev for probs in [0,1]
                normalized_std_dev_m4 = np.clip(
                    std_dev_support_c_star_agg / max_possible_std_dev_m4 if max_possible_std_dev_m4 > 0 else 0, 0, 1.0)
                raw_score_m4 = (1.0 - mean_p_support_c_star_agg) + weight_std_dev * normalized_std_dev_m4
                results_data["ood_metric4_variance_support_c_star"] = np.clip(raw_score_m4 / (1.0 + weight_std_dev), 0,
                                                                              1.0)

            if "ood_metric5_robust_dissent_c_star" in final_active_optional_metric_names:
                results_data["ood_metric5_robust_dissent_c_star"] = 1.0 - q1_support_c_star_agg

            if "ood_metric5_adapted_robust_dissent_c_star" in final_active_optional_metric_names:
                results_data["ood_metric5_adapted_robust_dissent_c_star"] = 1.0 - sigmoid(
                    q1_logit_support_c_star_agg_adapted)

            # --- New Metrics (Idea 1, 2, 3) ---
            if "ood_metric6_q1_margin_c_star" in final_active_optional_metric_names:
                # Q1 of (LL_feature(c_star) - LL_feature(next_best_class_for_feature))
                # Default to 0 dissent (large positive margin -> sigmoid -> 0)
                q1_margins_val = np.full(n_instances, np.abs(self._log_zero_penalty / 10))  # Large positive margin
                if c_star_indices is not None and n_features > 0 and n_classes > 1:
                    feature_margins = np.full((n_instances, n_features), np.nan, dtype=np.float64)
                    valid_c_star_mask = (c_star_indices != -1)

                    for i_idx in np.where(valid_c_star_mask)[0]:  # Iterate only over instances with valid c_star
                        c_star_for_instance = c_star_indices[i_idx]
                        for f_idx in range(n_features):
                            ll_vector_feature = feature_log_likelihoods_per_instance[i_idx, f_idx, :]
                            ll_c_star_feature = ll_vector_feature[c_star_for_instance]

                            # Find next best log-likelihood for this feature, excluding c_star
                            competitor_ll = np.delete(ll_vector_feature, c_star_for_instance)
                            if competitor_ll.size > 0:
                                ll_next_best_feature = np.max(competitor_ll)
                                feature_margins[i_idx, f_idx] = ll_c_star_feature - ll_next_best_feature
                            else:  # Only one class, margin is effectively infinite (no competitor)
                                feature_margins[i_idx, f_idx] = np.abs(self._log_zero_penalty / 10)

                                # Calculate Q1 only for instances that had valid c_star and thus feature_margins computed
                    if np.any(valid_c_star_mask):
                        q1_margins_val[valid_c_star_mask] = np.nan_to_num(
                            np.nanpercentile(feature_margins[valid_c_star_mask, :], 25, axis=1),
                            nan=np.abs(self._log_zero_penalty / 10)  # Default if all margins for an instance are NaN
                        )
                # If n_classes <= 1, q1_margins_val remains default (large positive margin)
                results_data["ood_metric6_q1_margin_c_star"] = sigmoid(
                    -0.1 * q1_margins_val)  # k=0.1, dissent if margin is negative

            if "ood_metric7_entropy_weighted_dissent_c_star" in final_active_optional_metric_names:
                # Q1 of (p_support_c_star_per_feature * (1 - normalized_feature_entropy))
                q1_weighted_p_support = np.full(n_instances, 0.0)  # Default leads to max dissent (1.0)
                if (p_support_c_star_per_feature is not None and
                        normalized_entropy_if is not None and  # Requires n_features > 0 and n_classes > 1 for meaningful entropy
                        n_features > 0 and n_classes > 0):  # p_support needs n_classes > 0

                    decisiveness_if = 1.0 - normalized_entropy_if  # decisiveness_if is (I,F) or (I,F) if n_classes <=1
                    weighted_p_support = p_support_c_star_per_feature * decisiveness_if  # Element-wise
                    q1_weighted_p_support = np.nan_to_num(np.nanpercentile(weighted_p_support, 25, axis=1), nan=0.0)

                elif n_features > 0 and n_classes == 1:  # Special case: 1 class
                    # p_support_c_star_per_feature should be ~1.0 (if computed)
                    # normalized_entropy_if is 0, so decisiveness is 1.0
                    # Thus, weighted_p_support should be ~1.0
                    q1_weighted_p_support = np.full(n_instances, 1.0)

                results_data["ood_metric7_entropy_weighted_dissent_c_star"] = 1.0 - q1_weighted_p_support

            if "ood_metric8_prop_clear_dissent_c_star" in final_active_optional_metric_names:
                # Proportion of features where feature_top_choice != c_star AND LL(feature_top) > LL(c_star) + margin
                prop_dissenting = np.full(n_instances, 1.0)  # Default to max dissent
                if c_star_indices is not None and n_features > 0 and n_classes > 1:  # Need >1 class for dissent
                    num_clearly_dissenting = np.zeros(n_instances, dtype=int)
                    valid_c_star_mask = (c_star_indices != -1)

                    for i_idx in np.where(valid_c_star_mask)[0]:
                        c_star_for_instance = c_star_indices[i_idx]
                        for f_idx in range(n_features):
                            feature_ll_all_classes = feature_log_likelihoods_per_instance[i_idx, f_idx, :]
                            ll_c_star_feature = feature_ll_all_classes[c_star_for_instance]

                            feature_pref_class_idx = np.argmax(feature_ll_all_classes)
                            if feature_pref_class_idx != c_star_for_instance:  # Feature prefers a different class
                                ll_feature_pref = feature_ll_all_classes[feature_pref_class_idx]
                                if ll_feature_pref > (ll_c_star_feature + self.dissent_margin_threshold):
                                    num_clearly_dissenting[i_idx] += 1

                    # Calculate proportion only for instances with valid c_star
                    prop_dissenting[valid_c_star_mask] = num_clearly_dissenting[valid_c_star_mask] / n_features
                    # For instances with invalid c_star, prop_dissenting remains 1.0 (max dissent)

                elif n_features > 0 and n_classes <= 1:  # No basis for dissent if only 0 or 1 class
                    prop_dissenting = np.full(n_instances, 0.0)
                elif n_features == 0:  # No features, cannot dissent
                    prop_dissenting = np.full(n_instances, 0.0)

                results_data["ood_metric8_prop_clear_dissent_c_star"] = prop_dissenting

        # --- Final DataFrame Assembly ---
        final_data_for_df = {}
        for col_name in all_output_cols:  # all_output_cols is the definitive list of columns to output
            if col_name in results_data:
                final_data_for_df[col_name] = results_data[col_name]
            else:
                # This fallback should ideally not be hit if logic for active_metric_col_names is correct
                # and all computed metrics are added to results_data.
                default_val = self._log_zero_penalty if "loglik" in col_name else \
                    (1.0 if "ood" in col_name else 0.0)  # Default to max OOD for scores
                warnings.warn(
                    f"Column '{col_name}' was expected in output but not found in results_data. Defaulting value.",
                    UserWarning)
                final_data_for_df[col_name] = np.full(n_instances, default_val, dtype=np.float64)

        results_df_ordered = pl.DataFrame(final_data_for_df).select(all_output_cols)  # Ensure correct column order
        return results_df_ordered

