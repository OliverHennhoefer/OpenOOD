import numpy as np
import polars as pl
import polars.selectors as cs

from tqdm import tqdm
from scipy.special import softmax  # Already here
from scipy.stats import mode  # Added for Metric 2
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

        if n == 0:
            return 0.01

        std_dev = np.std(data)

        if n == 1:
            return 0.01

        if std_dev == 0:
            return 1e-6

        n_factor = n ** (-0.2)
        if method == "silverman":
            factor = 0.9 * std_dev * n_factor
        elif method == "scott":
            factor = 1.06 * std_dev * n_factor
        else:
            warnings.warn(f"Unknown bandwidth method '{method}', defaulting to 'silverman'.", UserWarning)
            factor = 0.9 * std_dev * n_factor
        return max(factor, 1e-6)

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
        if not isinstance(data, pl.DataFrame):
            raise TypeError("Input data must be a Polars DataFrame.")
        if label_col not in data.columns:
            raise ValueError(f"Label column '{label_col}' not found in DataFrame.")

        self.bandwidth_method_ = bandwidth_method

        try:
            potential_cols_selector = cs.contains(like)
            potential_cols_df = data.select(potential_cols_selector)
            # Exclude label_col AFTER selecting potential_cols_df to avoid error if label_col itself matches 'like'
            feature_cols_from_like = [col for col in potential_cols_df.columns if col != label_col]

            # Further filter to only numeric columns from the selection
            self.fitted_columns_ = data.select(feature_cols_from_like).select(cs.numeric()).columns

            non_numeric_excluded = set(feature_cols_from_like) - set(self.fitted_columns_)
            if non_numeric_excluded:
                warnings.warn(
                    f"Non-numeric columns matching pattern '{like}' (and not label_col) were excluded: {list(non_numeric_excluded)}",
                    UserWarning
                )
        except pl.exceptions.ColumnNotFoundError:
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
                feature_and_labels_df = data.select(
                    pl.col(col).alias("feature_value"),
                    pl.col(label_col).alias("label_value")
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
                    if bw < 1e-7:
                        warnings.warn(
                            f"Calculated bandwidth for col='{col}', label='{label}' is too small ({bw:.2e}). Skipping KDE.",
                            UserWarning)
                        continue

                    kde = KernelDensity(kernel=kernel, bandwidth=bw)
                    kde.fit(data_reshaped)
                    self.kdes_[col][label] = kde
                except Exception as e:
                    warnings.warn(f"Error fitting KDE for col='{col}', label='{label}'. Error: {e}. Storing None.",
                                  UserWarning)

        fitted_kde_count = sum(kde is not None for col_kdes in self.kdes_.values() for kde in col_kdes.values())
        if fitted_kde_count == 0:
            warnings.warn(
                "No KDEs were successfully fitted. Check data and parameters.",
                UserWarning)

        self.likelihood_mean_vector_ = None
        self.likelihood_inv_covariance_ = None
        if calculate_mahalanobis_params:
            if fitted_kde_count == 0:
                warnings.warn("Cannot calculate Mahalanobis params as no KDEs were fitted.", UserWarning)
                return self

            train_likelihood_df = self.transform(data, calculate_mahalanobis_score=False,
                                                 calculate_new_ood_metrics=False)  # Avoid recursion

            likelihood_cols_for_maha = [f"total_loglik_class_{lbl}" for lbl in self.normal_labels_]
            actual_cols = train_likelihood_df.columns
            missing_maha_cols = [lc for lc in likelihood_cols_for_maha if lc not in actual_cols]

            if missing_maha_cols:
                warnings.warn(
                    f"Missing likelihood columns for Mahalanobis: {missing_maha_cols}. Cannot calculate params.",
                    UserWarning)
                return self

            if not likelihood_cols_for_maha:  # No labels, hence no likelihood columns
                warnings.warn("No normal labels found, cannot calculate Mahalanobis parameters.", UserWarning)
                return self

            train_likelihood_vectors = train_likelihood_df.select(likelihood_cols_for_maha).to_numpy()
            train_likelihood_vectors = np.nan_to_num(
                train_likelihood_vectors, nan=self._log_zero_penalty,
                posinf=self._log_zero_penalty, neginf=self._log_zero_penalty
            )

            if train_likelihood_vectors.shape[0] < 2 or train_likelihood_vectors.shape[1] == 0:
                warnings.warn(
                    "Insufficient data for Mahalanobis parameters reliably (need at least 2 samples and 1 class).",
                    UserWarning)  # Corrected warning
                return self

            try:
                self.likelihood_mean_vector_ = np.mean(train_likelihood_vectors, axis=0)
                covariance = np.cov(train_likelihood_vectors, rowvar=False)
                if covariance.ndim == 0: covariance = np.array([[covariance]])  # Handle 1-class case
                ridge_epsilon = 1e-6
                covariance_reg = covariance + np.eye(covariance.shape[0]) * ridge_epsilon
                self.likelihood_inv_covariance_ = np.linalg.pinv(covariance_reg)
            except Exception as e:
                warnings.warn(f"Error calculating Mahalanobis parameters: {e}. Parameters not stored.", UserWarning)
                self.likelihood_mean_vector_ = None
                self.likelihood_inv_covariance_ = None
        return self

    def transform(
            self,
            inference_data: pl.DataFrame,
            calculate_mahalanobis_score: bool = True,
            calculate_new_ood_metrics: bool = True
    ) -> pl.DataFrame:
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

        likelihood_col_names = [f"total_loglik_class_{lbl}" for lbl in self.normal_labels_]
        metric_col_names = [
            "max_log_likelihood", "mean_log_likelihood",
            "ood_score_likelihood_difference", "ood_score_gap",
            "ood_score_entropy", "ood_score_mahalanobis",
            "ood_metric1_avg_support_c_star",
            "ood_metric2_feature_agreement",
            "ood_metric3_avg_feature_entropy",
            "ood_metric4_variance_support_c_star",
            "ood_metric5_robust_dissent_c_star"  # New metric name
        ]
        if not calculate_mahalanobis_score:
            metric_col_names = [m for m in metric_col_names if m != "ood_score_mahalanobis"]
        if not calculate_new_ood_metrics:
            new_metrics_to_remove = [
                "ood_metric1_avg_support_c_star",
                "ood_metric2_feature_agreement",
                "ood_metric3_avg_feature_entropy",
                "ood_metric4_variance_support_c_star",
                "ood_metric5_robust_dissent_c_star"  # New metric name
            ]
            metric_col_names = [m for m in metric_col_names if m not in new_metrics_to_remove]

        all_output_cols = likelihood_col_names + metric_col_names

        if n_instances == 0:
            schema = {col_name: pl.Float64 for col_name in all_output_cols}
            return pl.DataFrame(schema=schema)

        if not self.fitted_columns_:
            inference_features_np = np.empty((n_instances, 0), dtype=np.float64)
        else:
            select_exprs = [pl.col(col).cast(pl.Float64, strict=False) for col in self.fitted_columns_]
            inference_features_np = inference_data.select(select_exprs).to_numpy()

        total_log_likelihoods = np.full((n_instances, n_classes), 0.0, dtype=np.float64)
        feature_log_likelihoods_per_instance = np.full(
            (n_instances, n_features, n_classes), self._log_zero_penalty, dtype=np.float64
        )

        if n_features == 0 or n_classes == 0:
            if n_classes > 0:
                total_log_likelihoods[:] = self._log_zero_penalty * (n_features if n_features > 0 else 1)
        else:
            for col_idx, col_name in enumerate(
                    tqdm(self.fitted_columns_, desc="Scoring Instances", unit="feature", leave=False,
                         dynamic_ncols=True)):
                current_feature_values = inference_features_np[:, col_idx]
                current_feature_values_reshaped = current_feature_values.reshape(-1, 1)
                mask_finite_input = np.isfinite(current_feature_values)

                for class_idx, label in enumerate(self.normal_labels_):
                    kde = self.kdes_[col_name].get(label)
                    log_likes_this_feature_this_class = np.full(n_instances, self._log_zero_penalty, dtype=float)

                    if kde is not None:
                        finite_values_to_score = current_feature_values_reshaped[mask_finite_input]
                        if finite_values_to_score.size > 0:
                            scored_finite_values = kde.score_samples(finite_values_to_score)
                            scored_finite_values_clean = np.nan_to_num(
                                scored_finite_values, nan=self._log_zero_penalty,
                                posinf=self._log_zero_penalty, neginf=self._log_zero_penalty
                            )
                            log_likes_this_feature_this_class[mask_finite_input] = scored_finite_values_clean

                    feature_log_likelihoods_per_instance[:, col_idx, class_idx] = log_likes_this_feature_this_class
                    total_log_likelihoods[:, class_idx] += log_likes_this_feature_this_class

        results_data = {}
        if self.normal_labels_:
            for i, lbl_col_name in enumerate(likelihood_col_names):
                results_data[lbl_col_name] = total_log_likelihoods[:, i]

        logliks_for_stats = np.where(total_log_likelihoods <= (self._log_zero_penalty + 1e-9),
                                     np.nan, total_log_likelihoods)
        if n_classes == 0: logliks_for_stats = np.empty((n_instances, 0))

        with np.errstate(invalid='ignore', divide='ignore'):
            if n_classes > 0:
                max_log_likelihood = np.nanmax(logliks_for_stats, axis=1)
                mean_log_likelihood = np.nanmean(logliks_for_stats, axis=1)
            else:
                max_log_likelihood = np.full(n_instances, np.nan)
                mean_log_likelihood = np.full(n_instances, np.nan)

            results_data["max_log_likelihood"] = np.nan_to_num(max_log_likelihood, nan=self._log_zero_penalty)
            results_data["mean_log_likelihood"] = np.nan_to_num(mean_log_likelihood, nan=self._log_zero_penalty)
            results_data["ood_score_likelihood_difference"] = -(
                        results_data["max_log_likelihood"] - results_data["mean_log_likelihood"])

            ood_score_gap = np.full(n_instances, 0.0)
            if n_classes >= 2:
                sorted_logliks = np.sort(logliks_for_stats, axis=1)
                valid_gaps = ~np.isnan(sorted_logliks[:, -1]) & ~np.isnan(sorted_logliks[:, -2])
                gap = np.full(n_instances, np.nan)
                gap[valid_gaps] = sorted_logliks[valid_gaps, -1] - sorted_logliks[valid_gaps, -2]
                ood_score_gap = np.nan_to_num(-gap, nan=0.0)
            results_data["ood_score_gap"] = ood_score_gap

            ood_score_entropy = np.full(n_instances, 0.0)
            if n_classes > 0:
                mask_meaningful_ll_rows = np.any(total_log_likelihoods > (self._log_zero_penalty + 1e-9), axis=1)
                probabilities = np.full_like(total_log_likelihoods, 1.0 / n_classes if n_classes > 0 else 0.0)
                if np.any(mask_meaningful_ll_rows):
                    meaningful_logliks = total_log_likelihoods[mask_meaningful_ll_rows, :]
                    probabilities[mask_meaningful_ll_rows, :] = softmax(meaningful_logliks, axis=1)
                log_probabilities = np.log(probabilities + 1e-30)
                entropy_values = -np.sum(probabilities * log_probabilities, axis=1)
                default_entropy_val = np.log(n_classes) if n_classes > 1 else 0.0
                ood_score_entropy = np.nan_to_num(entropy_values, nan=default_entropy_val)
            results_data["ood_score_entropy"] = ood_score_entropy

            if calculate_mahalanobis_score:
                ood_score_mahalanobis = np.full(n_instances, 0.0)
                if (self.likelihood_mean_vector_ is not None and
                        self.likelihood_inv_covariance_ is not None and
                        n_classes > 0 and
                        self.likelihood_mean_vector_.shape[0] == n_classes):
                    likelihood_vectors_clean = np.nan_to_num(
                        total_log_likelihoods, nan=self._log_zero_penalty,
                        posinf=self._log_zero_penalty, neginf=self._log_zero_penalty
                    )
                    if likelihood_vectors_clean.shape[1] == self.likelihood_mean_vector_.shape[0]:
                        delta = likelihood_vectors_clean - self.likelihood_mean_vector_
                        term1 = delta @ self.likelihood_inv_covariance_
                        m_dist_sq = np.sum(term1 * delta, axis=1)
                        ood_score_mahalanobis = np.maximum(m_dist_sq, 0)
                    else:
                        warnings.warn("Dimension mismatch for Mahalanobis. Skipping.", UserWarning)
                results_data["ood_score_mahalanobis"] = np.nan_to_num(ood_score_mahalanobis, nan=0.0)

            if calculate_new_ood_metrics:
                P_ifc = np.empty((n_instances, n_features, n_classes), dtype=np.float64)
                if n_features > 0 and n_classes > 0:
                    P_ifc = softmax(feature_log_likelihoods_per_instance, axis=2)
                elif n_features > 0:  # n_classes is 0
                    P_ifc = np.empty((n_instances, n_features, 0), dtype=np.float64)
                # if n_features == 0, P_ifc remains as initially declared (n_instances, 0, n_classes)

                # --- Shared computations for Metrics 1, 4, 5 (depend on c_star) ---
                # Initialize defaults for values derived from p_support_c_star_per_feature.
                # These defaults lead to OOD scores of 1.0 if n_features or n_classes is 0.
                mean_p_support_c_star_agg = np.full(n_instances, 0.0, dtype=np.float64)
                std_dev_support_c_star_agg = np.full(n_instances, 0.5,
                                                     dtype=np.float64)  # Default to max_possible_std_dev_approx
                q1_support_c_star_agg = np.full(n_instances, 0.0, dtype=np.float64)

                if n_features > 0 and n_classes > 0:
                    c_star_indices = np.argmax(total_log_likelihoods, axis=1)
                    idx_i = np.arange(n_instances)[:, None]
                    idx_j = np.arange(n_features)[None, :]
                    idx_k_c_star = c_star_indices[:, None]

                    # p_support_c_star_per_feature: prob assigned by each feature to the global best class c*
                    # Shape: (n_instances, n_features)
                    p_support_c_star_per_feature = P_ifc[idx_i, idx_j, idx_k_c_star]

                    current_mean_support = np.nanmean(p_support_c_star_per_feature, axis=1)
                    mean_p_support_c_star_agg = np.nan_to_num(current_mean_support, nan=0.0)

                    max_possible_std_dev_approx = 0.5  # For normalization and NaN replacement
                    current_std_dev_support = np.nanstd(p_support_c_star_per_feature, axis=1)
                    std_dev_support_c_star_agg = np.nan_to_num(current_std_dev_support, nan=max_possible_std_dev_approx)

                    current_q1_support = np.nanpercentile(p_support_c_star_per_feature, 25, axis=1)
                    q1_support_c_star_agg = np.nan_to_num(current_q1_support, nan=0.0)

                # Metric 1: Average Support for the "Best Guess" Overall Class
                ood_metric1 = 1.0 - mean_p_support_c_star_agg
                results_data["ood_metric1_avg_support_c_star"] = ood_metric1

                # Metric 2: Feature Agreement & Confidence Score
                ood_metric2 = np.full(n_instances, 1.0)
                if n_features > 0 and n_classes > 0:  # P_ifc is (I,F,C)
                    winner_if = np.argmax(P_ifc, axis=2)
                    confidence_if = np.max(P_ifc, axis=2)
                    combined_strength = np.zeros(n_instances)
                    for i in range(n_instances):
                        if n_features == 0: continue  # Should be caught by outer if, but defensive
                        mode_res = mode(winner_if[i, :], keepdims=False)
                        c_agree_i = mode_res.mode
                        count_agree_i = mode_res.count

                        if count_agree_i == 0:  # or n_features == 0 (covered by outer if)
                            agreement_ratio_i = 0.0
                            avg_confidence_for_agreed_class_i = 0.0
                        else:
                            agreement_ratio_i = count_agree_i / n_features
                            mask_agreed_features = (winner_if[i, :] == c_agree_i)
                            if np.sum(mask_agreed_features) > 0:
                                avg_confidence_for_agreed_class_i = np.mean(confidence_if[i, mask_agreed_features])
                            else:
                                avg_confidence_for_agreed_class_i = 0.0
                        combined_strength[i] = agreement_ratio_i * avg_confidence_for_agreed_class_i
                    ood_metric2 = 1.0 - combined_strength
                results_data["ood_metric2_feature_agreement"] = ood_metric2

                # Metric 3: Average Per-Feature Uncertainty (Entropy)
                ood_metric3 = np.full(n_instances, 0.0)
                if n_features > 0:
                    if n_classes > 1:  # P_ifc is (I,F,C)
                        log2_P_ifc = np.log2(P_ifc + 1e-30)
                        entropy_if = -np.sum(P_ifc * log2_P_ifc, axis=2)
                        max_entropy = np.log2(n_classes)
                        normalized_entropy_if = entropy_if / max_entropy
                        avg_normalized_entropy = np.nanmean(normalized_entropy_if, axis=1)
                        ood_metric3 = np.nan_to_num(avg_normalized_entropy, nan=1.0)
                    elif n_classes == 1:
                        ood_metric3 = np.full(n_instances, 0.0)
                    # if n_classes == 0, P_ifc has last dim 0, sum is 0, so ood_metric3 remains 0.0.
                else:  # No features
                    ood_metric3 = np.full(n_instances, 1.0 if n_classes > 1 else 0.0)
                results_data["ood_metric3_avg_feature_entropy"] = ood_metric3

                # Metric 4: Variance of Top Class Probabilities Across Features
                weight_std_dev = 0.5
                max_possible_std_dev_approx = 0.5  # Used in shared calcs too

                normalized_std_dev_m4 = np.clip(std_dev_support_c_star_agg / max_possible_std_dev_approx, 0, 1.0)
                raw_score_m4 = (1.0 - mean_p_support_c_star_agg) + weight_std_dev * normalized_std_dev_m4
                ood_metric4 = np.clip(raw_score_m4 / (1.0 + weight_std_dev), 0, 1.0)
                results_data["ood_metric4_variance_support_c_star"] = ood_metric4

                # New Metric 5: Robust Dissent for c_star (based on Q1 support)
                ood_metric5_robust_dissent_c_star = 1.0 - q1_support_c_star_agg
                results_data["ood_metric5_robust_dissent_c_star"] = ood_metric5_robust_dissent_c_star

        results_df = pl.DataFrame(results_data)
        ordered_cols_present = [col for col in all_output_cols if col in results_df.columns]
        return results_df.select(ordered_cols_present)