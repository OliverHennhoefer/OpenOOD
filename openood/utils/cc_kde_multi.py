import numpy as np
import polars as pl
import polars.selectors as cs

from tqdm import tqdm
from scipy.special import softmax, expit as sigmoid
from scipy.stats import mode
from sklearn.neighbors import KernelDensity
from typing import Literal, List, Dict, Optional, Any
import warnings
from joblib import Parallel, delayed


class DimensionWiseKdeOODMulti:
    """
    Performs Out-of-Distribution (OOD) detection based on comparing instance
    likelihoods across dimension-wise Kernel Density Estimates (KDEs) fitted
    for known normal classes. Supports parallel processing for fitting and transforming.
    """
    # --- Metric Definitions and Dependencies (adopted from reference DimensionWiseKdeOOD) ---
    ALWAYS_CALCULATED_METRIC_NAMES = [
        "max_log_likelihood", "mean_log_likelihood",
        "ood_score_likelihood_difference"
    ]
    ORDERED_OPTIONAL_METRIC_NAMES = [
        "ood_score_gap", "ood_score_entropy", "ood_score_mahalanobis",
        "ood_metric1_avg_support_c_star", "ood_metric2_feature_agreement",
        "ood_metric3_avg_feature_entropy", "ood_metric4_variance_support_c_star",
        "ood_metric5_robust_dissent_c_star", "ood_metric5_adapted_robust_dissent_c_star",
        "ood_metric6_q1_margin_c_star",
        "ood_metric7_entropy_weighted_dissent_c_star",
        "ood_metric8_prop_clear_dissent_c_star"
    ]
    KNOWN_OPTIONAL_METRIC_NAMES_SET = set(ORDERED_OPTIONAL_METRIC_NAMES)

    REQUIRES_C_STAR_INDICES_METRICS = {
        "ood_metric1_avg_support_c_star", "ood_metric4_variance_support_c_star",
        "ood_metric5_robust_dissent_c_star", "ood_metric5_adapted_robust_dissent_c_star",
        "ood_metric6_q1_margin_c_star", "ood_metric7_entropy_weighted_dissent_c_star",
        "ood_metric8_prop_clear_dissent_c_star"
    }
    REQUIRES_P_IFC_METRICS = {
        "ood_metric1_avg_support_c_star", "ood_metric2_feature_agreement",
        "ood_metric3_avg_feature_entropy", "ood_metric4_variance_support_c_star",
        "ood_metric5_robust_dissent_c_star", "ood_metric5_adapted_robust_dissent_c_star",
        "ood_metric7_entropy_weighted_dissent_c_star"
    }
    REQUIRES_P_SUPPORT_C_STAR_AGGREGATES_METRICS = {
        "ood_metric1_avg_support_c_star", "ood_metric4_variance_support_c_star",
        "ood_metric5_robust_dissent_c_star", "ood_metric5_adapted_robust_dissent_c_star",
        "ood_metric7_entropy_weighted_dissent_c_star"
    }
    REQUIRES_PER_FEATURE_ENTROPY_METRICS = {
        "ood_metric3_avg_feature_entropy", "ood_metric7_entropy_weighted_dissent_c_star"
    }

    # --- End Metric Definitions ---

    def __init__(self, dissent_margin_threshold: float = 1.0):
        """
        Initializes the DimensionWiseKdeOODMulti object.
        Args:
            dissent_margin_threshold (float): The log-likelihood margin required for a feature
                                              to be considered as "clearly dissenting" in
                                              ood_metric8_prop_clear_dissent_c_star.
        """
        self.kdes_: Optional[Dict[str, Dict[Any, Optional[KernelDensity]]]] = None
        self.fitted_columns_: Optional[List[str]] = None
        self.normal_labels_: Optional[List[Any]] = None
        self.bandwidth_method_: Optional[str] = None
        self._log_zero_penalty: float = -1e30
        self.likelihood_mean_vector_: Optional[np.ndarray] = None
        self.likelihood_inv_covariance_: Optional[np.ndarray] = None
        self._epsilon: float = 1e-30
        self.dissent_margin_threshold: float = dissent_margin_threshold

    @staticmethod
    def _calculate_bandwidth(data: np.ndarray, method: str) -> float:
        """
        Calculates bandwidth for KDE using Silverman's or Scott's rule.
        """
        n = len(data)
        if n == 0: return 0.01
        std_dev = np.std(data)
        if n == 1: return 0.01
        if std_dev == 0: return 1e-6

        n_factor = n ** (-0.2)
        if method == "silverman":
            factor = 0.9 * std_dev * n_factor
        elif method == "scott":
            factor = 1.06 * std_dev * n_factor
        else:
            warnings.warn(f"Unknown bandwidth method '{method}', defaulting to 'silverman'.", UserWarning)
            factor = 0.9 * std_dev * n_factor
        return max(factor, 1e-6)

    @staticmethod
    def _fit_kde_joblib_worker(
            col_name: str, label: Any, data_reshaped: np.ndarray,
            bandwidth: float, kernel_str: str
    ) -> tuple[str, Any, Optional[KernelDensity], Optional[str]]:
        """Worker function to fit a single KDE for joblib."""
        try:
            kde = KernelDensity(kernel=kernel_str, bandwidth=bandwidth)
            kde.fit(data_reshaped)
            return col_name, label, kde, None
        except Exception as e:
            return col_name, label, None, str(e)

    def fit(
            self,
            data: pl.DataFrame,
            like: str | list[str],
            label_col: str,
            bandwidth_method: Literal["silverman", "scott"] = "silverman",
            kernel: str = "gaussian",
            min_samples_for_kde: int = 5,
            calculate_mahalanobis_params: bool = False,
            n_jobs: int = -1,
            joblib_verbose: int = 0
    ):
        if not isinstance(data, pl.DataFrame):
            raise TypeError("Input data must be a Polars DataFrame.")
        if label_col not in data.columns:
            raise ValueError(f"Label column '{label_col}' not found in DataFrame.")

        self.bandwidth_method_ = bandwidth_method

        try:
            potential_cols_selector = cs.contains(like)
            potential_cols_df = data.select(potential_cols_selector)
            feature_cols_from_like = [col for col in potential_cols_df.columns if col != label_col]
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

        kde_fitting_tasks_params = []
        for col in tqdm(self.fitted_columns_, desc="Preparing KDE fit tasks", unit="feature", leave=False):
            try:
                feature_and_labels_df = data.select(
                    pl.col(col).alias("feature_value"),
                    pl.col(label_col).alias("label_value")
                ).filter(pl.col("feature_value").is_not_null())
                if feature_and_labels_df.is_empty(): continue
            except pl.exceptions.PolarsError as e:
                warnings.warn(f"Error preparing data for column '{col}': {e}. Skipping this column.", UserWarning)
                continue

            for label in self.normal_labels_:
                class_feature_data_series = feature_and_labels_df.filter(
                    pl.col("label_value") == label
                ).select("feature_value").to_series()

                if len(class_feature_data_series) < min_samples_for_kde: continue

                class_col_data_np = class_feature_data_series.to_numpy()
                bw = self._calculate_bandwidth(class_col_data_np, self.bandwidth_method_)
                if bw < 1e-7:
                    warnings.warn(
                        f"Calculated bandwidth for col='{col}', label='{label}' is too small ({bw:.2e}). Skipping KDE.",
                        UserWarning)
                    continue
                data_reshaped = class_col_data_np.reshape(-1, 1)
                kde_fitting_tasks_params.append((col, label, data_reshaped, bw, kernel))

        if kde_fitting_tasks_params:
            with Parallel(n_jobs=n_jobs, verbose=joblib_verbose) as parallel:
                results = []
                for result_tuple in tqdm(
                        parallel(
                            delayed(DimensionWiseKdeOODMulti._fit_kde_joblib_worker)(*params)
                            for params in kde_fitting_tasks_params
                        ),
                        total=len(kde_fitting_tasks_params),
                        desc="Fitting KDEs (parallel)",
                        unit="KDE",
                        leave=False
                ):
                    results.append(result_tuple)

            for res_col, res_label, res_kde, res_error_msg in results:
                if res_error_msg:
                    warnings.warn(
                        f"Error fitting KDE for col='{res_col}', label='{res_label}'. Error: {res_error_msg}. Storing None.",
                        UserWarning)
                else:
                    self.kdes_[res_col][res_label] = res_kde

        fitted_kde_count = sum(kde is not None for col_kdes in self.kdes_.values() for kde in col_kdes.values())
        if fitted_kde_count == 0:
            warnings.warn("No KDEs were successfully fitted. Check data and parameters.", UserWarning)

        self.likelihood_mean_vector_ = None
        self.likelihood_inv_covariance_ = None
        if calculate_mahalanobis_params:
            if fitted_kde_count == 0:
                warnings.warn("Cannot calculate Mahalanobis params as no KDEs were fitted.", UserWarning)
                return self

            train_likelihood_df = self.transform(
                data,
                metrics_to_compute=[],  # Only compute base likelihoods and always-on metrics
                n_jobs=n_jobs,
                joblib_verbose=joblib_verbose
            )

            likelihood_cols_for_maha = [f"total_loglik_class_{lbl}" for lbl in self.normal_labels_]
            actual_cols = train_likelihood_df.columns
            missing_maha_cols = [lc for lc in likelihood_cols_for_maha if lc not in actual_cols]

            if missing_maha_cols or not likelihood_cols_for_maha:
                warnings.warn(
                    f"Missing or no likelihood columns for Mahalanobis: {missing_maha_cols}. Cannot calculate params.",
                    UserWarning)
                return self

            train_likelihood_vectors = train_likelihood_df.select(likelihood_cols_for_maha).to_numpy()
            train_likelihood_vectors = np.nan_to_num(
                train_likelihood_vectors, nan=self._log_zero_penalty,
                posinf=self._log_zero_penalty, neginf=self._log_zero_penalty
            )

            if train_likelihood_vectors.shape[0] < 2 or train_likelihood_vectors.shape[1] == 0:
                warnings.warn(
                    "Insufficient data for Mahalanobis parameters reliably (need at least 2 samples and 1 class).",
                    UserWarning)
                return self

            try:
                self.likelihood_mean_vector_ = np.mean(train_likelihood_vectors, axis=0)
                covariance = np.cov(train_likelihood_vectors, rowvar=False)
                if covariance.ndim == 0: covariance = np.array([[covariance]])
                ridge_epsilon = 1e-6
                covariance_reg = covariance + np.eye(covariance.shape[0]) * ridge_epsilon
                self.likelihood_inv_covariance_ = np.linalg.pinv(covariance_reg)
            except Exception as e:
                warnings.warn(f"Error calculating Mahalanobis parameters: {e}. Parameters not stored.", UserWarning)
                self.likelihood_mean_vector_ = None
                self.likelihood_inv_covariance_ = None
        return self

    @staticmethod
    def _score_column_joblib_worker(
            feature_values_for_column_slice: np.ndarray,
            kdes_for_this_col: Dict[Any, Optional[KernelDensity]],
            normal_labels_order: List[Any],
            log_zero_penalty: float,
            n_total_instances: int
    ) -> np.ndarray:
        """Worker function to score instances for a single feature using joblib."""
        current_feature_values_reshaped = feature_values_for_column_slice.reshape(-1, 1)
        mask_finite_input = np.isfinite(feature_values_for_column_slice)

        log_likelihoods_contribution_this_feature = np.full(
            (n_total_instances, len(normal_labels_order)), log_zero_penalty, dtype=np.float64
        )

        for class_idx, label in enumerate(normal_labels_order):
            kde = kdes_for_this_col.get(label)
            if kde is not None:
                finite_values_to_score = current_feature_values_reshaped[mask_finite_input]
                if finite_values_to_score.size > 0:
                    try:
                        scored_finite_values = kde.score_samples(finite_values_to_score)
                        scored_finite_values_clean = np.nan_to_num(
                            scored_finite_values, nan=log_zero_penalty,
                            posinf=log_zero_penalty, neginf=log_zero_penalty
                        )
                        log_likelihoods_contribution_this_feature[
                            mask_finite_input, class_idx] = scored_finite_values_clean
                    except Exception:
                        pass
        return log_likelihoods_contribution_this_feature

    def transform(
            self,
            inference_data: pl.DataFrame,
            metrics_to_compute: Optional[List[str]] = None,
            n_jobs: int = -1,
            joblib_verbose: int = 0
    ) -> pl.DataFrame:
        # --- Initial Checks and Setup ---
        if not all(hasattr(self, attr) and getattr(self, attr) is not None for attr in
                   ['kdes_', 'fitted_columns_', 'normal_labels_']):
            raise RuntimeError("The '.fit()' method must be called before '.transform()'.")
        if not isinstance(inference_data, pl.DataFrame):
            raise TypeError("Input inference_data must be a Polars DataFrame.")
        if self.fitted_columns_ is None: self.fitted_columns_ = []

        missing_cols = [col for col in self.fitted_columns_ if col not in inference_data.columns]
        if missing_cols:
            raise ValueError(f"Inference data is missing required (fitted) columns: {missing_cols}")

        n_instances = len(inference_data)
        n_classes = len(self.normal_labels_)
        n_features = len(self.fitted_columns_)

        # --- Determine Active Metrics and Output Columns ---
        likelihood_col_names = [f"total_loglik_class_{lbl}" for lbl in self.normal_labels_]

        active_metric_col_names = list(self.ALWAYS_CALCULATED_METRIC_NAMES)
        runnable_optional_metrics_set = set()

        if metrics_to_compute is None:
            runnable_optional_metrics_set.update(self.KNOWN_OPTIONAL_METRIC_NAMES_SET)
        else:
            for m_name in metrics_to_compute:
                if m_name in self.KNOWN_OPTIONAL_METRIC_NAMES_SET:
                    runnable_optional_metrics_set.add(m_name)
                elif m_name not in self.ALWAYS_CALCULATED_METRIC_NAMES:
                    warnings.warn(
                        f"Requested metric '{m_name}' is unknown or not an optional metric. It will be ignored.",
                        UserWarning)

        final_active_optional_metric_names = []
        for m_name in self.ORDERED_OPTIONAL_METRIC_NAMES:
            if m_name in runnable_optional_metrics_set:
                if m_name == "ood_score_mahalanobis":
                    if self.likelihood_mean_vector_ is not None and self.likelihood_inv_covariance_ is not None:
                        final_active_optional_metric_names.append(m_name)
                    elif metrics_to_compute is not None and m_name in metrics_to_compute:
                        warnings.warn(
                            f"Metric '{m_name}' explicitly requested but its parameters were not fitted. It will not be included.",
                            UserWarning)
                else:
                    final_active_optional_metric_names.append(m_name)
        active_metric_col_names.extend(final_active_optional_metric_names)
        all_output_cols = likelihood_col_names + active_metric_col_names

        if n_instances == 0:
            return pl.DataFrame(schema={col_name: pl.Float64 for col_name in all_output_cols})

        # --- Feature Log-Likelihood Calculation (Parallel) ---
        inference_features_np = np.empty((n_instances, 0), dtype=np.float64)
        if self.fitted_columns_:
            select_exprs = [pl.col(col).cast(pl.Float64, strict=False) for col in self.fitted_columns_]
            inference_features_np = inference_data.select(select_exprs).to_numpy()

        total_log_likelihoods = np.full((n_instances, n_classes), 0.0, dtype=np.float64)
        feature_log_likelihoods_per_instance = np.full(
            (n_instances, n_features, n_classes), self._log_zero_penalty, dtype=np.float64
        )

        if n_features > 0 and n_classes > 0:
            scoring_tasks_params = []
            for col_idx, col_name in enumerate(self.fitted_columns_):
                task_args = (
                    inference_features_np[:, col_idx].copy(),
                    self.kdes_[col_name],
                    self.normal_labels_,
                    self._log_zero_penalty,
                    n_instances
                )
                scoring_tasks_params.append(task_args)

            if scoring_tasks_params:
                with Parallel(n_jobs=n_jobs, verbose=joblib_verbose, backend="loky") as parallel:
                    per_column_results_list = []
                    for res_array in tqdm(
                            parallel(
                                delayed(DimensionWiseKdeOODMulti._score_column_joblib_worker)(*params)
                                for params in scoring_tasks_params
                            ),
                            total=len(scoring_tasks_params),
                            desc="Scoring Instances (parallel)",
                            unit="feature",
                            leave=False,
                            dynamic_ncols=True
                    ):
                        per_column_results_list.append(res_array)

                if per_column_results_list:
                    raw_feature_ll_stacked_features_first = np.array(per_column_results_list)
                    total_log_likelihoods = np.sum(raw_feature_ll_stacked_features_first, axis=0)
                    feature_log_likelihoods_per_instance = np.transpose(raw_feature_ll_stacked_features_first,
                                                                        (1, 0, 2))
        elif n_features == 0 and n_classes > 0:
            total_log_likelihoods[:] = self._log_zero_penalty

        # --- Prepare results_data dictionary ---
        results_data = {}
        if self.normal_labels_:
            for i, lbl_col_name in enumerate(likelihood_col_names):
                results_data[lbl_col_name] = total_log_likelihoods[:, i]

        logliks_for_stats = np.where(np.abs(total_log_likelihoods - self._log_zero_penalty) < 1e-9,
                                     np.nan, total_log_likelihoods)
        if n_classes == 0: logliks_for_stats = np.empty((n_instances, 0))

        with np.errstate(invalid='ignore', divide='ignore'):
            # --- Always Calculated Metrics ---
            if n_classes > 0:
                max_log_likelihood_val = np.nanmax(logliks_for_stats, axis=1)
                mean_log_likelihood_val = np.nanmean(logliks_for_stats, axis=1)
            else:
                max_log_likelihood_val = np.full(n_instances, np.nan)
                mean_log_likelihood_val = np.full(n_instances, np.nan)

            results_data["max_log_likelihood"] = np.nan_to_num(max_log_likelihood_val, nan=self._log_zero_penalty)
            results_data["mean_log_likelihood"] = np.nan_to_num(mean_log_likelihood_val, nan=self._log_zero_penalty)
            results_data["ood_score_likelihood_difference"] = -(
                    results_data["max_log_likelihood"] - results_data["mean_log_likelihood"])

            # --- Intermediate Computations for Optional Metrics ---
            current_runnable_optional_metrics = set(final_active_optional_metric_names)
            compute_c_star_indices = any(
                m in current_runnable_optional_metrics for m in self.REQUIRES_C_STAR_INDICES_METRICS)
            compute_P_ifc = any(m in current_runnable_optional_metrics for m in self.REQUIRES_P_IFC_METRICS)
            compute_p_support_c_star_aggregates = any(
                m in current_runnable_optional_metrics for m in self.REQUIRES_P_SUPPORT_C_STAR_AGGREGATES_METRICS)
            compute_per_feature_entropy = any(
                m in current_runnable_optional_metrics for m in self.REQUIRES_PER_FEATURE_ENTROPY_METRICS)

            c_star_indices = None
            if compute_c_star_indices and n_classes > 0:
                safe_total_log_likelihoods = np.where(
                    np.all(total_log_likelihoods <= (self._log_zero_penalty + 1e-9), axis=1, keepdims=True),
                    np.zeros_like(total_log_likelihoods), total_log_likelihoods)
                c_star_indices = np.argmax(safe_total_log_likelihoods, axis=1)
            elif compute_c_star_indices:
                c_star_indices = np.full(n_instances, -1, dtype=int)

            P_ifc = None
            if compute_P_ifc:
                P_ifc = np.empty((n_instances, n_features, n_classes), dtype=np.float64)
                if n_features > 0 and n_classes > 0:
                    P_ifc = softmax(feature_log_likelihoods_per_instance, axis=2)

            mean_p_support_c_star_agg = np.full(n_instances, 0.0)
            std_dev_support_c_star_agg = np.full(n_instances, 0.5)
            q1_support_c_star_agg = np.full(n_instances, 0.0)
            default_q1_logit = np.log(self._epsilon / (1.0 - self._epsilon)) if (1.0 - self._epsilon) > 0 else -70.0
            q1_logit_support_c_star_agg_adapted = np.full(n_instances, default_q1_logit)
            p_support_c_star_per_feature = np.full((n_instances, n_features), np.nan, dtype=np.float64)

            if compute_p_support_c_star_aggregates and P_ifc is not None and c_star_indices is not None and n_features > 0 and n_classes > 0:
                valid_c_star_mask = (c_star_indices != -1)
                if np.any(valid_c_star_mask):
                    valid_instance_indices = np.where(valid_c_star_mask)[0]
                    P_ifc_valid = P_ifc[valid_instance_indices, :, :]
                    c_star_indices_valid = c_star_indices[valid_instance_indices]

                    if P_ifc_valid.size > 0:  # Ensure we have data to index
                        p_support_c_star_per_feature_for_valid_instances = P_ifc_valid[
                            np.arange(P_ifc_valid.shape[0])[:, None],
                            np.arange(n_features)[None, :],
                            c_star_indices_valid[:, None]
                        ]
                        p_support_c_star_per_feature[valid_instance_indices,
                        :] = p_support_c_star_per_feature_for_valid_instances

                        mean_p_support_c_star_agg[valid_instance_indices] = np.nan_to_num(
                            np.nanmean(p_support_c_star_per_feature_for_valid_instances, axis=1), nan=0.0)
                        std_dev_support_c_star_agg[valid_instance_indices] = np.nan_to_num(
                            np.nanstd(p_support_c_star_per_feature_for_valid_instances, axis=1), nan=0.5)
                        q1_support_c_star_agg[valid_instance_indices] = np.nan_to_num(
                            np.nanpercentile(p_support_c_star_per_feature_for_valid_instances, 25, axis=1), nan=0.0)

                        p_clipped_valid = np.clip(p_support_c_star_per_feature_for_valid_instances, self._epsilon,
                                                  1.0 - self._epsilon)
                        # Check if p_clipped_valid is all NaNs for some rows before logit
                        if p_clipped_valid.size > 0 and not np.all(np.isnan(p_clipped_valid)):
                            logit_p_support_valid = np.log(p_clipped_valid) - np.log(1.0 - p_clipped_valid)
                            if logit_p_support_valid.size > 0 and not np.all(
                                    np.isnan(logit_p_support_valid)):  # ensure some non-nan values for percentile
                                q1_logit_support_c_star_agg_adapted[valid_instance_indices] = np.nan_to_num(
                                    np.nanpercentile(logit_p_support_valid, 25, axis=1), nan=default_q1_logit)

            normalized_entropy_if = None
            if compute_per_feature_entropy and P_ifc is not None and n_features > 0:
                if n_classes > 1:
                    log2_P_ifc = np.log2(P_ifc + self._epsilon)
                    entropy_if = -np.sum(P_ifc * log2_P_ifc, axis=2)
                    max_ent = np.log2(n_classes)
                    normalized_entropy_if = entropy_if / max_ent if max_ent > 0 else np.zeros_like(entropy_if)
                else:
                    normalized_entropy_if = np.zeros((n_instances, n_features))

            # --- Optional Metrics Calculations ---
            if "ood_score_gap" in final_active_optional_metric_names:
                ood_score_gap_val = np.full(n_instances, 0.0)
                if n_classes >= 2:
                    sorted_logliks = np.sort(logliks_for_stats, axis=1)
                    valid_gaps = ~np.isnan(sorted_logliks[:, -1]) & ~np.isnan(sorted_logliks[:, -2])
                    gap = np.full(n_instances, np.nan)
                    if np.any(valid_gaps): gap[valid_gaps] = sorted_logliks[valid_gaps, -1] - sorted_logliks[
                        valid_gaps, -2]
                    ood_score_gap_val = np.nan_to_num(-gap, nan=0.0)
                results_data["ood_score_gap"] = ood_score_gap_val

            if "ood_score_entropy" in final_active_optional_metric_names:
                ood_score_entropy_val = np.full(n_instances, 0.0)
                if n_classes > 0:
                    mask_meaningful_ll_rows = np.any(total_log_likelihoods > (self._log_zero_penalty + 1e-9), axis=1)
                    probabilities = np.full_like(total_log_likelihoods, 1.0 / n_classes if n_classes > 0 else 0.0)
                    if np.any(mask_meaningful_ll_rows):
                        meaningful_logliks = total_log_likelihoods[mask_meaningful_ll_rows, :]
                        probabilities[mask_meaningful_ll_rows, :] = softmax(meaningful_logliks, axis=1)
                    log_probabilities = np.log(probabilities + self._epsilon)
                    entropy_values = -np.sum(probabilities * log_probabilities, axis=1)
                    default_entropy_val = np.log(n_classes) if n_classes > 1 else 0.0
                    ood_score_entropy_val = np.nan_to_num(entropy_values, nan=default_entropy_val)
                results_data["ood_score_entropy"] = ood_score_entropy_val

            if "ood_score_mahalanobis" in final_active_optional_metric_names:
                ood_score_mahalanobis_val = np.full(n_instances, 0.0)
                if (self.likelihood_mean_vector_ is not None and self.likelihood_inv_covariance_ is not None and
                        n_classes > 0 and self.likelihood_mean_vector_.shape[0] == n_classes and
                        total_log_likelihoods.shape[1] == n_classes):
                    likelihood_vectors_clean = np.nan_to_num(
                        total_log_likelihoods, nan=self._log_zero_penalty,
                        posinf=self._log_zero_penalty, neginf=self._log_zero_penalty)
                    delta = likelihood_vectors_clean - self.likelihood_mean_vector_
                    term1 = delta @ self.likelihood_inv_covariance_
                    m_dist_sq = np.sum(term1 * delta, axis=1)
                    ood_score_mahalanobis_val = np.maximum(m_dist_sq, 0)
                results_data["ood_score_mahalanobis"] = np.nan_to_num(ood_score_mahalanobis_val, nan=0.0)

            if "ood_metric1_avg_support_c_star" in final_active_optional_metric_names:
                results_data["ood_metric1_avg_support_c_star"] = 1.0 - mean_p_support_c_star_agg

            if "ood_metric2_feature_agreement" in final_active_optional_metric_names:
                ood_metric2_val = np.full(n_instances, 1.0)
                if P_ifc is not None and n_features > 0 and n_classes > 0:
                    winner_if = np.argmax(P_ifc, axis=2)
                    confidence_if = np.max(P_ifc, axis=2)
                    combined_strength = np.zeros(n_instances)
                    for i in range(n_instances):
                        mode_res = mode(winner_if[i, :], keepdims=False)
                        c_agree_i, count_agree_i = -1, 0
                        if mode_res.count is not None and mode_res.count.size > 0 and mode_res.count[0] > 0:
                            c_agree_i = mode_res.mode.item(0) if isinstance(mode_res.mode,
                                                                            np.ndarray) else mode_res.mode
                            count_agree_i = mode_res.count.item(0) if isinstance(mode_res.count,
                                                                                 np.ndarray) else mode_res.count
                        if count_agree_i > 0:
                            agreement_ratio_i = count_agree_i / n_features
                            mask_agreed_features = (winner_if[i, :] == c_agree_i)
                            avg_confidence_for_agreed_class_i = np.mean(
                                confidence_if[i, mask_agreed_features]) if np.sum(mask_agreed_features) > 0 else 0.0
                            combined_strength[i] = agreement_ratio_i * avg_confidence_for_agreed_class_i
                    ood_metric2_val = 1.0 - combined_strength
                results_data["ood_metric2_feature_agreement"] = ood_metric2_val

            if "ood_metric3_avg_feature_entropy" in final_active_optional_metric_names:
                ood_metric3_val = np.full(n_instances, 1.0 if n_classes > 1 else 0.0)
                if normalized_entropy_if is not None and n_features > 0:
                    if n_classes > 1:
                        ood_metric3_val = np.nan_to_num(np.nanmean(normalized_entropy_if, axis=1), nan=1.0)
                    else:
                        ood_metric3_val = np.full(n_instances, 0.0)
                results_data["ood_metric3_avg_feature_entropy"] = ood_metric3_val

            if "ood_metric4_variance_support_c_star" in final_active_optional_metric_names:
                weight_std_dev_m4 = 0.5
                max_possible_std_dev_m4 = 0.5
                normalized_std_dev_m4 = np.clip(
                    std_dev_support_c_star_agg / max_possible_std_dev_m4 if max_possible_std_dev_m4 > 0 else 0, 0, 1.0)
                raw_score_m4 = (1.0 - mean_p_support_c_star_agg) + weight_std_dev_m4 * normalized_std_dev_m4
                results_data["ood_metric4_variance_support_c_star"] = np.clip(raw_score_m4 / (1.0 + weight_std_dev_m4),
                                                                              0, 1.0)

            if "ood_metric5_robust_dissent_c_star" in final_active_optional_metric_names:
                results_data["ood_metric5_robust_dissent_c_star"] = 1.0 - q1_support_c_star_agg

            if "ood_metric5_adapted_robust_dissent_c_star" in final_active_optional_metric_names:
                results_data["ood_metric5_adapted_robust_dissent_c_star"] = 1.0 - sigmoid(
                    q1_logit_support_c_star_agg_adapted)

            if "ood_metric6_q1_margin_c_star" in final_active_optional_metric_names:
                q1_margins_val = np.full(n_instances, np.abs(self._log_zero_penalty / 10))
                if c_star_indices is not None and n_features > 0 and n_classes > 1:
                    feature_margins = np.full((n_instances, n_features), np.nan, dtype=np.float64)
                    valid_c_star_mask = (c_star_indices != -1)
                    for i_idx in np.where(valid_c_star_mask)[0]:
                        c_star_for_instance = c_star_indices[i_idx]
                        for f_idx in range(n_features):
                            ll_vector_feature = feature_log_likelihoods_per_instance[i_idx, f_idx, :]
                            ll_c_star_feature = ll_vector_feature[c_star_for_instance]
                            competitor_ll_indices = np.arange(n_classes) != c_star_for_instance
                            competitor_ll_values = ll_vector_feature[competitor_ll_indices]
                            if competitor_ll_values.size > 0:
                                ll_next_best_feature = np.max(competitor_ll_values)
                                feature_margins[i_idx, f_idx] = ll_c_star_feature - ll_next_best_feature
                            else:
                                feature_margins[i_idx, f_idx] = np.abs(self._log_zero_penalty / 10)
                    if np.any(valid_c_star_mask):
                        q1_margins_val[valid_c_star_mask] = np.nan_to_num(
                            np.nanpercentile(feature_margins[valid_c_star_mask, :], 25, axis=1),
                            nan=np.abs(self._log_zero_penalty / 10))
                results_data["ood_metric6_q1_margin_c_star"] = sigmoid(-0.1 * q1_margins_val)

            if "ood_metric7_entropy_weighted_dissent_c_star" in final_active_optional_metric_names:
                q1_weighted_p_support = np.full(n_instances, 0.0)
                if (p_support_c_star_per_feature is not None and
                        normalized_entropy_if is not None and
                        n_features > 0 and n_classes > 0):
                    decisiveness_if = 1.0 - normalized_entropy_if
                    mask_valid_p_support_rows = ~np.all(np.isnan(p_support_c_star_per_feature), axis=1)
                    if np.any(mask_valid_p_support_rows):
                        weighted_p_support_valid_rows = p_support_c_star_per_feature[mask_valid_p_support_rows,
                                                        :] * decisiveness_if[mask_valid_p_support_rows, :]
                        q1_values_temp = np.nanpercentile(weighted_p_support_valid_rows, 25, axis=1)
                        q1_weighted_p_support[mask_valid_p_support_rows] = np.nan_to_num(q1_values_temp, nan=0.0)
                elif n_features > 0 and n_classes == 1:
                    q1_weighted_p_support = np.full(n_instances, 1.0)
                results_data["ood_metric7_entropy_weighted_dissent_c_star"] = 1.0 - q1_weighted_p_support

            if "ood_metric8_prop_clear_dissent_c_star" in final_active_optional_metric_names:
                prop_dissenting = np.full(n_instances, 1.0)
                if c_star_indices is not None and n_features > 0 and n_classes > 1:
                    num_clearly_dissenting = np.zeros(n_instances, dtype=int)
                    valid_c_star_mask = (c_star_indices != -1)
                    for i_idx in np.where(valid_c_star_mask)[0]:
                        c_star_for_instance = c_star_indices[i_idx]
                        for f_idx in range(n_features):
                            feature_ll_all_classes = feature_log_likelihoods_per_instance[i_idx, f_idx, :]
                            ll_c_star_feature = feature_ll_all_classes[c_star_for_instance]
                            feature_pref_class_idx = np.argmax(feature_ll_all_classes)
                            if feature_pref_class_idx != c_star_for_instance:
                                ll_feature_pref = feature_ll_all_classes[feature_pref_class_idx]
                                if ll_feature_pref > (ll_c_star_feature + self.dissent_margin_threshold):
                                    num_clearly_dissenting[i_idx] += 1
                    if np.any(valid_c_star_mask):
                        prop_dissenting[valid_c_star_mask] = num_clearly_dissenting[valid_c_star_mask] / n_features
                elif n_features > 0 and n_classes <= 1:
                    prop_dissenting = np.full(n_instances, 0.0)
                elif n_features == 0:
                    prop_dissenting = np.full(n_instances, 0.0)
                results_data["ood_metric8_prop_clear_dissent_c_star"] = prop_dissenting

        # --- Final DataFrame Assembly ---
        final_data_for_df = {}
        for col_name in all_output_cols:
            if col_name in results_data:
                final_data_for_df[col_name] = results_data[col_name]
            else:
                default_val = self._log_zero_penalty if "loglik" in col_name else \
                    (1.0 if "ood" in col_name.lower() and (
                                "metric" in col_name.lower() or "score" in col_name.lower()) else 0.0)
                warnings.warn(
                    f"Column '{col_name}' was expected in output but not found in results_data. Defaulting value to {default_val}.",
                    UserWarning)
                final_data_for_df[col_name] = np.full(n_instances, default_val, dtype=np.float64)

        results_df_ordered = pl.DataFrame(final_data_for_df).select(all_output_cols)
        return results_df_ordered