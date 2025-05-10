import numpy as np
import polars as pl
import polars.selectors as cs

from tqdm import tqdm
from scipy.special import softmax
from sklearn.neighbors import KernelDensity
from typing import Literal, List, Dict, Optional, Any
import warnings

from joblib import Parallel, delayed



class DimensionWiseKdeOODMulti:
    # ... (rest of the __init__ and _calculate_bandwidth methods are unchanged) ...
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

    @staticmethod
    def _fit_kde_joblib_worker(col_name: str, label: Any,
                               data_reshaped: np.ndarray, bandwidth: float, kernel_str: str
                               ) -> tuple[str, Any, Optional[KernelDensity], Optional[str]]:
        # DEBUG: Print worker start
        # print(f"[Worker Fit START] col: {col_name}, label: {label}, bw: {bandwidth:.4f}, data_shape: {data_reshaped.shape}")
        try:
            kde = KernelDensity(kernel=kernel_str, bandwidth=bandwidth)
            kde.fit(data_reshaped)
            # DEBUG: Print worker success
            # print(f"[Worker Fit SUCCESS] col: {col_name}, label: {label}")
            return col_name, label, kde, None
        except Exception as e:
            # DEBUG: Print worker error
            # print(f"[Worker Fit ERROR] col: {col_name}, label: {label}, error: {e}")
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
            n_jobs: int = 10,
            joblib_verbose: int = 0  # Added for controlling joblib verbosity
    ):
        # ... (initial checks and column selection unchanged) ...
        if not isinstance(data, pl.DataFrame):
            raise TypeError("Input data must be a Polars DataFrame.")
        if label_col not in data.columns:
            raise ValueError(f"Label column '{label_col}' not found in DataFrame.")

        self.bandwidth_method_ = bandwidth_method

        try:
            potential_cols_selector = cs.contains(like)
            temp_selected_cols = data.select(potential_cols_selector).columns
            if label_col in temp_selected_cols:
                potential_cols_selector = potential_cols_selector.exclude(label_col)

            potential_cols_df = data.select(potential_cols_selector)
            self.fitted_columns_ = potential_cols_df.select(cs.numeric()).columns

            non_numeric_excluded = set(potential_cols_df.columns) - set(self.fitted_columns_)
            if non_numeric_excluded:
                warnings.warn(
                    f"Non-numeric columns matching pattern '{like}' were excluded: {list(non_numeric_excluded)}",
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
        desc_fit_prepare = "Preparing KDE fit tasks"
        # This loop prepares tasks. If it hangs here, the issue is with Polars/data access.
        print("Starting KDE task preparation...")
        for col in tqdm(self.fitted_columns_, desc=desc_fit_prepare, unit="feature", leave=False):
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

                bw = self._calculate_bandwidth(class_col_data_np, self.bandwidth_method_)
                if bw < 1e-7:
                    warnings.warn(
                        f"Calculated bandwidth for col='{col}', label='{label}' is too small ({bw:.2e}). Skipping KDE.",
                        UserWarning)
                    continue

                data_reshaped = class_col_data_np.reshape(-1, 1)
                kde_fitting_tasks_params.append(
                    (col, label, data_reshaped, bw, kernel)
                )
        print(f"Prepared {len(kde_fitting_tasks_params)} KDE fitting tasks.")

        if kde_fitting_tasks_params:
            desc_fit_parallel = "Fitting KDEs (parallel)"
            print(f"Starting parallel KDE fitting with n_jobs={n_jobs}, joblib_verbose={joblib_verbose}...")
            # ADDED joblib_verbose to Parallel call
            with Parallel(n_jobs=n_jobs, verbose=joblib_verbose) as parallel:
                results = []
                # This loop consumes results from joblib. TQDM tracks this consumption.
                # If the bar doesn't move, joblib's parallel(...) call isn't yielding results.
                for result_tuple in tqdm(
                        parallel(
                            delayed(DimensionWiseKdeOODMulti._fit_kde_joblib_worker)(*params)
                            for params in kde_fitting_tasks_params
                        ),
                        total=len(kde_fitting_tasks_params),
                        desc=desc_fit_parallel,
                        unit="KDE",
                        leave=False  # Progress bar will be removed upon completion of this loop
                ):
                    results.append(result_tuple)

            print("Finished parallel KDE fitting. Processing results...")
            for res_col, res_label, res_kde, res_error_msg in results:
                if res_error_msg:
                    warnings.warn(
                        f"Error fitting KDE for col='{res_col}', label='{res_label}'. Error: {res_error_msg}. Storing None.",
                        UserWarning)
                    # self.kdes_[res_col][res_label] = None # Already default
                else:
                    self.kdes_[res_col][res_label] = res_kde

        fitted_kde_count = sum(kde is not None for col_kdes in self.kdes_.values() for kde in col_kdes.values())
        if fitted_kde_count == 0:
            warnings.warn(
                "No KDEs were successfully fitted. Check data and parameters.", UserWarning)
        print(f"Total KDEs successfully fitted: {fitted_kde_count}")

        # ... (Mahalanobis calculation part - remember to pass n_jobs and joblib_verbose if it calls transform)
        self.likelihood_mean_vector_ = None
        self.likelihood_inv_covariance_ = None
        if calculate_mahalanobis_params:
            if fitted_kde_count == 0:
                warnings.warn("Cannot calculate Mahalanobis params as no KDEs were fitted.", UserWarning)
                return self

            print("Calculating Mahalanobis parameters (requires calling transform)...")
            # Pass n_jobs and joblib_verbose to the internal transform call
            train_likelihood_df = self.transform(
                data,
                calculate_mahalanobis_score=False,
                n_jobs=n_jobs,
                joblib_verbose=joblib_verbose
            )
            # ... (rest of Mahalanobis logic)
            likelihood_cols_for_maha = [f"total_loglik_class_{lbl}" for lbl in self.normal_labels_]
            actual_cols = train_likelihood_df.columns
            missing_maha_cols = [lc for lc in likelihood_cols_for_maha if lc not in actual_cols]

            if missing_maha_cols:
                warnings.warn(
                    f"Missing likelihood columns for Mahalanobis: {missing_maha_cols}. Cannot calculate params.",
                    UserWarning)
                return self

            if not all(col in actual_cols for col in likelihood_cols_for_maha):
                warnings.warn(
                    f"Not all expected likelihood columns ({likelihood_cols_for_maha}) "
                    f"found in transform output ({actual_cols}) for Mahalanobis. Skipping calculation.",
                    UserWarning
                )
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
                covariance = np.cov(train_likelihood_vectors, rowvar=False)
                if covariance.ndim == 0:
                    covariance = np.array([[covariance]])
                ridge_epsilon = 1e-6
                covariance_reg = covariance + np.eye(covariance.shape[0]) * ridge_epsilon
                self.likelihood_inv_covariance_ = np.linalg.pinv(covariance_reg)
                print("Successfully calculated Mahalanobis parameters.")
            except Exception as e:
                warnings.warn(f"Error calculating Mahalanobis parameters: {e}. Parameters not stored.", UserWarning)
                self.likelihood_mean_vector_ = None
                self.likelihood_inv_covariance_ = None

        return self

    @staticmethod
    def _score_column_joblib_worker(
            col_name_debug: str,  # Added for debug prints
            feature_values_for_column_slice: np.ndarray,
            kdes_for_this_col: Dict[Any, Optional[KernelDensity]],
            normal_labels_order: List[Any],
            log_zero_penalty: float,
            n_total_instances: int
    ) -> np.ndarray:
        # DEBUG: Print worker start
        # print(f"[Worker Score START] col: {col_name_debug}, data_shape: {feature_values_for_column_slice.shape}")
        current_feature_values_reshaped = feature_values_for_column_slice.reshape(-1, 1)
        mask_finite_input = np.isfinite(feature_values_for_column_slice)

        log_likelihoods_contribution = np.zeros(
            (n_total_instances, len(normal_labels_order)), dtype=np.float64
        )

        for class_idx, label in enumerate(normal_labels_order):
            kde = kdes_for_this_col.get(label)

            if kde is not None:
                scores_for_this_feature_class = np.full(n_total_instances, log_zero_penalty, dtype=float)
                finite_values_to_score = current_feature_values_reshaped[mask_finite_input]

                if finite_values_to_score.size > 0:
                    try:
                        # DEBUG: Print before potentially long operation
                        # print(f"[Worker Score] Scoring col: {col_name_debug}, label: {label}, finite_values_shape: {finite_values_to_score.shape}")
                        scored_finite_values = kde.score_samples(finite_values_to_score)
                        scored_finite_values_clean = np.nan_to_num(
                            scored_finite_values, nan=log_zero_penalty,
                            posinf=log_zero_penalty, neginf=log_zero_penalty
                        )
                        scores_for_this_feature_class[mask_finite_input] = scored_finite_values_clean
                    except Exception as e:
                        # DEBUG: Print worker error during scoring
                        # print(f"[Worker Score ERROR] col: {col_name_debug}, label: {label}, error: {e}")
                        # scores_for_this_feature_class remains log_zero_penalty
                        pass
                log_likelihoods_contribution[:, class_idx] = scores_for_this_feature_class
            else:
                log_likelihoods_contribution[:, class_idx] = log_zero_penalty

        # DEBUG: Print worker end
        # print(f"[Worker Score END] col: {col_name_debug}")
        return log_likelihoods_contribution

    def transform(
            self, inference_data: pl.DataFrame, calculate_mahalanobis_score: bool = True,
            n_jobs: int = 10,
            joblib_verbose: int = 0  # Added for controlling joblib verbosity
    ) -> pl.DataFrame:
        # ... (initial checks and data setup unchanged) ...
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

        if not self.fitted_columns_:
            inference_features_np = np.empty((n_instances, 0), dtype=np.float64)
        else:
            select_exprs = [pl.col(col).cast(pl.Float64, strict=False) for col in self.fitted_columns_]
            inference_features_np = inference_data.select(select_exprs).to_numpy()

        total_log_likelihoods = np.zeros((n_instances, n_classes), dtype=np.float64)

        if self.fitted_columns_ and n_classes > 0:
            scoring_tasks = []
            print("Preparing scoring tasks...")
            for col_idx, col_name in enumerate(self.fitted_columns_):
                task_args = (
                    col_name,  # Pass col_name for debugging prints in worker
                    inference_features_np[:, col_idx],
                    self.kdes_[col_name],
                    self.normal_labels_,
                    self._log_zero_penalty,
                    n_instances
                )
                scoring_tasks.append(task_args)
            print(f"Prepared {len(scoring_tasks)} scoring tasks.")

            if scoring_tasks:
                desc_score_parallel = "Scoring Instances (parallel by feature)"
                print(f"Starting parallel scoring with n_jobs={n_jobs}, joblib_verbose={joblib_verbose}...")
                # ADDED joblib_verbose to Parallel call
                with Parallel(n_jobs=n_jobs, verbose=joblib_verbose) as parallel:
                    per_column_likelihoods_list = []
                    for res_array in tqdm(
                            parallel(
                                delayed(DimensionWiseKdeOODMulti._score_column_joblib_worker)(*params)
                                for params in scoring_tasks
                            ),
                            total=len(scoring_tasks),
                            desc=desc_score_parallel,
                            unit="feature",
                            leave=False,
                            dynamic_ncols=True
                    ):
                        per_column_likelihoods_list.append(res_array)

                print("Finished parallel scoring. Aggregating results...")
                if per_column_likelihoods_list:
                    total_log_likelihoods = np.sum(np.array(per_column_likelihoods_list), axis=0)

        elif n_classes > 0:
            num_pseudo_features = len(self.fitted_columns_) if self.fitted_columns_ else 1
            total_log_likelihoods[:] = self._log_zero_penalty * num_pseudo_features

        print("Calculating final metrics...")
        # ... (rest of the metric calculations and DataFrame creation unchanged) ...
        results_data = {}
        if self.normal_labels_:
            for i, lbl_col_name in enumerate(likelihood_col_names):
                results_data[lbl_col_name] = total_log_likelihoods[:, i]

        logliks_for_stats = np.where(total_log_likelihoods <= (self._log_zero_penalty + 1e-9),
                                     np.nan, total_log_likelihoods)
        if n_classes == 0:
            logliks_for_stats = np.empty((n_instances, 0))

        with np.errstate(invalid='ignore', divide='ignore'):
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
                if logliks_for_stats.shape[1] >= 2:
                    sorted_logliks = np.sort(logliks_for_stats, axis=1)
                    gap = sorted_logliks[:, -1] - sorted_logliks[:, -2]
                    ood_score_gap = np.nan_to_num(-gap, nan=0.0)
            results_data["ood_score_gap"] = ood_score_gap

            ood_score_entropy = np.full(n_instances, 0.0)
            if n_classes > 0:
                mask_meaningful_ll_rows = np.any(logliks_for_stats > (self._log_zero_penalty + 1e-9), axis=1)
                probabilities = np.full_like(total_log_likelihoods, 1.0 / n_classes if n_classes > 0 else 0.0)

                if np.any(mask_meaningful_ll_rows):
                    meaningful_logliks = total_log_likelihoods[mask_meaningful_ll_rows, :]
                    if meaningful_logliks.size > 0:
                        probabilities[mask_meaningful_ll_rows, :] = softmax(meaningful_logliks, axis=1)

                log_probabilities = np.log(probabilities + 1e-30)
                entropy_values = -np.sum(probabilities * log_probabilities, axis=1)
                default_entropy_val = np.log(n_classes) if n_classes > 1 else 0.0
                ood_score_entropy = np.nan_to_num(entropy_values, nan=default_entropy_val)
            results_data["ood_score_entropy"] = ood_score_entropy

            ood_score_mahalanobis = np.full(n_instances, 0.0)
            if (
                    calculate_mahalanobis_score and
                    self.likelihood_mean_vector_ is not None and
                    self.likelihood_inv_covariance_ is not None and
                    n_classes > 0 and
                    total_log_likelihoods.shape[1] == n_classes and
                    self.likelihood_mean_vector_.shape[0] == n_classes
            ):
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
                    warnings.warn("Dimension mismatch for Mahalanobis calculation. Skipping.", UserWarning)

            results_data["ood_score_mahalanobis"] = np.nan_to_num(ood_score_mahalanobis, nan=0.0)

        results_df = pl.DataFrame(results_data)
        ordered_cols_present = [col for col in all_output_cols if col in results_df.columns]
        print("Transform method finished.")
        return results_df.select(ordered_cols_present)