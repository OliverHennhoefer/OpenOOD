# -*- coding: utf-8 -*-
import torch
import polars  # Using 'polars' as per original, user can alias as 'pl' if preferred
import warnings
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Tuple, Union, Optional, Literal
import traceback  # Kept for the final try-except during DataFrame creation


class NetworkScanner:
    """
    Analyzes trained PyTorch models by extracting layer inputs/outputs and model outputs.

    Captures EITHER the primary input tensor fed into ('input' mode) OR the primary
    output tensor produced by ('output' mode) the forward pass of specified layers.

    Use 'input' mode on an activation layer (e.g., nn.ReLU) to capture data *before* activation.
    Use 'output' mode on a Linear/Conv layer to capture data *after* its computation
    but *before* any subsequent activation (if structured sequentially).

    Captured tensors with spatial dimensions (ndim >= 3) can optionally be summarized
    via Global Average Pooling (GAP).

    Outputs a tuple containing:
        1. A Polars DataFrame with captured data exploded into columns.
        2. A list containing the raw model outputs for each sample processed.
    """

    def __init__(
            self,
            model: nn.Module,
            target_layer_names: List[str],
            capture_mode: Literal['input', 'output'] = 'output'
    ):
        self.model = model
        self.model.eval()  # Set model to evaluation mode
        self.capture_mode = capture_mode
        # Suffix for captured data columns, determined by capture_mode
        self.data_suffix = "_layer_input" if capture_mode == 'input' else "_layer_output"

        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:  # Model has no parameters
            warnings.warn("Model has no parameters. Assuming CPU.", UserWarning, stacklevel=2)
            self.device = torch.device("cpu")
        self.model.to(self.device)  # Ensure model is on the correct device

        self.target_layer_names = target_layer_names
        self.target_modules: Dict[str, nn.Module] = {}  # Stores references to target nn.Module objects

        if self.target_layer_names:
            self._identify_target_modules()
            if not self.target_modules:
                warnings.warn(
                    "None of the specified `target_layer_names` were found in the model. "
                    "Only model outputs will be collected.", UserWarning, stacklevel=2
                )
        else:  # Empty target_layer_names
            warnings.warn(
                "`target_layer_names` is empty. Only model outputs will be collected.",
                UserWarning, stacklevel=2
            )

        # These will be populated during predict() and reset for each call
        self.extracted_data_list: List[Dict[str, Any]] = []
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self.model_outputs_list: List[Any] = []

    def _identify_target_modules(self):
        """Identifies and stores references to the target nn.Module objects."""
        target_layer_names_set = set(self.target_layer_names)
        found_names = set()
        for name, module in self.model.named_modules():
            if name in target_layer_names_set:
                self.target_modules[name] = module
                found_names.add(name)
                if len(found_names) == len(target_layer_names_set):  # Early exit if all found
                    break

        missing_names = target_layer_names_set - found_names
        if missing_names:
            warnings.warn(
                f"Target layers specified but not found in model: {list(missing_names)}. They will be ignored.",
                UserWarning, stacklevel=2
            )

    def _clear_hooks(self):
        """Removes all registered PyTorch hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def predict(
            self,
            dataloader: DataLoader,
            include_label: bool = True,
            include_image_id: bool = True,
            image_key: str = "data",  # Key for image tensor in batch dict
            label_key: str = "label",  # Key for label in batch dict
            id_key: str = "index",  # Key for image ID in batch dict
            apply_gap_to_high_dim: bool = True,  # Apply Global Average Pooling to >2D tensors
    ) -> Tuple[Union[polars.DataFrame, List[Dict[str, Any]]], List[Any]]:
        """
        Processes data through the model, captures layer data, and model outputs.
        """
        self.extracted_data_list = []  # Reset for current prediction run
        self.model_outputs_list = []  # Reset for current prediction run

        self.model.eval()  # Ensure model is in eval mode

        hooks_needed = bool(self.target_modules)
        final_id_col_name = id_key  # Use user-specified id_key as the column name in DataFrame

        iterator = tqdm(dataloader, desc="Scanning Batches", leave=False)
        for batch_idx, batch in enumerate(iterator):
            images: Optional[torch.Tensor] = None
            labels_batch: Optional[Union[torch.Tensor, List[Any]]] = None
            image_ids_batch: Optional[Union[torch.Tensor, List[Any]]] = None
            batch_size = 0

            # --- 1. Batch Data Extraction ---
            if isinstance(batch, dict):  # Batch is a dictionary
                images = batch.get(image_key)
                if include_label: labels_batch = batch.get(label_key)
                if include_image_id: image_ids_batch = batch.get(id_key)
            elif isinstance(batch, (list, tuple)) and len(batch) > 0:  # Batch is a list/tuple
                images = batch[0]
                if include_label and len(batch) > 1: labels_batch = batch[1]
                if include_image_id and len(batch) > 2: image_ids_batch = batch[2]
            else:  # Unsupported batch type
                continue  # Skip to next batch

            if images is None:  # Images not found in batch
                continue

            try:  # Ensure images is a tensor and move to device
                images = images.to(self.device)
                batch_size = images.shape[0]
                if batch_size == 0: continue  # Skip empty batch
            except (AttributeError, TypeError, RuntimeError) as e:
                warnings.warn(f"Failed to process images for batch {batch_idx}: {e}. Skipping batch.", RuntimeWarning,
                              stacklevel=2)
                continue

            # Move labels/IDs to CPU and convert to list later if they are tensors
            if labels_batch is not None and isinstance(labels_batch, torch.Tensor):
                labels_batch = labels_batch.cpu()
            if image_ids_batch is not None and isinstance(image_ids_batch, torch.Tensor):
                image_ids_batch = image_ids_batch.cpu()

            # --- 2. Hook Setup ---
            batch_storage = defaultdict(list)  # Temporary storage for hooked tensors for this batch
            self._hook_handles.clear()  # Clear any stale handles from previous iterations (shouldn't be any if logic is correct)

            if hooks_needed:
                def create_forward_hook(layer_name: str):
                    # Closure to capture layer_name
                    def forward_hook(_module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: Any):
                        tensor_to_capture = None
                        if self.capture_mode == 'input':
                            if inputs and isinstance(inputs[0], torch.Tensor):
                                tensor_to_capture = inputs[0]
                        elif self.capture_mode == 'output':  # capture_mode == 'output'
                            if isinstance(output, torch.Tensor):
                                tensor_to_capture = output
                            elif isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
                                tensor_to_capture = output[0]  # Default to first tensor if output is a sequence
                                # Optional: Warn if layer output is a multi-tensor sequence
                                # if len(output) > 1:
                                #     warnings.warn(f"Layer '{layer_name}' output is a sequence. Capturing only the first tensor.", RuntimeWarning, stacklevel=2)
                        if tensor_to_capture is not None:
                            batch_storage[f"{layer_name}{self.data_suffix}"].append(tensor_to_capture.detach().cpu())

                    return forward_hook

                for name, module_to_hook in self.target_modules.items():
                    handle = module_to_hook.register_forward_hook(create_forward_hook(name))
                    self._hook_handles.append(handle)

            # --- 3. Forward Pass & Hook Removal ---
            model_outputs_batch: Optional[torch.Tensor] = None
            try:
                with torch.no_grad():  # Ensure no gradients are computed during inference
                    model_outputs_batch = self.model(images)
            except Exception as e:
                warnings.warn(f"Forward pass failed for batch {batch_idx}: {e}. Skipping batch.", RuntimeWarning,
                              stacklevel=2)
                # self._clear_hooks() already in finally
                continue  # Skip to next batch
            finally:
                self._clear_hooks()  # CRITICAL: Always remove hooks after forward pass

            # --- 4. Consolidate Hooked Data ---
            # consolidated_batch_data maps a base column name to the batch tensor for that layer
            consolidated_batch_data: Dict[str, torch.Tensor] = {}
            if hooks_needed:
                for key, tensor_list in batch_storage.items():  # key is e.g. "layerName_layer_output"
                    if tensor_list:  # Should contain one tensor if layer called once, or multiple if called multiple times
                        # If a layer is called multiple times in one forward pass (e.g. shared weights in a loop),
                        # its hook fires multiple times. We take the first one by default.
                        # layer_base_name_for_warning = key.replace(self.data_suffix, '')
                        # if len(tensor_list) > 1:
                        #     warnings.warn(f"Layer '{layer_base_name_for_warning}' ({self.capture_mode}) hook fired {len(tensor_list)} times for batch {batch_idx}. Using first capture.", RuntimeWarning, stacklevel=2)
                        consolidated_batch_data[key] = tensor_list[0]

            # --- 5. Store Model Outputs ---
            if model_outputs_batch is not None:
                try:
                    self.model_outputs_list.extend(model_outputs_batch.detach().cpu().tolist())
                except Exception:  # Catch broad error if .tolist() or other ops fail
                    self.model_outputs_list.extend([None] * batch_size)  # Append Nones if conversion fails

            # --- 6. Prepare Data for DataFrame (Per-Sample Processing) ---
            # Convert labels and IDs to Python lists if they aren't already
            labels_list: Optional[List[Any]] = None
            if include_label and labels_batch is not None:
                labels_list = labels_batch.tolist() if isinstance(labels_batch, torch.Tensor) else list(labels_batch)

            image_ids_list: Optional[List[Any]] = None
            if include_image_id and image_ids_batch is not None:
                image_ids_list = image_ids_batch.tolist() if isinstance(image_ids_batch, torch.Tensor) else list(
                    image_ids_batch)

            # Pre-calculate final column names for layer data for this batch (avoids re-computation per sample)
            current_batch_layer_col_names: Dict[
                str, Dict[str, str]] = {}  # Maps internal key to {'data': data_col_name, 'error': error_col_name}
            if hooks_needed:
                for internal_key, batch_tensor_for_naming in consolidated_batch_data.items():
                    base_name_for_col = internal_key  # e.g., "layer1_layer_output"
                    layer_name_part = internal_key.replace(self.data_suffix, "")  # e.g., "layer1"
                    error_col_name = f"{layer_name_part}_error"

                    # Determine the actual data column name based on tensor dimension and GAP flag
                    if batch_tensor_for_naming.ndim >= 3 and apply_gap_to_high_dim:
                        data_col_name = f"{base_name_for_col}_avg"  # e.g. "layer1_layer_output_avg"
                    else:
                        data_col_name = base_name_for_col  # e.g. "layer1_layer_output"
                    current_batch_layer_col_names[internal_key] = {'data': data_col_name, 'error': error_col_name}

            for i in range(batch_size):  # Iterate over samples in the batch
                sample_data: Dict[str, Any] = {}  # Data for this single sample
                if include_label:
                    sample_data["label"] = labels_list[i] if labels_list and i < len(labels_list) else None
                if include_image_id:
                    sample_data[final_id_col_name] = image_ids_list[i] if image_ids_list and i < len(
                        image_ids_list) else None

                if hooks_needed:
                    for internal_key, batch_tensor in consolidated_batch_data.items():
                        col_names_for_layer = current_batch_layer_col_names[internal_key]
                        data_col_to_populate = col_names_for_layer['data']
                        error_col_to_populate = col_names_for_layer['error']
                        try:
                            sample_item = batch_tensor[i]  # Tensor for the i-th sample from this layer's batch output

                            if sample_item.ndim >= 3:  # High-dimensional (e.g., image feature map C,H,W)
                                if apply_gap_to_high_dim:
                                    spatial_dims = tuple(range(1, sample_item.ndim))  # Dims other than channel
                                    sample_data[data_col_to_populate] = sample_item.mean(dim=spatial_dims).tolist()
                                else:  # Flatten high-dim tensor
                                    sample_data[data_col_to_populate] = sample_item.flatten().tolist()
                            elif 0 < sample_item.ndim <= 2:  # 1D or 2D tensor (e.g. features, or already GAPped)
                                sample_data[data_col_to_populate] = sample_item.flatten().tolist()
                            elif sample_item.ndim == 0:  # Scalar tensor
                                sample_data[data_col_to_populate] = sample_item.item()
                        except Exception as e:
                            sample_data[error_col_to_populate] = str(e)
                            # Ensure the data column is not populated or is None if an error occurred
                            if data_col_to_populate in sample_data: del sample_data[data_col_to_populate]
                self.extracted_data_list.append(sample_data)

            # Batch Cleanup: Python's GC handles most objects. `torch.cuda.empty_cache()` is slow.
            # If facing extreme memory pressure, it could be enabled here, but it has a performance cost.
            # torch.cuda.empty_cache() if self.device.type == 'cuda' else None

        # --- 7. Final DataFrame Creation & Post-processing ---
        if not self.extracted_data_list and not self.model_outputs_list:  # No data collected at all
            warnings.warn("No layer data extracted AND no model outputs collected. Returning empty.", UserWarning,
                          stacklevel=2)
            return polars.DataFrame(), []

        results_df: Optional[polars.DataFrame] = None
        try:
            results_df = polars.DataFrame(self.extracted_data_list, strict=False)

            potential_list_col_suffixes = [self.data_suffix, f"{self.data_suffix}_avg"]
            cols_to_check_for_list_type = [
                c for c in results_df.columns
                if any(c.endswith(suffix) for suffix in potential_list_col_suffixes)
            ]
            list_cols_to_explode = [
                c for c in cols_to_check_for_list_type
                if results_df[c].dtype == polars.List(polars.Unknown) or isinstance(results_df[c].dtype, polars.List)
            ]

            if list_cols_to_explode:
                max_len_exprs = [
                    polars.col(c).list.len().max().fill_null(0).alias(c) for c in list_cols_to_explode
                ]
                # .row(0, named=True) gets the single row of max lengths as a dict
                max_lengths_dict = results_df.select(max_len_exprs).row(0, named=True) if max_len_exprs else {}

                all_new_col_exprs = []
                for col_name in list_cols_to_explode:
                    max_len = max_lengths_dict.get(col_name, 0)
                    if max_len > 0:  # Only create expressions if there's something to get
                        for i in range(max_len):
                            new_col_name = f"{col_name}_{i}"  # Exploded column name
                            expr = polars.col(col_name).list.get(i).alias(new_col_name)
                            all_new_col_exprs.append(expr)

                if all_new_col_exprs:
                    existing_cols_to_keep = [c for c in results_df.columns if c not in list_cols_to_explode]
                    results_df = results_df.select(existing_cols_to_keep + all_new_col_exprs)

            # Reorder columns: ID, label, then sorted remaining columns
            if results_df is not None:
                current_df_cols = results_df.columns
                ordered_cols = []
                # Add ID and label first if they exist
                if final_id_col_name in current_df_cols: ordered_cols.append(final_id_col_name)
                if "label" in current_df_cols: ordered_cols.append("label")

                remaining_cols = sorted([c for c in current_df_cols if c not in ordered_cols])
                ordered_cols.extend(remaining_cols)
                results_df = results_df.select(ordered_cols)

            # Ensure a DataFrame is returned, even if it's empty
            return results_df if results_df is not None else polars.DataFrame(), self.model_outputs_list

        except Exception as e:  # Fallback if DataFrame processing fails
            error_msg = f"Polars DataFrame processing failed: {e}\n{traceback.format_exc()}"
            warnings.warn(f"{error_msg}\nReturning raw list of dictionaries for layer data.", UserWarning, stacklevel=2)
            return self.extracted_data_list, self.model_outputs_list