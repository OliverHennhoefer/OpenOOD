# -*- coding: utf-8 -*-
import torch
import polars
import warnings
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Tuple, Union, Optional, Literal # Added Literal
import traceback

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
        # <<< NEW: Capture mode >>>
        capture_mode: Literal['input', 'output'] = 'output'
    ):
        """
        Args:
            model: Pre-trained PyTorch model (nn.Module).
            target_layer_names: List of layer names whose inputs or outputs
                                should be captured.
            capture_mode (Literal['input', 'output']):
                - 'input': Capture the input tensor(s) passed to the layer's forward().
                           Default. Use for capturing pre-activation data by targeting
                           the activation layer (e.g., nn.ReLU).
                - 'output': Capture the output tensor(s) from the layer's forward().
                            Use for capturing post-Linear/Conv data by targeting
                            the Linear/Conv layer itself.
        """
        self.model = model
        self.model.eval()
        self.capture_mode = capture_mode
        self.data_suffix = "_layer_input" if capture_mode == 'input' else "_layer_output" # <<< Adjust suffix based on mode

        # ... (rest of __init__ is the same as before) ...
        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:
            warnings.warn("Model has no parameters. Assuming CPU.", UserWarning)
            self.device = torch.device("cpu")
        self.model.to(self.device)

        self.target_layer_names = target_layer_names
        self.target_modules: Dict[str, nn.Module] = {}

        if self.target_layer_names:
            self._identify_target_modules()
            if not self.target_modules:
                 warnings.warn(
                    "None of the specified `target_layer_names` were found in the model. "
                    "Only model outputs will be collected.", UserWarning
                 )
        else:
             warnings.warn(
                "`target_layer_names` is empty. Only model outputs will be collected.",
                UserWarning
            )

        self.extracted_data_list: List[Dict[str, Any]] = []
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self.model_outputs_list: List[Any] = []


    def _identify_target_modules(self):
        # ... (same as before) ...
        found_names = set()
        for name, module in self.model.named_modules():
            if name in self.target_layer_names:
                self.target_modules[name] = module
                found_names.add(name)
        missing_names = set(self.target_layer_names) - found_names
        if missing_names:
            warnings.warn(
                f"Target layers specified but not found in model: {missing_names}. They will be ignored.", UserWarning
            )

    def _clear_hooks(self):
        # ... (same as before) ...
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def predict(
        # ... (arguments are the same, except maybe adjust docstring for data name) ...
        self,
        dataloader: DataLoader,
        include_label: bool = True,
        include_image_id: bool = True,
        image_key: str = "data",
        label_key: str = "label",
        id_key: str = "index",
        apply_gap_to_high_dim: bool = True,
    ) -> Tuple[Union[polars.DataFrame, List[Dict[str, Any]]], List[Any]]:
        # ... (initial setup is the same) ...
        self.extracted_data_list = []
        self.model_outputs_list = []
        self.model.eval()
        self.model.to(self.device)
        batch_num = 0
        hooks_needed = bool(self.target_modules)
        iterator = tqdm(dataloader, desc="Scanning Batches")
        final_id_col_name = id_key

        for batch in iterator:
             # ... (batch setup is the same) ...
            batch_num += 1
            images = None
            labels_batch = None
            image_ids_batch = None
            batch_size = 0
            if isinstance(batch, (list, tuple)): # Simplified batch setup
                if not batch: continue
                try: images = batch[0].to(self.device); batch_size = images.shape[0]
                except (IndexError, AttributeError, TypeError, RuntimeError): continue # Basic check
            elif isinstance(batch, dict):
                try: images = batch[image_key].to(self.device); batch_size = images.shape[0]
                except (KeyError, AttributeError, TypeError, RuntimeError): continue # Basic check
            else: continue
            if images is None or batch_size == 0: continue
            if include_label: labels_batch = batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else batch.get(label_key) if isinstance(batch, dict) else None
            if include_image_id: image_ids_batch = batch[2] if isinstance(batch, (list, tuple)) and len(batch) > 2 else batch.get(id_key) if isinstance(batch, dict) else None
            if labels_batch is not None and isinstance(labels_batch, torch.Tensor): labels_batch = labels_batch.cpu()
            if image_ids_batch is not None and isinstance(image_ids_batch, torch.Tensor): image_ids_batch = image_ids_batch.cpu()

            batch_storage = defaultdict(list)
            self._hook_handles = []

            # --- 3. Hook Registration ---
            if hooks_needed:
                def create_forward_hook(layer_name: str):
                    def forward_hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: Any):
                        tensor_to_capture = None
                        # <<< MODIFIED: Choose input or output based on mode >>>
                        if self.capture_mode == 'input':
                            if inputs and isinstance(inputs[0], torch.Tensor):
                                tensor_to_capture = inputs[0]
                            # else: Optional warning
                        elif self.capture_mode == 'output':
                             # Output might be a tensor or a tuple/list containing tensors
                             if isinstance(output, torch.Tensor):
                                 tensor_to_capture = output
                             elif isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
                                 # Take the first tensor if output is sequence (common case)
                                 tensor_to_capture = output[0]
                                 if len(output) > 1:
                                     warnings.warn(f"Layer '{layer_name}' output is a sequence with >1 tensor. Capturing only the first.", RuntimeWarning)
                             # else: Optional warning if output structure is unexpected

                        if tensor_to_capture is not None:
                            # Use the suffix determined in __init__
                            batch_storage[f"{layer_name}{self.data_suffix}"].append(tensor_to_capture.detach().cpu())

                    return forward_hook

                for name, module in self.target_modules.items():
                    handle = module.register_forward_hook(create_forward_hook(name))
                    self._hook_handles.append(handle)

            # --- 4. Forward Pass & Output Capture ---
            # ... (same as before) ...
            model_outputs_batch = None
            try:
                with torch.no_grad(): model_outputs_batch = self.model(images)
            except Exception as e:
                warnings.warn(f"Forward pass failed batch {batch_num}: {e}. Skipping batch.", RuntimeWarning)
                self._clear_hooks(); del images, labels_batch, image_ids_batch, batch_storage, model_outputs_batch
                torch.cuda.empty_cache() if self.device.type == 'cuda' else None; continue

            # --- 5. Hook Removal ---
            self._clear_hooks()

            # --- 6. Data Consolidation (Layer Inputs/Outputs) ---
            consolidated_batch_data: Dict[str, torch.Tensor] = {}
            if hooks_needed:
                 for key, tensor_list in batch_storage.items():
                    if tensor_list:
                        # Extract base name using the current suffix
                        layer_base_name = key.replace(self.data_suffix, '')
                        if len(tensor_list) > 1:
                            warnings.warn(f"Layer '{layer_base_name}' ({self.capture_mode}) hook fired {len(tensor_list)} times in batch {batch_num}. Using first capture.", RuntimeWarning)
                        consolidated_batch_data[key] = tensor_list[0]

            # --- 6b. Store Model Outputs ---
            # ... (same as before) ...
            if model_outputs_batch is not None:
                try: self.model_outputs_list.extend(model_outputs_batch.detach().cpu().tolist())
                except Exception as e:
                    warnings.warn(f"Could not store model outputs batch {batch_num}: {e}.", RuntimeWarning)
                    self.model_outputs_list.extend([None] * batch_size)


            # --- 7. Sample Data Extraction & Appending ---
            # ... (label/id processing is the same) ...
            labels_list = None # Simplified label/id prep
            if include_label and labels_batch is not None: labels_list = labels_batch.tolist() if isinstance(labels_batch, torch.Tensor) else list(labels_batch)
            image_ids_list = None
            if include_image_id and image_ids_batch is not None: image_ids_list = image_ids_batch.tolist() if isinstance(image_ids_batch, torch.Tensor) else list(image_ids_batch)

            for i in range(batch_size):
                sample_data: Dict[str, Any] = {}
                if include_label: sample_data["label"] = labels_list[i] if labels_list and i < len(labels_list) else None
                if include_image_id: sample_data[final_id_col_name] = image_ids_list[i] if image_ids_list and i < len(image_ids_list) else None

                if hooks_needed:
                    for key, batch_tensor in consolidated_batch_data.items():
                        # Use the suffix determined in __init__
                        layer_name = key.replace(self.data_suffix, "")
                        col_name_base = f"{layer_name}{self.data_suffix}" # Base name for columns

                        try:
                            if i < batch_tensor.shape[0]:
                                sample_item = batch_tensor[i]
                                # Processing logic (GAP vs Flatten) remains the same based on tensor dim
                                if sample_item.ndim >= 3:
                                    if apply_gap_to_high_dim:
                                        spatial_dims = tuple(range(1, sample_item.ndim))
                                        col_name_gap = f"{col_name_base}_avg"
                                        sample_data[col_name_gap] = sample_item.mean(dim=spatial_dims).tolist()
                                    else:
                                        sample_data[col_name_base] = sample_item.flatten().tolist()
                                elif sample_item.ndim <= 2 and sample_item.ndim > 0 : # Covers 2D, 1D
                                     sample_data[col_name_base] = sample_item.flatten().tolist() # Flatten 2D, keep 1D as list
                                elif sample_item.ndim == 0:
                                    sample_data[col_name_base] = sample_item.item()
                                else: # Should not happen
                                     sample_data[f"{layer_name}_error"] = f"Unexpected shape {sample_item.shape}"

                            else: # Index out of bounds
                                col_to_store_none = f"{col_name_base}_avg" if apply_gap_to_high_dim and batch_tensor.ndim >=3 else col_name_base
                                sample_data[col_to_store_none] = None
                        except Exception as e:
                            sample_data[f"{layer_name}_error"] = str(e)

                self.extracted_data_list.append(sample_data)
            # --- End Sample Extraction ---

            # --- 8. Batch Cleanup ---
            # ... (same as before) ...
            del consolidated_batch_data, batch_storage, images, labels_batch, image_ids_batch, model_outputs_batch
            torch.cuda.empty_cache() if self.device.type == 'cuda' else None

        # --- 9. Final Output Creation & Exploding ---
        if not self.extracted_data_list and not self.model_outputs_list:
             warnings.warn("No layer data extracted AND no model outputs collected. Returning empty.", UserWarning)
             return polars.DataFrame(), []

        print(f"Converting {len(self.extracted_data_list)} sample records to Polars DataFrame...")
        results_df = None
        try:
            results_df = polars.DataFrame(self.extracted_data_list, strict=False)

            # <<< MODIFIED: Use dynamic suffix >>>
            potential_list_cols = [
                c for c in results_df.columns
                if c.endswith(self.data_suffix) or c.endswith(f"{self.data_suffix}_avg")
            ]
            list_cols_to_explode = [
                c for c in potential_list_cols if results_df.columns and c in results_df.columns and results_df[c].dtype == polars.List
            ] # Added check c in results_df.columns
            # ... (rest of exploding logic is the same, using the identified list_cols_to_explode) ...
            non_list_activation_cols = set(potential_list_cols) - set(list_cols_to_explode)
            if non_list_activation_cols: print(f"Note: Columns {non_list_activation_cols} match naming but are not List type (likely scalar).")

            if list_cols_to_explode:
                print(f"Exploding list columns: {list_cols_to_explode}")
                all_new_col_exprs = []
                max_lengths = {}
                for col_name in list_cols_to_explode:
                    try: # Simplified max_len calculation from previous step
                        if col_name in results_df.columns:
                            max_len = results_df.select(polars.col(col_name).list.len().max()).item()
                            max_lengths[col_name] = max_len if max_len is not None else 0
                        else: max_lengths[col_name] = 0
                    except Exception as e: warnings.warn(f"Max length error for '{col_name}': {e}. Skipping.", UserWarning); max_lengths[col_name] = 0

                for col_name in list_cols_to_explode:
                     if col_name in results_df.columns:
                        max_len = max_lengths.get(col_name, 0)
                        if max_len > 0:
                            for i in range(max_len):
                                new_col_name = f"{col_name}_{i}"
                                expr = polars.col(col_name).list.get(i).alias(new_col_name)
                                all_new_col_exprs.append(expr)

                if all_new_col_exprs:
                    existing_cols = [c for c in results_df.columns if c not in list_cols_to_explode]
                    results_df = results_df.select(existing_cols + all_new_col_exprs)
                    print(f"DataFrame shape after exploding: {results_df.shape}")
                else: print("No list columns with elements found to explode.") # This might appear if max_len was always 0
            else: print("No columns identified as List type for explosion.")

            # Reorder columns
            final_cols = results_df.columns
            ordered_cols = []
            if final_id_col_name in final_cols: ordered_cols.append(final_id_col_name)
            if "label" in final_cols: ordered_cols.append("label")
            ordered_cols.extend(sorted([c for c in final_cols if c not in ordered_cols]))
            results_df = results_df.select(ordered_cols)

            return results_df, self.model_outputs_list

        except Exception as e:
            error_msg = f"Polars DataFrame processing failed: {e}\n{traceback.format_exc()}"
            warnings.warn(f"{error_msg}\nReturning raw list of dictionaries for layer data.", UserWarning)
            return self.extracted_data_list, self.model_outputs_list