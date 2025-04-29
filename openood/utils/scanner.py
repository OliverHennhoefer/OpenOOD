# -*- coding: utf-8 -*-
import torch
import polars
import warnings
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Tuple, Union, Optional


class NetworkScanner:
    """
    Analyzes trained PyTorch models by extracting pre-activations (layer inputs)
    and model outputs.

    Targets inputs to specified nn.Linear/nn.ConvNd layers.
    Conv layer inputs (multi-dimensional) are summarized via Global Average Pooling (GAP).
    Outputs a tuple containing:
        1. A Polars DataFrame where each row is a sample, and pre-activations
           are expanded into individual columns (e.g., 'layer_name_pre_activation_0', ...).
        2. A list containing the raw model outputs for each sample processed.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer_names: List[str],
    ):
        """
        Args:
            model: Pre-trained PyTorch model (nn.Module).
            target_layer_names: List of layer names (from model.named_modules())
                                whose inputs should be captured. Can be empty if
                                only model outputs are desired.
        """
        # Allow empty target_layer_names if only outputs are needed
        # if not target_layer_names:
        #     raise ValueError("`target_layer_names` list cannot be empty.")

        self.model = model
        self.model.eval()

        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:
            warnings.warn("Model has no parameters. Assuming CPU.", UserWarning)
            self.device = torch.device("cpu")
        self.model.to(self.device)

        self.target_layer_names = target_layer_names
        self.target_modules: Dict[str, nn.Module] = {}
        # Only identify modules if names are provided
        if self.target_layer_names:
            self._identify_target_modules()
        else:
             warnings.warn(
                "`target_layer_names` is empty. Only model outputs will be collected.",
                UserWarning
            )


        self.extracted_data_list: List[Dict[str, Any]] = []
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self.model_outputs_list: List[Any] = [] # <<< ADDED: To store model outputs

    def _identify_target_modules(self):
        """Finds the nn.Module objects corresponding to target_layer_names."""
        found_names = set()
        for name, module in self.model.named_modules():
            if name in self.target_layer_names:
                # Ensure only valid layer types are targeted for hooks if needed
                # (Original code didn't check type here, assuming user provides valid names)
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    self.target_modules[name] = module
                    found_names.add(name)
                else:
                    warnings.warn(
                        f"Layer '{name}' is not a Linear or ConvNd layer, "
                        f"but was included in target_layer_names. Hooking it anyway.",
                        UserWarning,
                    )
                    self.target_modules[name] = (
                        module  # Hook anyway as per original logic
                    )
                    found_names.add(name)

        missing_names = set(self.target_layer_names) - found_names
        if missing_names:
            warnings.warn(
                f"Target layers not found and ignored: {missing_names}.", UserWarning
            )
        # Allow proceeding even if no target layers are found, if list was initially non-empty but contained invalid names
        # if not self.target_modules and self.target_layer_names: # Check if list was non-empty
        #     raise ValueError(
        #         "None of the specified target_layer_names were found or are valid types."
        #     )
        # Don't raise error if target_layer_names was empty from the start

    def _clear_hooks(self):
        """Removes all registered forward hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def predict(
        self,
        dataloader: DataLoader,
        include_label: bool = True,
        include_image_id: bool = True,
        image_key: str = "data",
        label_key: str = "label",
        id_key: str = "index",  # Common key for sample index/ID
    ) -> Tuple[Union[polars.DataFrame, List[Dict[str, Any]]], List[Any]]: # <<< MODIFIED Return Type
        """
        Performs inference, extracts pre-activations, and captures model outputs.

        Args:
            dataloader: DataLoader yielding batches (list, tuple, or dict).
            include_label (bool): Include 'label' column in the DataFrame.
            include_image_id (bool): Include column named by `id_key` in the DataFrame.
            image_key (str): Key for image data if batch is dict.
            label_key (str): Key for label data if batch is dict.
            id_key (str): Key for image/sample ID if batch is dict; also the output column name.

        Returns:
            A tuple containing:
            - Polars DataFrame with pre-activations exploded into columns (or the
              raw list of dictionaries if DataFrame processing fails).
            - A list of the raw model outputs for each processed sample.
        """
        self.extracted_data_list = []
        self.model_outputs_list = [] # <<< ADDED: Reset output list
        self.model.eval()
        self.model.to(self.device)  # Ensure model is on correct device

        batch_num = 0
        hooks_needed = bool(self.target_modules) # Only need hooks if target layers exist
        iterator = tqdm(dataloader, desc="Scanning Batches")
        final_id_col_name = id_key  # Store the intended ID column name

        # --- TEMPORARY MODIFICATION ---
        #max_batches_to_process = 30  # <<< SET YOUR DESIRED NUMBER OF BATCHES HERE
        # --- END TEMPORARY MODIFICATION ---

        for batch in iterator:
            batch_num += 1

            # --- TEMPORARY MODIFICATION ---
            #if batch_num > max_batches_to_process:
            #    print(
            #        f"\nStopping early after processing {max_batches_to_process} batches."
            #    )
            #    break
            # --- END TEMPORARY MODIFICATION ---

            images = None
            labels_batch = None
            image_ids_batch = None
            batch_size = 0

            # --- 1. Batch Setup (Handles list/tuple/dict) ---
            # (Code identical to original)
            if isinstance(batch, (list, tuple)):
                if not batch: continue
                try:
                    images = batch[0].to(self.device)
                    batch_size = images.shape[0]
                    if include_label and len(batch) > 1: labels_batch = batch[1]
                    if include_image_id and len(batch) > 2: image_ids_batch = batch[2]
                    if not isinstance(images, torch.Tensor): raise TypeError("Image data is not a Tensor")
                except (IndexError, AttributeError, TypeError, RuntimeError) as e:
                    warnings.warn(f"Skipping batch {batch_num}: Error accessing batch data - {e}", UserWarning); continue
            elif isinstance(batch, dict):
                if image_key not in batch: warnings.warn(f"Skipping batch {batch_num}: Missing image key '{image_key}'.", UserWarning); continue
                try:
                    images = batch[image_key].to(self.device)
                    batch_size = images.shape[0]
                    if include_label: labels_batch = batch.get(label_key)
                    if include_image_id: image_ids_batch = batch.get(id_key)
                    if not isinstance(images, torch.Tensor): raise TypeError("Image data is not a Tensor")
                except (AttributeError, TypeError, RuntimeError) as e:
                    warnings.warn(f"Skipping batch {batch_num}: Error accessing dict batch data - {e}", UserWarning); continue
            else:
                warnings.warn(f"Skipping batch {batch_num}: Unsupported type {type(batch)}.", UserWarning); continue

            if images is None or batch_size == 0: continue

            if labels_batch is not None and isinstance(labels_batch, torch.Tensor): labels_batch = labels_batch.cpu()
            if image_ids_batch is not None and isinstance(image_ids_batch, torch.Tensor): image_ids_batch = image_ids_batch.cpu()
            # --- End Batch Setup ---

            # --- 2. Batch Storage Init ---
            batch_storage = defaultdict(list)
            self._hook_handles = []

            # --- 3. Hook Registration ---
            # (Code identical to original, but only runs if hooks_needed is True)
            if hooks_needed:
                def create_forward_hook(layer_name: str):
                    def forward_hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: Any):
                        if inputs and isinstance(inputs[0], torch.Tensor):
                            batch_storage[f"{layer_name}_pre_activation"].append(inputs[0].detach().cpu())
                    return forward_hook

                for name, module in self.target_modules.items():
                    handle = module.register_forward_hook(create_forward_hook(name))
                    self._hook_handles.append(handle)

            # --- 4. Forward Pass & Output Capture ---
            model_outputs_batch = None # <<< Initialize variable
            try:
                with torch.no_grad():
                    # <<< MODIFIED: Capture the output
                    model_outputs_batch = self.model(images)
            except Exception as e:
                warnings.warn(
                    f"Forward pass failed batch {batch_num}: {e}. Skipping.",
                    RuntimeWarning,
                )
                self._clear_hooks()
                del images, labels_batch, image_ids_batch, batch_storage
                torch.cuda.empty_cache() if self.device.type == "cuda" else None
                continue

            # --- 5. Hook Removal ---
            # (Clear hooks regardless of whether they were needed, safer)
            self._clear_hooks()

            # --- 6. Data Consolidation (Pre-Activations) ---
            # (Code identical to original)
            consolidated_batch_data: Dict[str, torch.Tensor] = {}
            if hooks_needed: # Only process if hooks were registered and might have captured data
                 for key, tensor_list in batch_storage.items():
                    if tensor_list:
                        if len(tensor_list) > 1:
                            warnings.warn(f"Layer '{key.replace('_pre_activation', '')}' hook fired {len(tensor_list)} times in batch {batch_num}. Using first capture.", RuntimeWarning)
                        consolidated_batch_data[key] = tensor_list[0]

            # --- 6b. Store Model Outputs --- <<< ADDED
            if model_outputs_batch is not None:
                try:
                    # Move to CPU, detach, and convert to list (list of lists/scalars per sample)
                    processed_outputs = model_outputs_batch.detach().cpu().tolist()
                    self.model_outputs_list.extend(processed_outputs)
                except Exception as e:
                     warnings.warn(
                        f"Could not process/store model outputs for batch {batch_num}: {e}. Appending None placeholders.",
                        RuntimeWarning,
                    )
                     # Add placeholders if conversion fails to keep output list length consistent
                     self.model_outputs_list.extend([None] * batch_size)


            # --- 7. Sample Data Extraction & Appending (Pre-Activations) ---
            # (Code identical to original)
            labels_list = None
            if include_label and labels_batch is not None:
                try:
                    labels_list = labels_batch.tolist() if isinstance(labels_batch, torch.Tensor) else list(labels_batch)
                    if len(labels_list) != batch_size:
                        warnings.warn(f"Label count mismatch in batch {batch_num} (Expected {batch_size}, Got {len(labels_list)}). Padding with None.", UserWarning)
                        labels_list.extend([None] * (batch_size - len(labels_list)))
                except Exception as e:
                    warnings.warn(f"Error processing labels in batch {batch_num}: {e}", UserWarning); labels_list = [None] * batch_size

            image_ids_list = None
            if include_image_id and image_ids_batch is not None:
                try:
                    image_ids_list = image_ids_batch.tolist() if isinstance(image_ids_batch, torch.Tensor) else list(image_ids_batch)
                    if len(image_ids_list) != batch_size:
                        warnings.warn(f"Image ID count mismatch in batch {batch_num} (Expected {batch_size}, Got {len(image_ids_list)}). Padding with None.", UserWarning)
                        image_ids_list.extend([None] * (batch_size - len(image_ids_list)))
                except Exception as e:
                    warnings.warn(f"Error processing image IDs in batch {batch_num}: {e}", UserWarning); image_ids_list = [None] * batch_size

            # Process activations sample by sample
            for i in range(batch_size):
                sample_data: Dict[str, Any] = {}
                if include_label: sample_data["label"] = labels_list[i] if labels_list and i < len(labels_list) else None
                if include_image_id: sample_data[final_id_col_name] = image_ids_list[i] if image_ids_list and i < len(image_ids_list) else None

                # Extract activation data for this sample if hooks were used
                if hooks_needed:
                    for key, batch_tensor in consolidated_batch_data.items():
                        layer_name = key.replace("_pre_activation", "")
                        col_name_act = f"{layer_name}_pre_activation"
                        col_name_gap = f"{layer_name}_pre_activation_avg"

                        try:
                            if i < batch_tensor.shape[0]:
                                sample_item = batch_tensor[i]
                                if sample_item.ndim >= 3:
                                    spatial_dims = tuple(range(1, sample_item.ndim))
                                    sample_data[col_name_gap] = sample_item.mean(dim=spatial_dims).tolist()
                                elif sample_item.ndim == 2:
                                    warnings.warn(f"Unexpected 2D tensor shape {sample_item.shape} for {layer_name}. Applying GAP over dim 0.", UserWarning)
                                    sample_data[col_name_gap] = sample_item.mean(dim=0).tolist()
                                elif sample_item.ndim == 1:
                                    sample_data[col_name_act] = sample_item.tolist()
                                elif sample_item.ndim == 0:
                                    sample_data[col_name_act] = sample_item.item()
                                else:
                                    warnings.warn(f"Unexpected tensor shape {sample_item.shape} (ndim={sample_item.ndim}) for {layer_name}. Skipping.", UserWarning)
                                    sample_data[f"{layer_name}_error"] = f"Unexpected shape {sample_item.shape}"
                            else:
                                warnings.warn(f"Sample index {i} out of bounds for tensor {key} (shape {batch_tensor.shape}) in batch {batch_num}.", UserWarning)
                                sample_data[col_name_gap if batch_tensor.ndim >= 3 else col_name_act] = None
                        except Exception as e:
                            warnings.warn(f"Error processing {key} for sample {i} (index) in batch {batch_num}: {e}", UserWarning)
                            sample_data[f"{layer_name}_error"] = str(e)

                self.extracted_data_list.append(sample_data)
            # --- End Sample Extraction ---


            # --- 8. Batch Cleanup ---
            # (Code identical to original)
            del consolidated_batch_data, batch_storage, images, labels_batch, image_ids_batch, model_outputs_batch # <<< Added model_outputs_batch
            torch.cuda.empty_cache() if self.device.type == "cuda" else None
            # --- End Batch Cleanup ---

        # --- 9. Final Output Creation & Exploding ---
        # (Code identical to original for DataFrame part)
        if not self.extracted_data_list and not self.model_outputs_list: # Check both lists
             warnings.warn("No pre-activation data extracted AND no model outputs collected. Returning empty DataFrame and empty list.", UserWarning)
             return polars.DataFrame(), [] # <<< MODIFIED return

        # Create DataFrame even if only model outputs were collected (will contain only id/label if requested)
        print(f"Converting {len(self.extracted_data_list)} sample records to Polars DataFrame...")
        results_df = None # Initialize
        try:
            # Create initial DataFrame with potentially mixed dtypes and list columns
            results_df = polars.DataFrame(self.extracted_data_list, strict=False)

            potential_list_cols = [c for c in results_df.columns if c.endswith("_pre_activation") or c.endswith("_pre_activation_avg")]
            list_cols_to_explode = [c for c in potential_list_cols if results_df[c].dtype == polars.List]
            non_list_activation_cols = set(potential_list_cols) - set(list_cols_to_explode)
            if non_list_activation_cols: print(f"Note: Columns {non_list_activation_cols} match naming but are not List type (likely scalar activations).")

            if list_cols_to_explode:
                print(f"Exploding list columns: {list_cols_to_explode}")
                all_new_col_exprs = []
                max_lengths = {}
                for col_name in list_cols_to_explode:
                    try:
                        max_len = results_df.select(polars.col(col_name).filter(polars.col(col_name).is_not_null()).list.len().max()).item()
                        max_lengths[col_name] = max_len if max_len is not None else 0
                    except Exception as e:
                        warnings.warn(f"Could not determine max length for column '{col_name}': {e}. Skipping explosion for this column.", UserWarning)
                        max_lengths[col_name] = 0

                for col_name in list_cols_to_explode:
                    max_len = max_lengths.get(col_name, 0)
                    if max_len > 0:
                        for i in range(max_len):
                            new_col_name = f"{col_name}_{i}"
                            expr = polars.col(col_name).list.get(i).alias(new_col_name)
                            all_new_col_exprs.append(expr)

                if all_new_col_exprs:
                    results_df = results_df.with_columns(all_new_col_exprs).drop(list_cols_to_explode)
                    print(f"DataFrame shape after exploding: {results_df.shape}")
                else: print("No list columns with elements found to explode.")
            else: print("No columns identified as List type for explosion.")

            # Reorder columns: Put ID (using final_id_col_name) and Label first
            final_cols = results_df.columns
            ordered_cols = []
            if final_id_col_name in final_cols: ordered_cols.append(final_id_col_name)
            if "label" in final_cols: ordered_cols.append("label")
            ordered_cols.extend(sorted([c for c in final_cols if c not in ordered_cols]))
            results_df = results_df.select(ordered_cols)

            # <<< MODIFIED: Return tuple
            return results_df, self.model_outputs_list

        except Exception as e:
            import traceback
            error_msg = f"Polars DataFrame processing failed: {e}\n{traceback.format_exc()}"
            warnings.warn(f"{error_msg}\nReturning raw list of dictionaries for pre-activations.", UserWarning)

            # <<< MODIFIED: Return tuple even in error case
            # Return the raw pre-activation list and the collected model outputs
            return self.extracted_data_list, self.model_outputs_list