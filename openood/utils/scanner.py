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
    Analyzes trained PyTorch models by extracting pre-activations (layer inputs).

    Targets inputs to specified nn.Linear/nn.ConvNd layers.
    Conv layer inputs (multi-dimensional) are summarized via Global Average Pooling (GAP).
    Outputs a Polars DataFrame where each row is a sample, and pre-activations
    are expanded into individual columns (e.g., 'layer_name_pre_activation_0', ...).
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
                                whose inputs should be captured.
        """
        if not target_layer_names:
            raise ValueError("`target_layer_names` list cannot be empty.")

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
        self._identify_target_modules()

        self.extracted_data_list: List[Dict[str, Any]] = []
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []

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
        if not self.target_modules:
            raise ValueError(
                "None of the specified target_layer_names were found or are valid types."
            )

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
    ) -> Union[polars.DataFrame, List[Dict[str, Any]]]:
        """
        Performs inference and extracts pre-activations, returning an exploded DataFrame.

        Args:
            dataloader: DataLoader yielding batches (list, tuple, or dict).
            include_label (bool): Include 'label' column.
            include_image_id (bool): Include column named by `id_key`.
            image_key (str): Key for image data if batch is dict.
            label_key (str): Key for label data if batch is dict.
            id_key (str): Key for image/sample ID if batch is dict; also the output column name.

        Returns:
            Polars DataFrame with pre-activations exploded into columns, or
            the raw list of dictionaries if DataFrame processing fails.
        """
        self.extracted_data_list = []
        self.model.eval()
        self.model.to(self.device)  # Ensure model is on correct device

        batch_num = 0
        hooks_needed = bool(self.target_modules)
        iterator = tqdm(dataloader, desc="Scanning Batches")
        final_id_col_name = id_key  # Store the intended ID column name

        # --- TEMPORARY MODIFICATION ---
        max_batches_to_process = 50  # <<< SET YOUR DESIRED NUMBER OF BATCHES HERE
        # --- END TEMPORARY MODIFICATION ---

        for batch in iterator:
            batch_num += 1

            # --- TEMPORARY MODIFICATION ---
            if batch_num > max_batches_to_process:
                print(
                    f"\nStopping early after processing {max_batches_to_process} batches."
                )
                break
            # --- END TEMPORARY MODIFICATION ---

            images = None
            labels_batch = None
            image_ids_batch = None
            batch_size = 0

            # --- 1. Batch Setup (Handles list/tuple/dict) ---
            if isinstance(batch, (list, tuple)):
                if not batch:
                    continue  # Skip empty batch
                try:
                    images = batch[0].to(self.device)
                    batch_size = images.shape[0]
                    if include_label and len(batch) > 1:
                        labels_batch = batch[1]
                    if include_image_id and len(batch) > 2:
                        image_ids_batch = batch[2]
                    # Basic validation: ensure images is a tensor
                    if not isinstance(images, torch.Tensor):
                        raise TypeError("Image data is not a Tensor")
                except (IndexError, AttributeError, TypeError, RuntimeError) as e:
                    warnings.warn(
                        f"Skipping batch {batch_num}: Error accessing batch data - {e}",
                        UserWarning,
                    )
                    continue  # Skip malformed batch
            elif isinstance(batch, dict):
                if image_key not in batch:
                    warnings.warn(
                        f"Skipping batch {batch_num}: Missing image key '{image_key}'.",
                        UserWarning,
                    )
                    continue  # Skip if missing image key
                try:
                    images = batch[image_key].to(self.device)
                    batch_size = images.shape[0]
                    if include_label:
                        labels_batch = batch.get(label_key)  # Safely get label
                    if include_image_id:
                        image_ids_batch = batch.get(id_key)  # Safely get ID
                    # Basic validation: ensure images is a tensor
                    if not isinstance(images, torch.Tensor):
                        raise TypeError("Image data is not a Tensor")
                except (AttributeError, TypeError, RuntimeError) as e:
                    warnings.warn(
                        f"Skipping batch {batch_num}: Error accessing dict batch data - {e}",
                        UserWarning,
                    )
                    continue  # Skip if image access fails
            else:
                warnings.warn(
                    f"Skipping batch {batch_num}: Unsupported type {type(batch)}.",
                    UserWarning,
                )
                continue

            if images is None or batch_size == 0:
                continue  # Should not happen if logic above is sound

            if labels_batch is not None and isinstance(labels_batch, torch.Tensor):
                labels_batch = labels_batch.cpu()  # Move labels to CPU early
            if image_ids_batch is not None and isinstance(
                image_ids_batch, torch.Tensor
            ):
                image_ids_batch = image_ids_batch.cpu()  # Move IDs to CPU early

            # --- 2. Batch Storage Init ---
            batch_storage = defaultdict(list)
            self._hook_handles = []

            # --- 3. Hook Registration ---
            if hooks_needed:

                def create_forward_hook(layer_name: str):
                    def forward_hook(
                        module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: Any
                    ):
                        # Capture first input tensor if valid
                        if inputs and isinstance(inputs[0], torch.Tensor):
                            batch_storage[f"{layer_name}_pre_activation"].append(
                                inputs[0].detach().cpu()
                            )

                    return forward_hook

                for name, module in self.target_modules.items():
                    handle = module.register_forward_hook(create_forward_hook(name))
                    self._hook_handles.append(handle)

            # --- 4. Forward Pass ---
            try:
                with torch.no_grad():
                    _ = self.model(images)
            except Exception as e:
                warnings.warn(
                    f"Forward pass failed batch {batch_num}: {e}. Skipping.",
                    RuntimeWarning,
                )
                self._clear_hooks()
                # Cleanup potentially large tensors from failed batch
                del images, labels_batch, image_ids_batch, batch_storage
                torch.cuda.empty_cache() if self.device.type == "cuda" else None
                continue

            # --- 5. Hook Removal ---
            self._clear_hooks()

            # --- 6. Data Consolidation ---
            consolidated_batch_data: Dict[str, torch.Tensor] = {}
            for key, tensor_list in batch_storage.items():
                if tensor_list:
                    # Use first captured tensor if hook fired multiple times (unusual)
                    # Ensure consistency: stack if multiple tensors *per batch* were intended,
                    # but here we expect one tensor *representing the batch*.
                    if len(tensor_list) > 1:
                        warnings.warn(
                            f"Layer '{key.replace('_pre_activation', '')}' hook fired {len(tensor_list)} times in batch {batch_num}. Using first capture.",
                            RuntimeWarning,
                        )
                    consolidated_batch_data[key] = tensor_list[
                        0
                    ]  # Assumes hook gives full batch tensor

            # --- 7. Sample Data Extraction & Appending ---
            labels_list = None
            if include_label and labels_batch is not None:
                try:
                    labels_list = (
                        labels_batch.tolist()
                        if isinstance(labels_batch, torch.Tensor)
                        else list(labels_batch)
                    )
                    if len(labels_list) != batch_size:
                        warnings.warn(
                            f"Label count mismatch in batch {batch_num} (Expected {batch_size}, Got {len(labels_list)}). Padding with None.",
                            UserWarning,
                        )
                        labels_list.extend([None] * (batch_size - len(labels_list)))
                except Exception as e:
                    warnings.warn(
                        f"Error processing labels in batch {batch_num}: {e}",
                        UserWarning,
                    )
                    labels_list = [None] * batch_size

            image_ids_list = None
            if include_image_id and image_ids_batch is not None:
                try:
                    image_ids_list = (
                        image_ids_batch.tolist()
                        if isinstance(image_ids_batch, torch.Tensor)
                        else list(image_ids_batch)
                    )
                    if len(image_ids_list) != batch_size:
                        warnings.warn(
                            f"Image ID count mismatch in batch {batch_num} (Expected {batch_size}, Got {len(image_ids_list)}). Padding with None.",
                            UserWarning,
                        )
                        image_ids_list.extend(
                            [None] * (batch_size - len(image_ids_list))
                        )
                except Exception as e:
                    warnings.warn(
                        f"Error processing image IDs in batch {batch_num}: {e}",
                        UserWarning,
                    )
                    image_ids_list = [None] * batch_size

            # Process activations sample by sample
            for i in range(batch_size):
                sample_data: Dict[str, Any] = {}
                if include_label:
                    sample_data["label"] = (
                        labels_list[i] if labels_list and i < len(labels_list) else None
                    )
                if include_image_id:
                    sample_data[final_id_col_name] = (
                        image_ids_list[i]
                        if image_ids_list and i < len(image_ids_list)
                        else None
                    )

                # Extract activation data for this sample
                for key, batch_tensor in consolidated_batch_data.items():
                    layer_name = key.replace("_pre_activation", "")
                    # Determine column names based on processing type
                    col_name_act = (
                        f"{layer_name}_pre_activation"  # For Linear layers (or scalar)
                    )
                    col_name_gap = f"{layer_name}_pre_activation_avg"  # For Conv layers (GAP applied)

                    try:
                        if i < batch_tensor.shape[0]:
                            sample_item = batch_tensor[
                                i
                            ]  # Data for sample i (already on CPU)

                            # Apply GAP to multi-dim tensors (likely Conv inputs), store list/scalar otherwise
                            if sample_item.ndim >= 3:  # e.g., (C, H, W) or (C, D, H, W)
                                spatial_dims = tuple(
                                    range(1, sample_item.ndim)
                                )  # Dims other than Channel
                                sample_data[col_name_gap] = sample_item.mean(
                                    dim=spatial_dims
                                ).tolist()
                            elif (
                                sample_item.ndim == 2
                            ):  # e.g., (SeqLen, Features) -> GAP over SeqLen? Or error? Assume GAP for now.
                                warnings.warn(
                                    f"Unexpected 2D tensor shape {sample_item.shape} for {layer_name}. Applying GAP over dim 0.",
                                    UserWarning,
                                )
                                sample_data[col_name_gap] = sample_item.mean(
                                    dim=0
                                ).tolist()
                            elif sample_item.ndim == 1:  # Linear input (Features,)
                                sample_data[col_name_act] = sample_item.tolist()
                            elif sample_item.ndim == 0:  # Scalar input/activation
                                sample_data[col_name_act] = sample_item.item()
                            else:  # Should not happen
                                warnings.warn(
                                    f"Unexpected tensor shape {sample_item.shape} (ndim={sample_item.ndim}) for {layer_name}. Skipping.",
                                    UserWarning,
                                )
                                sample_data[f"{layer_name}_error"] = (
                                    f"Unexpected shape {sample_item.shape}"
                                )
                        else:
                            # Index out of bounds for this tensor in the batch (shouldn't normally happen if batching is correct)
                            warnings.warn(
                                f"Sample index {i} out of bounds for tensor {key} (shape {batch_tensor.shape}) in batch {batch_num}.",
                                UserWarning,
                            )
                            # Assign None to a potential column, maybe the GAP one as a default
                            sample_data[
                                col_name_gap if batch_tensor.ndim >= 3 else col_name_act
                            ] = None

                    except Exception as e:
                        warnings.warn(
                            f"Error processing {key} for sample {i} (index) in batch {batch_num}: {e}",
                            UserWarning,
                        )
                        sample_data[f"{layer_name}_error"] = str(
                            e
                        )  # Log error per sample

                self.extracted_data_list.append(sample_data)

            # --- 8. Batch Cleanup ---
            del (
                consolidated_batch_data,
                batch_storage,
                images,
                labels_batch,
                image_ids_batch,
            )
            torch.cuda.empty_cache() if self.device.type == "cuda" else None

        # --- 9. Final Output Creation & Exploding ---
        if not self.extracted_data_list:
            warnings.warn("No data extracted. Returning empty DataFrame.", UserWarning)
            return polars.DataFrame()

        print(
            f"Converting {len(self.extracted_data_list)} samples to Polars DataFrame..."
        )
        try:
            # Create initial DataFrame with potentially mixed dtypes and list columns
            results_df = polars.DataFrame(self.extracted_data_list, strict=False)

            # Identify list columns potentially needing explosion (based on naming convention)
            # Also check if they are actually List type after initial DF creation
            potential_list_cols = [
                c
                for c in results_df.columns
                if c.endswith("_pre_activation") or c.endswith("_pre_activation_avg")
            ]
            list_cols_to_explode = [
                c for c in potential_list_cols if results_df[c].dtype == polars.List
            ]
            non_list_activation_cols = set(potential_list_cols) - set(
                list_cols_to_explode
            )
            if non_list_activation_cols:
                print(
                    f"Note: Columns {non_list_activation_cols} match naming but are not List type (likely scalar activations)."
                )

            # Explode list columns if any were generated AND are of List type
            if list_cols_to_explode:
                print(f"Exploding list columns: {list_cols_to_explode}")
                all_new_col_exprs = []

                # Calculate max length for each list column and generate expressions
                # This requires executing a small query per column before building the main one
                max_lengths = {}
                for col_name in list_cols_to_explode:
                    try:
                        # Compute the actual max length by executing the expression immediately
                        # Filter out nulls before checking length, in case some samples errored out
                        max_len = results_df.select(
                            polars.col(col_name)
                            .filter(polars.col(col_name).is_not_null())
                            .list.len()
                            .max()
                        ).item()
                        max_lengths[col_name] = max_len if max_len is not None else 0
                    except Exception as e:
                        warnings.warn(
                            f"Could not determine max length for column '{col_name}': {e}. Skipping explosion for this column.",
                            UserWarning,
                        )
                        max_lengths[col_name] = 0  # Treat as empty if error

                # Generate the list.get expressions
                for col_name in list_cols_to_explode:
                    max_len = max_lengths.get(col_name, 0)
                    if max_len > 0:
                        # Generate expressions to extract each element into a new column
                        for i in range(max_len):
                            new_col_name = f"{col_name}_{i}"
                            # Use list.get(i), which handles nulls/short lists gracefully (returns null)
                            expr = polars.col(col_name).list.get(i).alias(new_col_name)
                            all_new_col_exprs.append(expr)
                    # else: # No need to warn again if max_len is 0
                    #    warnings.warn(f"Column '{col_name}' has max list length 0. No columns generated.", UserWarning)

                if all_new_col_exprs:
                    # Add all the new exploded columns and drop the original list columns in one go
                    results_df = results_df.with_columns(all_new_col_exprs).drop(
                        list_cols_to_explode
                    )
                    print(f"DataFrame shape after exploding: {results_df.shape}")
                else:
                    print("No list columns with elements found to explode.")
            else:
                print("No columns identified as List type for explosion.")

            # Reorder columns: Put ID (using final_id_col_name) and Label first
            final_cols = results_df.columns
            ordered_cols = []
            if final_id_col_name in final_cols:
                ordered_cols.append(final_id_col_name)
            if "label" in final_cols:
                ordered_cols.append("label")
            # Sort remaining columns alphabetically for consistency
            ordered_cols.extend(
                sorted([c for c in final_cols if c not in ordered_cols])
            )

            return results_df.select(ordered_cols)

        except Exception as e:
            # Fallback if DF creation, exploding, or reordering fails
            # Include traceback for better debugging
            import traceback

            error_msg = (
                f"Polars DataFrame processing failed: {e}\n{traceback.format_exc()}"
            )
            warnings.warn(
                f"{error_msg}\nReturning raw list of dictionaries.", UserWarning
            )
            return self.extracted_data_list  # Return original list on failure
