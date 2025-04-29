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
    Analyzes trained PyTorch CNN models (like ResNet) by performing inference
    and extracting pre-activations (inputs to specified layers).

    Specifically targets inputs to nn.Linear and nn.ConvNd layers from a
    user-defined list. Multi-dimensional inputs (typically to Conv layers)
    are summarized using Global Average Pooling (GAP) per channel.

    Outputs results as a Polars DataFrame, where each row represents an input
    sample and columns contain the extracted pre-activation vectors (for Linear)
    or GAPped vectors (for Conv).
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer_names: List[str],
    ):
        """
        Initializes the NetworkScanner.

        Args:
            model: The pre-trained PyTorch model (nn.Module or optionally
                   LightningModule if installed) to analyze. Should be
                   pre-loaded with weights and set to the correct device
                   before passing.
            target_layer_names: A list of strings specifying the exact names
                                (from `model.named_modules()`) of the layers
                                whose *inputs* (pre-activations) should be
                                captured. Typically Linear or ConvNd layers.
        """

        if not target_layer_names:
            raise ValueError("`target_layer_names` list cannot be empty.")

        self.model = model
        self.model.eval() # Ensure evaluation mode

        # Determine device safely
        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:
            warnings.warn("Model has no parameters. Assuming CPU.", UserWarning)
            self.device = torch.device("cpu")
        self.model.to(self.device) # Ensure model is on the determined device

        self.target_layer_names = target_layer_names
        self.target_modules: Dict[str, nn.Module] = {}
        self._identify_target_modules() # Find the actual module objects

        # Internal state reset in predict()
        self.extracted_data_list: List[Dict[str, Any]] = []
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []

    def _identify_target_modules(self):
        """Finds the nn.Module objects corresponding to target_layer_names."""
        found_names = set()
        for name, module in self.model.named_modules():
            if name in self.target_layer_names:
                self.target_modules[name] = module
                found_names.add(name)

        # Warn about any requested names that weren't found
        missing_names = set(self.target_layer_names) - found_names
        if missing_names:
            warnings.warn(
                f"Could not find the following target layers in the model: {missing_names}. "
                "These layers will be ignored.", UserWarning
            )
        if not self.target_modules:
             raise ValueError("None of the specified `target_layer_names` were found in the model.")


    def _clear_hooks(self):
        """Removes all registered forward hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def predict(
        self, dataloader: DataLoader, include_label: bool = True, include_image_id: bool = True
    ) -> Union[polars.DataFrame, List[Dict[str, Any]]]:
        """
        Performs inference and extracts pre-activations for the specified layers.

        Multi-dimensional pre-activation tensors (e.g., inputs to Conv layers)
        are summarized via Global Average Pooling (GAP) per channel and stored
        in columns named `{layer_name}_pre_activation_avg`.
        1D pre-activation tensors (e.g., inputs to Linear layers) are stored
        as lists in columns named `{layer_name}_pre_activation`.

        Args:
            dataloader: DataLoader yielding batches. Expects tuples like
                        `(images, labels)` or `(images, labels, image_ids)`.
                        Assumes images are the first element.
            include_label (bool): Whether to include the 'label' column in the output.
                                  Requires labels to be the second element in the batch.
            include_image_id (bool): Whether to include the 'image_id' column.
                                     Requires image IDs to be the third element
                                     in the batch tuple AND the dataloader must yield them.

        Returns:
            A Polars DataFrame with extracted pre-activation data for each
            sample, along with optional label and image_id. Returns the raw
            list of dictionaries if DataFrame creation fails.
        """
        self.extracted_data_list = []  # Reset storage
        self.model.eval()  # Ensure model is in evaluation mode

        # Re-check device and move model just in case it was moved after init
        try:
            current_device = next(self.model.parameters()).device
            if current_device != self.device:
                warnings.warn(f"Model device changed from {self.device} to {current_device}. Updating scanner device.", UserWarning)
                self.device = current_device
        except StopIteration:
            # Use the device determined during init (likely CPU)
             pass
        self.model.to(self.device)

        batch_num = 0
        hooks_needed = bool(self.target_modules) # Hooks are always needed if targets exist

        # Configure iterator with tqdm if available
        iterator = dataloader
        iterator = tqdm.tqdm(dataloader, desc="Scanning Batches")

        # --- Batch Iteration ---
        for batch in iterator:
            batch_num += 1

            # --- 1. Batch Setup & Data Movement ---
            if not isinstance(batch, (list, tuple)) or not batch:
                warnings.warn(f"Skipping batch {batch_num}: Expected list or tuple, got {type(batch)}.", UserWarning)
                continue

            # Determine batch content flexibly
            images = batch[0].to(self.device)
            batch_size = images.shape[0]
            labels_batch : Optional[torch.Tensor] = None
            image_ids_batch : Optional[Union[List, torch.Tensor]] = None

            if include_label and len(batch) > 1:
                 labels_batch = batch[1]
                 if isinstance(labels_batch, torch.Tensor):
                     labels_batch = labels_batch.to(self.device) # Move labels if tensor
                 # If labels aren't tensors, assume they are list/tuple and don't move

            if include_image_id and len(batch) > 2:
                image_ids_batch = batch[2] # Don't move IDs, usually list/tuple

            # --- 2. Batch Storage Initialization ---
            batch_storage = defaultdict(list) # Stores captured tensors for the batch
            self._hook_handles = [] # Clear hook handles for the new batch

            # --- 3. Hook Registration (Forward Hooks Only) ---
            if hooks_needed:
                # Define hook creation function *inside* the loop to close over batch_storage
                def create_forward_hook(layer_name: str):
                    def forward_hook(
                        module: nn.Module,
                        inputs: Tuple[torch.Tensor, ...], # Use Tuple signature
                        output: Any, # Output not used, but part of signature
                    ):
                        # Capture the *first* input tensor (pre-activation)
                        # Check if inputs tuple is not empty and first element is a tensor
                        if inputs and isinstance(inputs[0], torch.Tensor):
                            input_tensor = inputs[0]
                            key = f"{layer_name}_pre_activation"
                            # Store detached tensor on CPU to free GPU memory quickly
                            batch_storage[key].append(input_tensor.detach().cpu())
                        # else: No tensor input, or unexpected input structure - do nothing

                    return forward_hook

                # Register hooks for the target modules
                for name, module in self.target_modules.items():
                    handle = module.register_forward_hook(create_forward_hook(name))
                    self._hook_handles.append(handle)

            # --- 4. Forward Pass (No Gradients Needed) ---
            try:
                with torch.no_grad():
                    _ = self.model(images) # Execute forward pass, ignore output
                    # Forward hooks are executed during this call
            except Exception as e:
                warnings.warn(
                    f"Error during model forward pass in batch {batch_num}: {e}. "
                    "Skipping pre-activation extraction for this batch.",
                    RuntimeWarning,
                )
                self._clear_hooks() # Clean up hooks if forward pass failed
                continue # Skip to the next batch

            # --- 5. Hook Removal ---
            # Crucial: Remove hooks *after* the forward pass
            self._clear_hooks()

            # --- 6. Data Consolidation (Batch Storage -> Single Tensor per Key) ---
            consolidated_batch_data: Dict[str, torch.Tensor] = {}
            for key, tensor_list in batch_storage.items():
                if tensor_list:
                    # Expect exactly one tensor per key after the single forward pass hook trigger
                    if len(tensor_list) == 1:
                        consolidated_batch_data[key] = tensor_list[0]
                    else:
                        # This indicates multiple hook calls for the same layer in one forward pass
                        # Might happen with shared layers or unusual architectures. Use first capture.
                        warnings.warn(
                            f"Layer '{key.replace('_pre_activation', '')}' hook fired "
                            f"{len(tensor_list)} times in batch {batch_num}. "
                            "Using only the first captured pre-activation tensor. Check model structure.",
                            RuntimeWarning,
                        )
                        consolidated_batch_data[key] = tensor_list[0]

            # --- 7. Sample Data Extraction & Appending (with GAP integration) ---
            labels_list = None
            if include_label and labels_batch is not None:
                if isinstance(labels_batch, torch.Tensor):
                    labels_list = labels_batch.cpu().tolist()
                else: # Assume list/tuple-like
                    labels_list = list(labels_batch)

            image_ids_list = None
            if include_image_id and image_ids_batch is not None:
                if isinstance(image_ids_batch, torch.Tensor):
                     image_ids_list = image_ids_batch.tolist() # Assume CPU tensor or move if needed
                else:
                     image_ids_list = list(image_ids_batch)


            for i in range(batch_size): # Iterate through each sample in the batch
                sample_data: Dict[str, Any] = {} # Dictionary for the current sample

                # Add label if requested and available
                if labels_list is not None:
                    try:
                        sample_data["label"] = labels_list[i]
                    except IndexError:
                         warnings.warn(f"Label index {i} out of bounds for batch {batch_num}.", UserWarning)
                         sample_data["label"] = None

                # Add image ID if requested and available
                if image_ids_list is not None:
                    try:
                        sample_data["image_id"] = image_ids_list[i]
                    except IndexError:
                        warnings.warn(f"Image ID index {i} out of bounds for batch {batch_num}.", UserWarning)
                        sample_data["image_id"] = None

                # Extract pre-activation data for this sample
                for key, batch_tensor in consolidated_batch_data.items():
                    layer_name = key.replace("_pre_activation", "") # Original layer name
                    try:
                        # Ensure tensor is valid and index is within bounds
                        if i < batch_tensor.shape[0]:
                            sample_item = batch_tensor[i] # Extract data for the i-th sample (on CPU)

                            # --- Apply GAP or store vector ---
                            if sample_item.ndim >= 3: # Input to ConvNd (e.g., C, H, W or C, D, H, W)
                                # Calculate GAP: Mean over all spatial dimensions (starting from dim 1)
                                spatial_dims = tuple(range(1, sample_item.ndim))
                                channel_means_tensor = sample_item.mean(dim=spatial_dims)
                                # Store the resulting 1D tensor (length C) as a list
                                sample_data[f"{layer_name}_pre_activation_avg"] = channel_means_tensor.tolist()

                            elif sample_item.ndim == 1: # Input to Linear (e.g., Features)
                                # Store the 1D tensor as a list
                                sample_data[f"{layer_name}_pre_activation"] = sample_item.tolist()

                            elif sample_item.ndim == 0: # Scalar input (unlikely but possible)
                                sample_data[f"{layer_name}_pre_activation"] = sample_item.item()

                            elif sample_item.ndim == 2: # E.g. input (Batch, SeqLen) to Conv1d? or (Batch, Features) to Linear?
                                # If Linear pre-act, should be caught by ndim==1 after batch removal?
                                # If Conv1D pre-act, should be caught by ndim >=3 ?
                                # This case might indicate unexpected input shape. Apply GAP over last dim as fallback.
                                warnings.warn(
                                    f"Unexpected 2D pre-activation tensor (shape {sample_item.shape}) "
                                    f"for layer '{layer_name}'. Applying mean over the last dimension "
                                    f"and storing as '{layer_name}_pre_activation_avg'.", UserWarning
                                )
                                mean_tensor = sample_item.mean(dim=-1) # Mean over last dimension
                                sample_data[f"{layer_name}_pre_activation_avg"] = mean_tensor.tolist() if mean_tensor.ndim > 0 else mean_tensor.item()


                            else: # Fallback for > 3D or other unexpected shapes
                                warnings.warn(
                                    f"Unhandled pre-activation tensor shape {sample_item.shape} for layer '{layer_name}'. "
                                    "Flattening and storing as list.", UserWarning
                                )
                                sample_data[f"{layer_name}_pre_activation"] = sample_item.flatten().tolist()

                        else: # Index out of bounds for tensor
                            warnings.warn(
                                f"Index {i} out of bounds for pre-activation tensor key '{key}' "
                                f"(shape {batch_tensor.shape}) in batch {batch_num}.", UserWarning
                            )
                            # Add placeholder None for this layer for this sample
                            if batch_tensor.ndim >= 3:
                                 sample_data[f"{layer_name}_pre_activation_avg"] = None
                            else:
                                 sample_data[f"{layer_name}_pre_activation"] = None


                    except Exception as e:
                        warnings.warn(
                            f"Error processing pre-activation key '{key}' for sample {i} in batch {batch_num}: {e}",
                            UserWarning,
                        )
                        # Store None to avoid crashing DataFrame creation
                        sample_data[key.replace("_pre_activation", "_error")] = str(e) # Or just None


                # Append the completed dictionary for this sample
                self.extracted_data_list.append(sample_data)

            # --- 8. Batch Cleanup (Optional Memory Management) ---
            del consolidated_batch_data, batch_storage
            del images, labels_batch, image_ids_batch # Delete tensors/references
            # Optional: Force garbage collection if memory is tight
            # import gc
            # gc.collect()
            # if self.device.type == 'cuda': torch.cuda.empty_cache()


        # --- 9. Final Output Creation (DataFrame) ---
        if not self.extracted_data_list:
            warnings.warn("No data extracted. Returning empty DataFrame.", UserWarning)
            return polars.DataFrame()

        print(f"Converting {len(self.extracted_data_list)} extracted samples to Polars DataFrame...")
        try:
            # Create DataFrame, be less strict about schema inference due to potential errors/Nones
            results_df = polars.DataFrame(self.extracted_data_list, strict=False)
        except Exception as e:
            warnings.warn(
                f"Polars DataFrame creation failed: {e}. Returning raw list of dictionaries.",
                UserWarning,
            )
            # Fallback: return the list of dicts if DataFrame creation fails
            return self.extracted_data_list

        # --- Reorder columns (Best Effort) ---
        cols = results_df.columns
        ordered_cols = []
        if "image_id" in cols: ordered_cols.append("image_id")
        if "label" in cols: ordered_cols.append("label")

        # Add pre-activation columns, sorted alphabetically
        activation_cols = sorted([c for c in cols if "_pre_activation" in c or "_error" in c])
        ordered_cols.extend(activation_cols)

        # Add any other unexpected columns at the end
        remaining_cols = sorted([c for c in cols if c not in ordered_cols])
        ordered_cols.extend(remaining_cols)


        # Select columns in the preferred order, handle potential errors
        try:
            # Ensure no duplicate columns before selecting
            if len(set(ordered_cols)) != len(ordered_cols):
                 warnings.warn("Duplicate column names detected during reordering. Using default order.", UserWarning)
                 return results_df
            return results_df.select(ordered_cols)
        except (polars.ColumnNotFoundError, polars.exceptions.DuplicateError, ValueError) as e: # Catch potential errors
            warnings.warn(
                f"Could not reorder/select columns ({e}). Returning DataFrame with default order.",
                UserWarning,
            )
            return results_df