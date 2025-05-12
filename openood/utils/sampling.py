from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

def stratified_subset_dataloader(dataloader: DataLoader, n_per_class: int, random_state: int = 42) -> DataLoader:
    """
    Creates a stratified subset of a DataLoader with a fixed number of instances per class.

    Args:
        dataloader (DataLoader): The original PyTorch DataLoader.
        n_per_class (int): Number of instances to retain per class.
        random_state (int): Seed for reproducibility.

    Returns:
        DataLoader: A new DataLoader with the stratified subset.
    """
    dataset = dataloader.dataset

    # Extract labels (assumes dataset[i] returns (input, label))
    labels = np.array([dataset[i]['label'] for i in range(len(dataset))])
    indices = np.arange(len(labels))
    unique_classes, class_counts = np.unique(labels, return_counts=True)

    if np.any(class_counts < n_per_class):
        raise ValueError(f"Not all classes have at least {n_per_class} samples.")

    # Stratified split
    total_samples = len(unique_classes) * n_per_class
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=total_samples, random_state=random_state)
    subset_indices, _ = next(splitter.split(indices, labels))

    # Subset dataset and recreate dataloader
    subset_dataset = Subset(dataset, subset_indices)
    subset_loader = DataLoader(
        subset_dataset,
        batch_size=dataloader.batch_size,
        shuffle=True,  # Reshuffle in subset
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
    )
    return subset_loader
