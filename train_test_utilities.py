"""This module is to split the data into training, validating, and testing sets."""

import numpy as np
from sklearn.model_selection import KFold

def split_k_fold(samples_num, k, excluded_indices=None, random_state=None):
    """
    Splits the indices of samples into k train/test folds, with optional exclusion of specified indices.
    
    Args:
        samples_num (int): Total number of samples.
        k (int): Number of folds.
        excluded_indices (list, optional): List of indices to exclude from the split. Defaults to None.
        random_state (int, optional): Random state for reproducibility. Defaults to None.
        
    Returns:
        train_folds (list of np.ndarray): List of training indices for each fold.
        test_folds (list of np.ndarray): List of test indices for each fold.
    """
    all_indices = np.arange(samples_num)
    
    # Handle excluded indices
    if excluded_indices is not None:
        excluded_indices = np.array(excluded_indices)
        if np.any((excluded_indices < 0) | (excluded_indices >= samples_num)):
            raise ValueError("Some excluded indices are out of bounds.")
        mask = np.ones(samples_num, dtype=bool)
        mask[excluded_indices] = False
        valid_indices = all_indices[mask]
    else:
        valid_indices = all_indices
    
    # Shuffle valid indices before splitting
    np.random.seed(random_state)
    np.random.shuffle(valid_indices)

    # K-Fold splitting
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    train_folds = []
    test_folds = []

    for train_idx, test_idx in kf.split(valid_indices):
        train_folds.append(valid_indices[train_idx])
        test_folds.append(valid_indices[test_idx])

    return train_folds, test_folds

def exclude_indices(samples_num, exclude_num, random_state=None):
    """
    Randomly selects `exclude_num` indices from `samples_num` to exclude.

    Args:
        samples_num (int): Total number of samples.
        exclude_num (int): Number of indices to exclude.
        random_state (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        excluded_indices (list): Sorted list of randomly selected indices to exclude.
    """
    if exclude_num > samples_num:
        raise ValueError("exclude_num cannot be greater than samples_num.")
    rng = np.random.default_rng(random_state)
    excluded = rng.choice(samples_num, size=exclude_num, replace=False)
    return sorted(excluded)

