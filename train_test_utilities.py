"""This module is to split the data into training, validating, and testing sets."""

import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf

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


class RBP_RNA_Combined_Dataset(tf.data.Dataset):
    def __new__(cls, rbps, rnas, intensities=None):
        n_rbps = rbps.shape[0]
        n_rnas = rnas.shape[0]
        L_rbp = rbps.shape[1]
        L_rna = rnas.shape[1]
        C = rbps.shape[2]

        def generator():
            for i in range(n_rbps):
                for j in range(n_rnas):
                    rbp = rbps[i]        # (L_rbp, C)
                    rna = rnas[j]        # (L_rna, C)
                    pair = tf.concat([rbp, rna], axis=0)  # (L_rbp + L_rna, C)

                    if intensities is not None:
                        label = intensities[j, i]  # RNA j, RBP i
                    else:
                        label = 0.0  # or tf.constant(0.0)

                    yield pair, label

        output_signature = (
    tf.TensorSpec(shape=(L_rbp + L_rna, C), dtype=tf.int8),
    tf.TensorSpec(shape=(), dtype=tf.float32)  # scalar label
)

        return tf.data.Dataset.from_generator(generator, output_signature=output_signature)
class RBP_RNA_separate_Dataset(tf.data.Dataset):
    def __new__(cls, rbps, rnas, intensities=None):
        n_rbps = rbps.shape[0]
        n_rnas = rnas.shape[0]
        L_rbp = rbps.shape[1]
        L_rna = rnas.shape[1]
        C_rbp = rbps.shape[2]
        C_rna = rnas.shape[2]

        def generator():
            for i in range(n_rbps):
                for j in range(n_rnas):
                    rbp = rbps[i]        # shape: (L_rbp, C_rbp)
                    rna = rnas[j]        # shape: (L_rna, C_rna)

                    if intensities is not None:
                        label = intensities[j, i]  # RNA j, RBP i
                    else:
                        label = 0.0  # or tf.constant(0.0)

                    # Yield them as TWO separate tensors
                    yield (rbp, rna), label

        output_signature = (
            (
                tf.TensorSpec(shape=(L_rbp, C_rbp), dtype=tf.int8),  # RBP branch
                tf.TensorSpec(shape=(L_rna, C_rna), dtype=tf.int8)   # RNA branch
            ),
            tf.TensorSpec(shape=(), dtype=tf.float32)  # scalar label
        )

        return tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    
class RBP_RNA_ConcatDataset(tf.data.Dataset):
    def __new__(cls, rbps, rnas, intensities=None):
        """
        rbps: (M, rbp_bits)
        rnas: (N, rna_bits)
        intensities: (N, M) labels matrix (optional)
        """
        n_rbps = rbps.shape[0]
        n_rnas = rnas.shape[0]
        rbp_bits = rbps.shape[1]
        rna_bits = rnas.shape[1]

        def generator():
            for i in range(n_rbps):
                for j in range(n_rnas):
                    rbp = rbps[i]        # shape: (rbp_bits,)
                    rna = rnas[j]        # shape: (rna_bits,)

                    # concatenate protein & RNA into one long vector
                    pair = tf.concat([rbp, rna], axis=0)  # shape: (rbp_bits + rna_bits,)

                    if intensities is not None:
                        label = intensities[j, i]  # RNA j, RBP i
                    else:
                        label = 0.0  # or tf.constant(0.0)

                    yield pair, label

        output_signature = (
            tf.TensorSpec(shape=(rbp_bits + rna_bits,), dtype=tf.int8),
            tf.TensorSpec(shape=(), dtype=tf.float32)  # scalar label
        )

        return tf.data.Dataset.from_generator(generator, output_signature=output_signature)