"""This module is to split the data into training, validating, and testing sets."""

import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd


def baseline_test():
    """
    Remove the last 10 RBPs and last 20678 RNAs for testing."""
    rbps_test = [i for i in range(190,200)]
    rnas_duplicates = pd.read_csv("Data_sets/rna_duplicate_indexes.csv")

def get_clusteres_indices(cluster_file = 'Data_sets/wass_no_dup_cluster.csv', cluster_id = None):
    """
    Reads a CSV file containing cluster information and returns the indices of samples belonging to a specified cluster.

    Args:
        cluster_file (str): Path to the CSV file containing cluster data.
        cluster_id (int, optional): The ID of the cluster to filter by. If None, all indices are returned.

    Returns:
        np.ndarray: Array of indices corresponding to the specified cluster.
    """
    import pandas as pd
    clusters_df = pd.read_csv(cluster_file,index_col=0)
    clusters_df.drop(columns=['Unnamed: 0'], inplace=True)
    if cluster_id is not None:
        if cluster_id == 'all': # randomly pick 20% for validation, 10% for testing from each cluster
            grouped = clusters_df.groupby('0')
            return {key: np.array(group.index) for key, group in grouped}
            
        elif cluster_id not in clusters_df['0'].values:
            raise ValueError(f"Cluster ID {cluster_id} not found in the data.")
        filtered_df = clusters_df[clusters_df['0'] == cluster_id]
    else:
        filtered_df = clusters_df
    return np.array(filtered_df.index)

def split_rbs_to_train_val_test(rbps_indices, val_ratio=0.2, test_ratio=0.1, random_state=42 ):
    """
    Splits the given indices into training, validation, and test sets based on specified ratios.

    Args:
        rbps_indices (np.ndarray): Array of indices to be split.
        val_ratio (float): Proportion of data to be used for validation.
        test_ratio (float): Proportion of data to be used for testing.
        random_state (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        train_indices (np.ndarray): Indices for the training set.
        val_indices (np.ndarray): Indices for the validation set.
        test_indices (np.ndarray): Indices for the test set.
    """
    def split_to_sets(rbps_indices):
        if not (0 <= val_ratio < 1) or not (0 <= test_ratio < 1) or (val_ratio + test_ratio >= 1):
            raise ValueError("val_ratio and test_ratio must be in [0, 1) and their sum must be less than 1.")
        np.random.seed(random_state)
        rng = np.random.default_rng(random_state)
        
        total_samples = len(rbps_indices)
        n_val = max(2, int(val_ratio * total_samples)) if val_ratio > 0 else 0
        n_test = max(1, int(test_ratio * total_samples)) if test_ratio > 0 else 0
        val_indices = np.random.choice(rbps_indices, size=n_val, replace=False)
        remaining = np.setdiff1d(rbps_indices, val_indices)
        test_indices = np.random.choice(remaining, size=n_test, replace=False)
        train_indices = np.setdiff1d(remaining, test_indices)
        if len(train_indices)+len(val_indices)+len(test_indices) != total_samples:
            raise ValueError("The sum of train, validation, and test indices does not equal the total number of samples.")
        
        return train_indices, val_indices, test_indices
    train_indices = []
    validation_indices = []
    test_indices = []
    if isinstance(rbps_indices, dict):
        for cluster,indices in rbps_indices.items():
            tr, val, te = split_to_sets(indices)
            train_indices.extend(tr)
            validation_indices.extend(val)
            test_indices.extend(te)
        return np.array(train_indices), np.array(validation_indices), np.array(test_indices)
    else:
        return split_to_sets(rbps_indices)
    



def stratified_split_multi(
    X_MxN, test_size=0.10, val_size=0.20, n_bins=50, random_state=42, method="rank-mean"
):
    """
    X_MxN: array-like of shape (M, N) with M vectors (features) and N = 100k samples.
    Returns: train_idx, val_idx, test_idx shared for all M vectors.
    """
    X = np.asarray(X_MxN)
    M, N = X.shape
    # 1) Composite score per sample
    if method == "rank-mean":
        # compute rank (1..N) per vector; convert to uniform [0,1] by /N; average across M
        ranks = np.apply_along_axis(lambda v: pd.Series(v).rank(method="average").values, 1, X)
        score = ranks.mean(axis=0) / N
    else:
        raise ValueError("Unsupported method")
    # 2) Bin the composite score for stratification
    bins = pd.qcut(pd.Series(score), q=n_bins, duplicates="drop")
    # 3) Test split
    idx_all = np.arange(N)
    if test_size > 0:
        # 1) Hold-out test
        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        trainval_mask, test_mask = next(sss_test.split(idx_all, bins))
        trainval_idx = idx_all[trainval_mask]
        test_idx     = idx_all[test_mask]

        # 2) From remaining, split val so that overall val fraction is val_size
        rel_val = val_size / (1 - test_size)  # fraction of the remainder
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=rel_val, random_state=random_state)
        bins_trainval = bins.iloc[trainval_idx]
        train_mask, val_mask = next(sss_val.split(trainval_idx, bins_trainval))

        train_idx = trainval_idx[train_mask]
        val_idx   = trainval_idx[val_mask]
        return train_idx, val_idx, test_idx
    else:
        # Only train/val
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
        train_mask, val_mask = next(sss_val.split(idx_all, bins))
        train_idx = idx_all[train_mask]
        val_idx   = idx_all[val_mask]
        return train_idx, val_idx, None
    


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


class PairDatasetFactory:
    """
    Holds TF tensors for rbps/rnas/intensities and produces tf.data datasets
    without duplicating memory. Convert to TF once, reuse many times.
    """
    def __init__(self, rbps_np, rnas_np, intensities_np=None, place_on_cpu=True, 
                 sample_weight_array = None):
        self.y = None
        self.use_weights = False
        if sample_weight_array is not None:
            if sample_weight_array.shape != intensities_np.shape:
                raise ValueError(f"sample_w_np shape {sample_weight_array.shape} must match intensities {intensities_np.shape}")
            self.use_weights = True
        # Optionally pin on CPU so you don't waste GPU VRAM on constants
        device = "/CPU:0" if place_on_cpu else None
        if device:
            with tf.device(device):
                self.rbps = tf.convert_to_tensor(rbps_np, dtype=tf.float32)        # [P, Dp]
                self.rnas = tf.convert_to_tensor(rnas_np, dtype=tf.int8)        # [R, L, A]
                if intensities_np is not None:
                    self.y    = tf.convert_to_tensor(intensities_np, dtype=tf.float32) # [R, P]
                if self.use_weights:
                
                    self.w = tf.convert_to_tensor(sample_weight_array, dtype=tf.float32)    # [R, P]
                else:
                    self.w = None
        else:
            self.rbps = tf.convert_to_tensor(rbps_np, dtype=tf.float32)
            self.rnas = tf.convert_to_tensor(rnas_np, dtype=tf.int8)
            if intensities_np is not None:
                self.y    = tf.convert_to_tensor(intensities_np, dtype=tf.float32)
            if self.use_weights:
                self.w = tf.convert_to_tensor(sample_weight_array, dtype=tf.float32)    # [R, P]
            else:
                self.w = None
            

        self.P = tf.shape(self.rbps)[0]
        self.R = tf.shape(self.rnas)[0]
    # returns x only
    def _map_pair_x_only(self, p, r, return_ids: bool):
        rbp_vec = tf.gather(self.rbps, p)
        rna_oh  = tf.gather(self.rnas, r)
        if return_ids:
            return (rbp_vec, rna_oh), (p, r)

        return (rbp_vec, rna_oh),0  
        
    # returns (x, y)
    def _map_pair_xy(self, p, r, return_ids: bool):
        rbp_vec = tf.gather(self.rbps, p)
        rna_oh  = tf.gather(self.rnas, r)
        y_val   = tf.gather(tf.gather(self.y, r), p)  # scalar
        features = (rbp_vec, rna_oh)
        label    = tf.expand_dims(y_val, -1)
        if return_ids:
            return (features, label), (p, r)
        return features, label

    # returns (x, y, w)
    def _map_pair_xyw(self, p, r, return_ids: bool):
        rbp_vec = tf.gather(self.rbps, p)
        rna_oh  = tf.gather(self.rnas, r)
        y_val   = tf.gather(tf.gather(self.y, r), p)
        w_val   = tf.gather(tf.gather(self.w, r), p)
        features = (rbp_vec, rna_oh)
        label    = tf.expand_dims(y_val, -1)
        if return_ids:
            return (features, label, w_val), (p, r)
        return features, label, w_val


    def make_dataset(self,
                     prot_ids=None,
                     rna_ids=None,
                     batch_size=256,
                     shuffle=True,
                     buffer_cap=100_000,
                     num_parallel_calls=tf.data.AUTOTUNE,
                     return_ids=False):
        """
        Build a dataset over the (protein_id, rna_id) Cartesian product lazily.
        """
        if prot_ids is None:
            prot_ids = tf.range(self.P, dtype=tf.int32)
        else:
            prot_ids = tf.convert_to_tensor(prot_ids, dtype=tf.int32)

        if rna_ids is None:
            rna_ids = tf.range(self.R, dtype=tf.int32)
        else:
            rna_ids = tf.convert_to_tensor(rna_ids, dtype=tf.int32)

        p_ds = tf.data.Dataset.from_tensor_slices(prot_ids)
        r_ds = tf.data.Dataset.from_tensor_slices(rna_ids)
        ds   = p_ds.flat_map(lambda p: r_ds.map(lambda r: (p, r)))

        if shuffle:
            est_size = tf.size(prot_ids) * tf.size(rna_ids)
            ds = ds.shuffle(buffer_size=tf.cast(tf.minimum(est_size, buffer_cap), tf.int64))
        if self.use_weights:
            map_fn = lambda p, r: self._map_pair_xyw(p, r, return_ids)
        elif self.y is not None:
            map_fn = lambda p, r: self._map_pair_xy(p, r, return_ids)
        else:
            map_fn = lambda p, r: self._map_pair_x_only(p, r, return_ids)
        ds = ds.map(map_fn,num_parallel_calls=num_parallel_calls)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    # Convenience: training vs debugging versions
    def make_train(self, prot_ids=None, rna_ids=None, **kw):
        # returns ( (rbp_vec, rna_oh), y )
        return self.make_dataset(prot_ids, rna_ids, return_ids=False, **kw)

    def make_debug(self, prot_ids=None, rna_ids=None, **kw):
        # returns ( (rbp_vec, rna_oh), y, (p, r) )
        return self.make_dataset(prot_ids, rna_ids, return_ids=True, **kw)
    

