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


# class RBP_RNA_Combined_Dataset(tf.data.Dataset):
#     def __new__(cls, rbps, rnas, intensities=None):
#         n_rbps = rbps.shape[0]
#         n_rnas = rnas.shape[0]
#         L_rbp = rbps.shape[1]
#         L_rna = rnas.shape[1]
#         C = rbps.shape[2]

#         def generator():
#             for i in range(n_rbps):
#                 for j in range(n_rnas):
#                     rbp = rbps[i]        # (L_rbp, C)
#                     rna = rnas[j]        # (L_rna, C)
#                     pair = tf.concat([rbp, rna], axis=0)  # (L_rbp + L_rna, C)

#                     if intensities is not None:
#                         label = intensities[j, i]  # RNA j, RBP i
#                     else:
#                         label = 0.0  # or tf.constant(0.0)

#                     yield pair, label

#         output_signature = (
#     tf.TensorSpec(shape=(L_rbp + L_rna, C), dtype=tf.int8),
#     tf.TensorSpec(shape=(), dtype=tf.float32)  # scalar label
# )

#         return tf.data.Dataset.from_generator(generator, output_signature=output_signature)
# class RBP_RNA_separate_Dataset(tf.data.Dataset):
#     def __new__(cls, rbps, rnas, intensities=None,if_rbp_coding = True):
#         n_rbps = rbps.shape[0]
#         n_rnas = rnas.shape[0]
#         L_rbp = rbps.shape[1]
#         L_rna = rnas.shape[1]
#         if if_rbp_coding:
#             C_rbp = rbps.shape[2]
#             rbp_tensor = tf.TensorSpec(shape=(L_rbp, C_rbp), dtype=tf.int8)
#         else :  rbp_tensor = tf.TensorSpec(shape=(L_rbp, ), dtype=tf.float32)
               
#         C_rna = rnas.shape[2]
#         output_signature = (
#             (
                
#                 rbp_tensor,  # RBP branch
#                 tf.TensorSpec(shape=(L_rna, C_rna), dtype=tf.int8)   # RNA branch
#             ),
#             tf.TensorSpec(shape=(), dtype=tf.float32)  # scalar label
#         )
#         def generator():
#             for i in range(n_rbps):
#                 for j in range(n_rnas):
#                     rbp = rbps[i]        # shape: (L_rbp, C_rbp)
#                     rna = rnas[j]        # shape: (L_rna, C_rna)

#                     if intensities is not None:
#                         label = intensities[j, i]  # RNA j, RBP i
#                     else:
#                         label = 0.0  # or tf.constant(0.0)

#                     # Yield them as TWO separate tensors
#                     yield (rbp, rna), label

        

#         return tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    
# class RBP_RNA_ConcatDataset(tf.data.Dataset):
#     def __new__(cls, rbps, rnas, intensities=None):
#         """
#         rbps: (M, rbp_bits)
#         rnas: (N, rna_bits)
#         intensities: (N, M) labels matrix (optional)
#         """
#         n_rbps = rbps.shape[0]
#         n_rnas = rnas.shape[0]
#         rbp_bits = rbps.shape[1]
#         rna_bits = rnas.shape[1]

#         def generator():
#             for i in range(n_rbps):
#                 for j in range(n_rnas):
#                     rbp = rbps[i]        # shape: (rbp_bits,)
#                     rna = rnas[j]        # shape: (rna_bits,)

#                     # concatenate protein & RNA into one long vector
#                     pair = tf.concat([rbp, rna], axis=0)  # shape: (rbp_bits + rna_bits,)

#                     if intensities is not None:
#                         label = intensities[j, i]  # RNA j, RBP i
#                     else:
#                         label = 0.0  # or tf.constant(0.0)

#                     yield pair, label

#         output_signature = (
#             tf.TensorSpec(shape=(rbp_bits + rna_bits,), dtype=tf.int8),
#             tf.TensorSpec(shape=(), dtype=tf.float32)  # scalar label
#         )

#         return tf.data.Dataset.from_generator(generator, output_signature=output_signature)



class PairDatasetFactory:
    """
    Holds TF tensors for rbps/rnas/intensities and produces tf.data datasets
    without duplicating memory. Convert to TF once, reuse many times.
    """
    def __init__(self, rbps_np, rnas_np, intensities_np, place_on_cpu=True):
        # Optionally pin on CPU so you don't waste GPU VRAM on constants
        device = "/CPU:0" if place_on_cpu else None
        if device:
            with tf.device(device):
                self.rbps = tf.convert_to_tensor(rbps_np, dtype=tf.float32)        # [P, Dp]
                self.rnas = tf.convert_to_tensor(rnas_np, dtype=tf.int8)        # [R, L, A]
                self.y    = tf.convert_to_tensor(intensities_np, dtype=tf.float32) # [R, P]
        else:
            self.rbps = tf.convert_to_tensor(rbps_np, dtype=tf.float32)
            self.rnas = tf.convert_to_tensor(rnas_np, dtype=tf.int8)
            self.y    = tf.convert_to_tensor(intensities_np, dtype=tf.float32)

        self.P = tf.shape(self.rbps)[0]
        self.R = tf.shape(self.rnas)[0]

    def _map_pair(self, p, r, return_ids: bool):
        rbp_vec = tf.gather(self.rbps, p)  # [Dp]
        rna_oh  = tf.gather(self.rnas, r)  # [L, A]
        y       = tf.gather(tf.gather(self.y, r), p)  # scalar
        features = (rbp_vec, rna_oh)
        label    = tf.expand_dims(y, -1)

        if return_ids:
            # Extra debugging info alongside batch
            return features, label, (p, r)
        return features, label

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

        ds = ds.map(lambda p, r: self._map_pair(p, r, return_ids),
                    num_parallel_calls=num_parallel_calls)
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
    


    # rbps_const = tf.constant(rbps, dtype=tf.float32)          # [P, Dp]
    # rnas_const = tf.constant(rnas, dtype=tf.int8)          # [R, L, A]
    # y_const    = tf.constant(intensities, dtype=tf.float32)   # [R, P]

    # P = rbps_const.shape[0]
    # R = rnas_const.shape[0]

    # # -----------------------------
    # # Dataset over (protein_id, rna_id) pairs
    # # -----------------------------
    # def make_pair_dataset(batch_size=batch_size, shuffle=True,
    #                     protein_ids=None, rna_ids=None,
    #                     buffer_cap=100_000):
    #     """
    #     Streams (p, r) -> ((rbp_vec, rna_onehot), y[r,p])
    #     without materializing repeats.
    #     """
    #     if protein_ids is None:
    #         protein_ids = tf.range(P, dtype=tf.int32)
    #     else:
    #         protein_ids = tf.convert_to_tensor(protein_ids, dtype=tf.int32)

    #     if rna_ids is None:
    #         rna_ids = tf.range(R, dtype=tf.int32)
    #     else:
    #         rna_ids = tf.convert_to_tensor(rna_ids, dtype=tf.int32)

    #     # Cartesian product lazily
    #     p_ds = tf.data.Dataset.from_tensor_slices(protein_ids)
    #     r_ds = tf.data.Dataset.from_tensor_slices(rna_ids)

    #     ds = p_ds.flat_map(lambda p: r_ds.map(lambda r: (p, r)))

    #     if shuffle:
    #         # Large but bounded buffer; tune for your PR size and RAM
    #         ds = ds.shuffle(buffer_size=min(buffer_cap, tf.size(protein_ids)*tf.size(rna_ids)))

    #     def map_to_tensors(p, r):
    #         rbp_vec = tf.gather(rbps_const, p)     # [Dp]
    #         rna_oh  = tf.gather(rnas_const, r)     # [L, A]
    #         y = tf.gather(tf.gather(y_const, r), p)  # scalar
            
    #         return ({"Protein_representation": rbp_vec, "RNA_Protein_Matrix": rna_oh}, tf.expand_dims(y, axis=-1))

    #     ds = ds.map(map_to_tensors, num_parallel_calls=tf.data.AUTOTUNE)
    #     ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    #     return ds