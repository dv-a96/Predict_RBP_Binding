'''
Module to preprocess rbp and rna sequences, and rbp-rna binding intensities.
'''
import pandas as pd
import re
import numpy as np
from scipy import stats
import warnings
from logger_utils import create_logger

# NOTE: 1. remove bad indexes from intensities.
# NOTE: Add processing description:
""" 1. one hot coding: lengths and padding - zero, uniform"""

def rna_one_hot(rna_df, max_length=41, pad_value=0):
    bases = ['A', 'C', 'G', 'U']
    vector_length = 20  # padding length
    base_to_vec = {}

    for i, base in enumerate(bases):
        vec = np.full(vector_length, pad_value)
        vec[i] = 1  # one-hot position for A/C/G/U at index 0–3
        base_to_vec[base] = vec

    encoded_rnas = []

    for rna in rna_df[0]:
        one_hot = [base_to_vec.get(base, np.full(vector_length, pad_value)) for base in rna]
        if len(one_hot) < max_length:
            one_hot += [np.full(vector_length, pad_value)] * (max_length - len(one_hot))
        else:
            one_hot = one_hot[:max_length]
        encoded_rnas.append(np.array(one_hot))

    return np.array(encoded_rnas).transpose(0, 1, 2).astype(np.int8)



def rbp_one_hot(protein_df, max_length=1000, pad_value=0):
# 20 standard amino acids
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    aa_to_vec = {aa: np.eye(len(amino_acids))[i] for i, aa in enumerate(amino_acids)}
    encoded_proteins = []

    for protein in protein_df[0]:
        # Convert each amino acid to one-hot vector, unknowns get pad_value
        one_hot = [aa_to_vec.get(aa, np.full(len(amino_acids), pad_value)) for aa in protein]
        # Pad or truncate to max_length
        if len(one_hot) < max_length:
            one_hot += [np.full(len(amino_acids), pad_value)] * (max_length - len(one_hot))
        else:
            one_hot = one_hot[:max_length]
        encoded_proteins.append(np.array(one_hot))

    return np.array(encoded_proteins).transpose(0, 1, 2).astype(np.int8)


def convert_txt_to_fast(input_file):
    """Convert txt file into fasta format.
    Assume the txt file is in Seq\nSeq\n format.
    Assign a random identifier to each sequence.    

    Args:
        input_file (path): to a .txt file
    
    Output:

    """
    output_file = input_file.replace('.txt','.fa')
    with open(input_file) as f_in, open(output_file, "w") as f_out:
        for i, line in enumerate(f_in):
            f_out.write(f">seq{i+1}\n{line.strip()}\n")

def validate_rna_sequences(df, min_length = 0, max_length= 1e5, logger =None):
    """
    Validates RNA sequences for correct nucleotide content and length range.
    Assumes the DataFrame has only one column and converts sequences to uppercase.

    Args:
        df (pd.DataFrame): DataFrame with a single column of RNA sequences.
        min_length (int): Minimum allowed sequence length (inclusive).
        max_length (int): Maximum allowed sequence length (inclusive).

    Returns:
        rna_mask (pd.Series): True if sequence contains only A, C, G, U.
        bad_indexes (nd.array): indexes if sequence dont have RNA letters.
    """
    # Get the single column name
    col = df.columns[0]

    # Convert all sequences to uppercase
    df[col] = df[col].astype(str).str.upper()

    # Define RNA pattern
    rna_pattern = re.compile(r'^[ACGU]+$')

    rna_mask = df[col].apply(lambda seq: bool(rna_pattern.fullmatch(seq)))
    length_mask = df[col].apply(lambda seq: min_length <= len(seq) <= max_length)
    # Logging invalid entries
    if logger is None:
        logger = create_logger('trail')
    invalid_rna = df[~rna_mask]
    invalid_length = df[~length_mask]
    bad_indexes = None
    rnas = df[rna_mask]
    if not invalid_rna.empty:
        logger.warning(f"{len(invalid_rna)} sequences have invalid RNA characters we removed them from the Data.")
        logger.debug(f"Invalid RNA sequences: {invalid_rna[col].tolist()[:5]}")  # preview first 5
        logger.debug(f"Indices: {invalid_rna.index}")
        
        bad_indexes = invalid_rna.index
    if not invalid_length.empty:
        logger.warning(f"{len(invalid_length)} sequences are out of length bounds. By defualt they are kept!")
        logger.debug(f"Invalid length sequences: {invalid_length[col].tolist()[:5]}")  # preview
        logger.debug(f"Indices: {invalid_length.index}")
    return rnas, bad_indexes

def validate_rbps_sequences(rbps_data, logger = None):
    bad_indexes = None
    return rbps_data,bad_indexes

def validate_intensities_values(intensities_df, logger = None):
    
    return intensities_df

def preprocess_intensities(intensities_df, logger=None):
    # negatives to zero?
    # Standarization/ Log transformation....
    return intensities_df
    
def prepare_training_data(rna_sequences = 'Data_sets/training_seqs.txt', rbps_sequences = 'Data_sets/training_RBPs2.txt',
                          rbps_rnas_binding_intensities = 'Data_sets/training_data2.txt.gz', logger=None):
    """Prepare training/testing data for the model.

    Args:
        rna_sequences (str, optional): Path to the RNA sequences file. Defaults to 'Data_sets/training_seqs.txt'.
        rbps_sequences (str, optional): Path to the RBPs sequences file. Defaults to 'Data_sets/training_RBPs2.txt'.
        rbps_rnas_binding_intensities (str, optional): Path to the RBPs-RNAs binding intensities file. Defaults to 'Data_sets/training_data2.txt.gz'.
        logger (_type_, optional): Logger instance for logging. Defaults to None.

    Returns:
        Tuple: (rnas, rbps, intensities) where:
            rnas (pd.DataFrame): DataFrame of validated RNA sequences.
            rbps (pd.DataFrame): DataFrame of validated RBPs sequences.
            intensities (pd.DataFrame): DataFrame of preprocessed binding intensities.
    """
    intensities = pd.read_csv(rbps_rnas_binding_intensities,sep='\t',header=None)
    rbps = pd.read_csv(rbps_sequences,header=None)
    rnas = pd.read_csv(rna_sequences,header=None)
    rnas, rna_bad_indexes = validate_rna_sequences(rnas,29,41,logger)
    if rna_bad_indexes: # remove them from intensities accordingly
        pass
    rbps, rbps_bad_indexes = validate_rbps_sequences(rbps, logger)
    if rbps_bad_indexes: # remove them from intensities accordingly
        pass
    intensities = preprocess_intensities(intensities, logger)
    #rbps = np.array(rbps)
    #rnas = np.array(rnas)
    intensities = np.array(intensities)
    return rnas,rbps,intensities



def fit_distribution_and_return_params(intensities_df: pd.DataFrame):
    """
    For each protein (column) in the binding intensities DataFrame,
    fit several continuous distributions and return the best-fitting one
    along with its estimated parameters and K-S statistic.

    Parameters:
        intensities_df (pd.DataFrame): columns = proteins, rows = continuous binding intensities

    Returns:
        pd.DataFrame: one row per protein with best distribution name, parameters, and K-S statistic
    """
    distributions = {
        'lognorm': stats.lognorm,
        'gamma': stats.gamma,
        'expon': stats.expon,
        'norm': stats.norm
    }

    results = []

    for protein in intensities_df.columns:
        data = intensities_df[protein].dropna().values
        data = data[np.isfinite(data)]

        if len(data) < 10:
            continue  # skip if too few data points

        best_fit = None
        best_stat = np.inf
        best_params = None

        for dist_name, dist in distributions.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    params = dist.fit(data)
                    ks_stat, _ = stats.kstest(data, dist_name, args=params)
                    if ks_stat < best_stat:
                        best_stat = ks_stat
                        best_fit = dist_name
                        best_params = params
            except Exception:
                continue

        results.append({
            'protein': protein,
            'best_fit_distribution': best_fit,
            'ks_statistic': best_stat,
            'fit_params': best_params
        })

    return pd.DataFrame(results)

# def sample_percentile_by_rbp(intensities: pd.DataFrame, percentile: float = 99):
#     """
#     For each RBP (column) in the RNA x RBP intensity matrix:
#     - Select all RNAs with intensities above the given percentile.
#     - Randomly sample an equal number of RNAs from the rest (≤ percentile).
#     - Return:
#         - A reduced intensity matrix with all selected RNAs (union of both sets).
#         - Dicts of indices per RBP for both groups.

#     Parameters:
#     ----------
#     intensities : pd.DataFrame
#         RNA (rows) × RBP (columns) binding intensity matrix.
#     percentile : float
#         Percentile threshold (default is 95).

#     Returns:
#     -------
#     sampled_matrix : pd.DataFrame
#         Reduced matrix with only selected RNAs.
#     selected_rnas
#     """
#     if isinstance(intensities,np.ndarray):
#         intensities=pd.DataFrame(intensities)
#         selected_rna_indices = set()

#     for rbp in intensities.columns:
#         col = intensities[rbp]
#         threshold = np.percentile(col, percentile)
#         above = col[col > threshold]
#         below = col[col <= threshold]

#         n_above = len(above)
#         if n_above == 0 or len(below) < n_above:
#             continue
#         sampled_below = np.random.choice(below.index, size=n_above, replace=False)
#         selected_rna_indices.update(above.index)
#         selected_rna_indices.update(sampled_below)

#     selected_rnas = list(selected_rna_indices)
#     sampled_matrix = intensities.loc[selected_rnas]
#     sampled_matrix = sampled_matrix.loc[intensities.index.intersection(sampled_matrix.index)]

#     return sampled_matrix, selected_rnas


def sample_global_rowwise_by_percentile(intensities: np.ndarray, percentile: float = 95, min_fraction: float = 0.5):
    """
    Sample RNA rows based on the fraction of values above a global percentile.

    Parameters:
    ----------
    intensities : np.ndarray
        2D array of shape (n_rnas, n_rbps)
    percentile : float
        Global threshold percentile (default: 95)
    min_fraction : float
        Minimum fraction of values in a row that must be above threshold (default: 0.5)

    Returns:
    -------
    selected_indices : np.ndarray
        Combined RNA row indices (above + sampled below)
    above_indices : np.ndarray
        RNA indices where >= min_fraction of values > threshold
    below_sampled_indices : np.ndarray
        Randomly sampled RNA indices from remaining rows
    reduced_matrix : np.ndarray
        Subset of input matrix with selected rows only
    """
    
    n_rnas, n_rbps = intensities.shape
    threshold = np.percentile(intensities, percentile)

    # Count how many values in each row are > threshold
    row_above_counts = (intensities > threshold).sum(axis=1)
    required_count = int(np.ceil(min_fraction * n_rbps))

    above_mask = row_above_counts >= required_count
    below_mask = ~above_mask

    above_indices = np.where(above_mask)[0]
    below_indices = np.where(below_mask)[0]

    n_above = len(above_indices)
    if n_above == 0 or len(below_indices) < n_above:
        raise ValueError("Not enough rows satisfying condition or not enough to sample from.")

    below_sampled_indices = np.random.choice(below_indices, size=n_above, replace=False)

    # Combine and return
    selected_indices = np.sort(np.concatenate([above_indices, below_sampled_indices]))
    reduced_matrix = intensities[selected_indices, :]

    return selected_indices, reduced_matrix

def process_for_cnn(rbps, rnas, intensities):
    """Process the rbps and rnas sequences to onehot encodings, and sample data using the internal
    sample_global_rowwise_by_percentile function. Sample intenseties over certain precentile.

    Args:
        rbps (pd.Series): protein seqeunces.
        rnas (pd.Series): rna sequences.
        intensities (nd.array): intensities matrix.

    Returns:
        _type_: _description_
    """
    selected_indices, intensities  = sample_global_rowwise_by_percentile(intensities,min_fraction=0.1)
    rbps = rbp_one_hot(rbps)
    rnas = rnas.iloc[selected_indices]
    rnas = rna_one_hot(rnas)
    return rbps, rnas, intensities
    