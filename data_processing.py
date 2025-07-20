'''
Module to preprocess rbp and rna sequences, and rbp-rna binding intensities.
'''
import pandas as pd
import re
import numpy as np
from logger_utils import create_logger

rnas = 'Data_sets/training_seqs.txt'
rbps = 'Data_sets/training_RBPs2.txt'
rna_df = pd.read_csv(rnas, header=None)
rbps_df = pd.read_csv(rbps, header=None)


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

    return np.array(encoded_rnas).transpose(0, 2, 1)



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

    return np.array(encoded_proteins).transpose(0, 2, 1)


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
    rnas = rnas[rna_mask]
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
    rnas, rna_bad_indexes = validate_rna_sequences(rnas,29,43,logger)
    if rna_bad_indexes: # remove them from intensities accordingly
        pass
    rbps, rbps_bad_indexes = validate_rbps_sequences(rbps, logger)
    if rbps_bad_indexes: # remove them from intensities accordingly
        pass
    intensities = preprocess_intensities(intensities, logger)
    rbps = np.array(rbps)
    rnas = np.array(rnas)
    intensities = np.array(intensities)
    return rnas,rbps,intensities
