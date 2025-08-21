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
    # if logger is None:
    #     logger = create_logger('trail')
    invalid_rna = df[~rna_mask]
    invalid_length = df[~length_mask]
    bad_indexes = None
    rnas = df[rna_mask]
    if not invalid_rna.empty:
        # logger.warning(f"{len(invalid_rna)} sequences have invalid RNA characters we removed them from the Data.")
        # logger.debug(f"Invalid RNA sequences: {invalid_rna[col].tolist()[:5]}")  # preview first 5
        # logger.debug(f"Indices: {invalid_rna.index}")
        
        bad_indexes = invalid_rna.index
    if not invalid_length.empty:
        # logger.warning(f"{len(invalid_length)} sequences are out of length bounds. By defualt they are kept!")
        # logger.debug(f"Invalid length sequences: {invalid_length[col].tolist()[:5]}")  # preview
        # logger.debug(f"Indices: {invalid_length.index}")
        pass
    return rnas, bad_indexes

def validate_rbps_sequences(rbps_data, logger = None):
    bad_indexes = None
    return rbps_data,bad_indexes

def validate_intensities_values(intensities_df, logger = None):
    
    return intensities_df

def preprocess_intensities(intensities_df, logger=None , method=None, unit_length=True):
    if method.lower() == 'quantile':
        intensities_df = quantile_normalize(intensities_df)
    elif method.lower() =='log':
        intensities_df = log_normalization(intensities_df,logger)
    else: method =''
    #logger.info(f"Normalization method: {method}")
    if unit_length:
        intensities_df = unit_scale(intensities_df)
    #logger.info(f"Unit scaling: {unit_length}")
    # negatives to zero?
    # Standarization/ Log transformation....
    return intensities_df

def clamp_by_precentile(df,precentile = 99.5):
    cutoffs = df.apply(lambda col: np.percentile(col, precentile)).to_numpy()
    df_clamped = df.apply(lambda col: np.minimum(col, np.percentile(col, precentile)))
    max_after = df_clamped.max().to_numpy()
    print(cutoffs,max_after)
    if np.array_equal(cutoffs,max_after):
        print('prenctile clamp worked')
    else: print('prenctile clamp did not work')
    return df_clamped


def quantile_normalize(df):
    """
    Quantile normalize a Pandas DataFrame (m rows, n cols).
    Each column will end up with the same distribution.
    """
    # Sort each column
    sorted_df = pd.DataFrame(
        np.sort(df.values, axis=0),
        index=df.index,
        columns=df.columns
    )
    # Mean across columns for each row (rank)
    rank_means = sorted_df.mean(axis=1).values
    # Map the sorted means back to the original data's ranks
    rank_dict = {rank: mean for rank, mean in enumerate(rank_means, start=1)}
    normalized_df = df.rank(method='min').stack().astype(int).map(rank_dict).unstack()
    return normalized_df

def log_normalization(df,logger):

    min_val = df.min().min()
    if min_val <0:
        #logger.warning(f"doing log transformation on negative values: doing values - values.min() before")
        df = df - min_val 
    df = np.log1p(df)
    return df


def unit_scale(df):
    X_normalized = df / np.linalg.norm(df, axis=0, keepdims=True)
    return X_normalized

def prepare_training_data(rna_sequences = 'Data_sets/training_seqs.txt', rbps_sequences = 'Data_sets/training_RBPs2.txt',
                          rbps_rnas_binding_intensities = 'Data_sets/training_data2.txt.gz', logger=None ,
                          normalization_method = None):
    """Prepare training/testing data for the model.

    Args:
        rna_sequences (str, optional): Path to the RNA sequences file. Defaults to 'Data_sets/training_seqs.txt'.
        rbps_sequences (str, optional): Path to the RBPs sequences file. Defaults to 'Data_sets/training_RBPs2.txt'.
        rbps_rnas_binding_intensities (str, optional): Path to the RBPs-RNAs binding intensities file. Defaults to 'Data_sets/training_data2.txt.gz'.
        logger (_type_, optional): Logger instance for logging. Defaults to None.
        normalization_method (str): How to normalize the intensities - log, quantile
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
    intensities = preprocess_intensities(intensities, logger,method=normalization_method,unit_length=False)
    #rbps = np.array(rbps)
    #rnas = np.array(rnas)
    intensities = np.array(intensities)
    return rnas,rbps,intensities

def get_ESM_prot_vecs(esm_vectors = '/home/dsi/lubosha/Predict_RBP_Binding/ESM/all_proteins_emb_vectors.csv.gz'):
    esm_vecs = pd.read_csv(esm_vectors)
    esm_vecs['protein_id']=esm_vecs['protein_id'].apply(lambda x: x[3:]).astype(int) - 1
    esm_vecs.set_index('protein_id',inplace=True)
    esm_vecs.sort_index(ascending=True,inplace=True)
    return np.array(esm_vecs)

def get_ESM_rna_vecs(esm_vectors = 'ESM/rnas_esm_embeddings.csv.gz'):
    # files = [os.path.join(esm_vectors,file) for file in os.listdir(esm_vectors) if 'rna' in file]
    # files = [pd.read_csv(file) for file in files]
    # all_rnas = pd.concat(files)
    # all_rnas['protein_id']=all_rnas['protein_id'].apply(lambda x: x[3:]).astype(int) - 1
    # all_rnas.set_index('protein_id',inplace=True)
    # all_rnas.sort_index(ascending=True,inplace=True)
    all_rnas = pd.read_csv(esm_vectors,index_col='protein_id')
    return np.array(all_rnas)
    


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



def sample_global_rowwise_by_percentile(intensities: np.ndarray, percentile: float = 95, min_fraction: float = 0.5,
                                        get_all=False):
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
    if get_all:
        return range(len(intensities)), intensities
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

# def process_for_cnn(rbps, rnas, intensities, get_all=False):
#     """Process the rbps and rnas sequences to onehot encodings, and sample data using the internal
#     sample_global_rowwise_by_percentile function. Sample intenseties over certain precentile.

#     Args:
#         rbps (pd.Series): protein seqeunces.
#         rnas (pd.Series): rna sequences.
#         intensities (nd.array): intensities matrix.

#     Returns:
#         _type_: _description_
#     """
#     #selected_indices, intensities  = sample_global_rowwise_by_percentile(intensities,min_fraction=0.1,get_all=get_all)
#     rbps = rbp_one_hot(rbps)
#     #rnas = rnas.iloc[selected_indices]sample_global_rowwise_by_percentile
#     rnas = rna_one_hot(rnas)
#     return rbps, rnas, intensities


def create_similar_matrix_from_mmseq2(similarity_output = 'Data_sets/similarity_train.tsv',col_value= 'pident'):
    ids_sorted = [f'seq{i+1}'for i in range(200)]
    cols = ["query","target","pident","alnlen","qlen","tlen","qcov","tcov","evalue","bits"]
    df = pd.read_csv(similarity_output, sep="\t", names=cols)
    
    mat = df.pivot(index="query", columns="target", values=col_value)
    mat = mat.reindex(index=ids_sorted, columns=ids_sorted)

    mat = mat.fillna(0.0)
    mat.to_csv(f"{col_value}.csv")

def get_simliary_dict(similarity_score='Data_sets/pident.csv', treshold=  80):
    mat = pd.read_csv(similarity_score)
    s = mat.where(mat > treshold).stack()
    similariteis = {index:s.loc[index].to_dict() for index in mat.index}
    return similariteis