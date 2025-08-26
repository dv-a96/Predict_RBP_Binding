'''
Module to preprocess rbp and rna sequences, and rbp-rna binding intensities.
'''
import pandas as pd
import re
import numpy as np
from scipy import stats
import warnings
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer



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
    rna_pattern = re.compile(r'^[ACGTU]+$')

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
    """Preprocess binding intensities DataFrame with various normalization methods.

    Args:
        intensities_df (data frame): rbp x rna intensities
        logger (Logger, optional): log errors. Defaults to None.
        method (str, optional): method to transform. Defaults to None.
        unit_length (bool, optional): scale to unit length. Defaults to True.

    Raises:
        ValueError: normalization method must be a string or None.
        ValueError: box cox requires strictly positive values.
        ValueError: method unknown.

    Returns:
        data frame: normalized/transformed intensities
    """
    if method is None:
        return intensities_df
    elif isinstance(method, str):
        method = method.lower()
        if method not in ['log', 'quantile','minmax', 'zscore', 'robust', 'meannorm', 'yeo-johnson']:
            print(f"Unknown normalization method: {method}\nNo normalization applied!")
            return intensities_df
    else:
        raise ValueError("Normalization method must be a string or None.")
    if 'log' in method:
        intensities_df = log_normalization(intensities_df,logger)
    if 'quantile' in method:
        intensities_df = quantile_normalize(intensities_df)
    
    X = intensities_df.values.astype(float)

    if "minmax" in method:
        intensities_df = pd.DataFrame(
            MinMaxScaler().fit_transform(X),
            columns=intensities_df.columns,
            index=intensities_df.index
        )

    elif "zscore" in method:
        intensities_df = pd.DataFrame(
            StandardScaler().fit_transform(X),
            columns=intensities_df.columns,
            index=intensities_df.index
        )

    elif "robust" in method:
        intensities_df = pd.DataFrame(
            RobustScaler().fit_transform(X),
            columns=intensities_df.columns,
            index=intensities_df.index
        )

    elif "meannorm" in method:
        col_max = intensities_df.max(axis=0)
        col_min = intensities_df.min(axis=0)
        col_mean = intensities_df.mean(axis=0)
        intensities_df = (intensities_df - col_mean) / (col_max - col_min + 1e-9)

    elif "yeo-johnson" in method:
        intensities_df = intensities_df+ 1e-6  # Box-Cox requires positive values
        if (intensities_df <= 0).any().any():
            raise ValueError("Box Cox requires strictly positive values.")
        intensities_df = pd.DataFrame(
            PowerTransformer(method="yeo-johnson").fit_transform(X),
            columns=intensities_df.columns,
            index=intensities_df.index
        )
    elif method == '': pass
    
    if unit_length:
        intensities_df = unit_scale(intensities_df)
    
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


def create_wieghts_per_bin(values, n_bins=10, alpha=1.0, eps=1e-8, normalize_mean_to_1=True):
    """
    split the data to n_bins and assign weights to each bin.
    
    Args:
    values: np.ndarray of any shape 
    n_bins: number of quantile bins
    alpha : 1.0 => weight ∝ width; 0.5 => sqrt(width); >1 strengthens tails
    """
    flat = values.reshape(-1)
    # Quantile edges (length n_bins+1)
    edges = np.quantile(flat, np.linspace(0.0, 1.0, n_bins + 1), method="linear")
    widths = np.maximum(np.diff(edges), eps)  # shape (n_bins,)
    w_per_bin = widths ** alpha
    # Assign each value to a bin index in [0, n_bins-1]
    bin_idx = np.searchsorted(edges[1:-1], flat, side="right")  # shape (N,)
    # Map to per-sample weights
    sample_w = w_per_bin[bin_idx].astype(np.float32)
    if normalize_mean_to_1:
        m = sample_w.mean() if sample_w.size else 1.0
        if m > 0:
            sample_w = sample_w / m
    return sample_w.reshape(values.shape).astype(np.float32), edges.astype(np.float32), w_per_bin.astype(np.float32)



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


def remove_rna_duplicates(rna_df, intensities_df, treshold = 0.8):
    """remove rna duplicates based on pearson correlation of their intensities.
    If pearson correlation is above treshold, replace the duplicates with their mean.
    Saves the indexes of the duplicates to a csv file.
    Args:
        rna_df (data frame): rna sequences
        intensities_df (data frame): intensities values
        treshold (float, optional): treshold for pearson corelation. Defaults to 0.8.

    Returns:
        data frame: intensities with duplicates removed and their mean replaced.
    """
    dup_groups = rna_df.groupby(list(rna_df.columns)).apply(lambda g: g.index.tolist())
    dup_groups = np.array(dup_groups[dup_groups.apply(len) > 1])
    df = intensities_df.copy().T
    pearsons = [intensities_df[idxs].corr().values[np.triu_indices(len(idxs), k=1)].mean() for idxs in dup_groups]
    pearsons = np.array(pearsons)
    bigger_than_08 = dup_groups[pearsons > treshold]
    for cols in bigger_than_08:
        mean_vals = df[cols].mean(axis=1)
        df[cols] = pd.DataFrame({c: mean_vals for c in cols}, index=df.index)
    duplicates = [val[1] for val in bigger_than_08]
    duplicates = pd.DataFrame(duplicates).to_csv("Data_sets/rna_duplicate_indexes.csv", index=False)
    return df.T

def prepare_training_data(rna_sequences = 'Data_sets/training_seqs.txt', rbps_sequences = 'Data_sets/training_RBPs2.txt',
                          rbps_rnas_binding_intensities = 'Data_sets/training_data2_deduped.csv', logger=None ,
                          normalization_method = None, if_clamp_by_percentile = False, percentile = 99.5,
                          if_sample_wieght=True, alpha=0.5, bins=20, if_remove_rna_duplicates = False):
    """Prepare training/testing data for the model.

    Args:
        rna_sequences (str, optional): Path to the RNA sequences file. Defaults to 'Data_sets/training_seqs.txt'.
        rbps_sequences (str, optional): Path to the RBPs sequences file. Defaults to 'Data_sets/training_RBPs2.txt'.
        rbps_rnas_binding_intensities (str, optional): Path to the RBPs-RNAs binding intensities file. Defaults to 'Data_sets/training_data2.txt.gz'.
        logger (_type_, optional): Logger instance for logging. Defaults to None.
        normalization_method (str): How to normalize the intensities - log, quantile, minmax, zscore, robust, meannorm, yeo-johnson. Defaults to None.
        if_clamp_by_percentile (bool, optional): Whether to clamp intensities by a percentile. Defaults to False.
        percentile (float, optional): Percentile for clamping if enabled. Defaults to 99.5.
        if_sample_wieght (bool, optional): Whether to create sample weights. Defaults to True.
        alpha (float, optional): Alpha parameter for sample weights. Defaults to 0.5.
        bins (int, optional): Number of bins for sample weights. Defaults to 20.
        if_remove_rna_duplicates (bool, optional): Whether to remove RNA duplicates. Defaults to False.
    Returns:
        Tuple: (rnas, rbps, intensities) where:
            rnas (pd.DataFrame): DataFrame of validated RNA sequences.
            rbps (pd.DataFrame): DataFrame of validated RBPs sequences.
            intensities (pd.DataFrame): DataFrame of preprocessed binding intensities.
    """
    
    rbps = pd.read_csv(rbps_sequences,header=None)
    rnas = pd.read_csv(rna_sequences,header=None)
    rnas, rna_bad_indexes = validate_rna_sequences(rnas,29,41, logger)
    rbps, rbps_bad_indexes = validate_rbps_sequences(rbps, logger)
    if if_remove_rna_duplicates:
        rbps_rnas_binding_intensities= "Data_sets/training_data2_deduped_rbp_rna.csv"
        rna_bad_indexes_ = pd.read_csv("Data_sets/rna_duplicate_indexes.csv").values.flatten()
        if rna_bad_indexes is not None:
            rna_bad_indexes = np.unique(np.concatenate([rna_bad_indexes, rna_bad_indexes_]))
        else: rna_bad_indexes = rna_bad_indexes_

    intensities = pd.read_csv(rbps_rnas_binding_intensities)
    if rna_bad_indexes is not None: # remove them from intensities accordingly
        intensities = intensities.drop(index=rna_bad_indexes, errors='ignore').reset_index(drop=True)
        rnas = rnas.drop(index=rna_bad_indexes, errors='ignore').reset_index(drop=True)
    if rbps_bad_indexes: # remove them from intensities accordingly
        pass
    if if_clamp_by_percentile:
        intensities = clamp_by_precentile(intensities,precentile=percentile)
    
    
    intensities = preprocess_intensities(intensities, logger,method=normalization_method,unit_length=False)
    sample_w, edges, bin_w = None, None, None
    if if_sample_wieght:
        sample_w, edges, bin_w = create_wieghts_per_bin(intensities.values, n_bins=bins, alpha=alpha)
    #rbps = np.array(rbps)
    #rnas = np.array(rnas)
    intensities = np.array(intensities)
    return rnas,rbps,intensities, sample_w, edges, bin_w


def avg_dup_per_cluster_preserve_order(idx_dup, cluster_file, data_file):
    """
    For each duplicate group:
        - Group duplicates by cluster
        - Average the values of each cluster separately
        - Put the mean in the first column of that cluster
        - Drop the rest of the duplicates in that cluster
    Preserves the original column order of the data.
    """
    # Read cluster mapping [sample, cluster]
    clusters = pd.read_csv(cluster_file, names=["sample", "cluster"])
    cluster_dict = clusters.set_index("sample")["cluster"].to_dict()

    # Read original data
    data = pd.read_csv(data_file, header=None, sep="\t")
    original_cols = list(data.columns)

    # Read duplicates mapping
    dup_dict = csv_to_dict(idx_dup)

    # Make a copy of the original data
    result = data.copy()

    for key, dup_list in dup_dict.items():
        # All IDs in the duplicate group
        all_ids = [key] + dup_list

        # Keep only valid IDs (exist in cluster_dict and data columns)
        # valid_ids = [s for s in all_ids if s in cluster_dict and s in data.columns]
        # if not valid_ids:
        #     continue

        # Group IDs by cluster
        cluster_groups = {}
        for s in all_ids:
            c = cluster_dict[s]
            cluster_groups.setdefault(c, []).append(s)

        # Process each cluster group separately
        for cluster_id, ids_in_cluster in cluster_groups.items():
            if not ids_in_cluster:
                continue

            # Compute mean for this cluster
            cluster_mean = data[ids_in_cluster].mean(axis=1)

            # Assign mean to all samples in cluster group
            for idx in ids_in_cluster:
                result[idx] = cluster_mean

    return result



def csv_to_dict(file_path):
    df = pd.read_csv(file_path)

    result = {}
    for _, row in df.iterrows():
        key = int(row["original_index"])
        values = [int(v) for v in str(row["duplicate_indices"]).split(";")]
        result[key] = values

    return result

def get_ESM_prot_vecs(esm_vectors = 'ESM/all_proteins_emb_vectors.csv.gz'):
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



def create_similar_matrix_from_mmseq2(similarity_output = 'Data_sets/similarity_train.tsv',col_value= 'pident'):
    """Create a similarity matrix from mmseq2 output tsv file.
    Col_value can be pident, qcov, evalue, bits.
    The similary is based on the col_value.

    Args:
        similarity_output (str, optional): path. Defaults to 'Data_sets/similarity_train.tsv'.
        col_value (str, optional): str. Defaults to 'pident'.
    """
    ids_sorted = [f'seq{i+1}'for i in range(200)] # CHNAGE THIS TO NUMBER OF SEQUENCES!!
    cols = ["query","target","pident","alnlen","qlen","tlen","qcov","tcov","evalue","bits"]
    df = pd.read_csv(similarity_output, sep="\t", names=cols)
    
    mat = df.pivot(index="query", columns="target", values=col_value)
    mat = mat.reindex(index=ids_sorted, columns=ids_sorted)

    mat = mat.fillna(0.0)
    mat.to_csv(f"{col_value}.csv")

def get_simliary_dict(similarity_score='Data_sets/pident.csv', treshold=  80):
    """Get a dictionary of similar protines/rnas based on the similarity score matrix.

    Args:
        similarity_score (matrix of simlirties scores, optional): path. Defaults to 'Data_sets/pident.csv'.
        treshold (int, optional): above what treshold to compare. Defaults to 80.

    Returns:
        dict: pairs of similartites
    """
    mat = pd.read_csv(similarity_score)
    s = mat.where(mat > treshold).stack()
    similariteis = {index:s.loc[index].to_dict() for index in mat.index}
    return similariteis

