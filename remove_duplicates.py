import pandas as pd
import csv
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

rbps = 'Data_sets/training_RBPs2.txt'
intensities = 'Data_sets/training_data2.txt.gz'
rnas = 'Data_sets/training_seqs.txt'
Figures = 'Figures/Duplicates'

rbps = pd.read_csv(rbps, header=None)
rnas = pd.read_csv(rnas, header=None)

def drop_duplicates_return(df, subset=None, keep='first'):
    """
    Removes duplicate rows and returns a new DataFrame along with indices of removed rows.
    
    Args:
        df (pd.DataFrame): Original DataFrame
        subset (list[str] | None): Columns to consider for identifying duplicates
        keep (str): 'first', 'last', or False
    
    Returns:
        df_unique (pd.DataFrame): DataFrame without duplicates
        removed_indices (pd.Index): Indices of rows that were removed
    """
    mask = ~df.duplicated(subset=subset, keep=keep)  # True for rows to keep
    df_unique = df[mask].copy()
    removed_indices = df.index[~mask]  # indices of removed rows
    return df_unique, removed_indices



def map_unique_to_removed(df_original, df_unique, subset=None):
    """
    Returns a dictionary mapping each unique row in df_unique 
    to the list of duplicate indices (from df_original) that were removed.

    Args:
        df_original (pd.DataFrame): Original DataFrame before removing duplicates
        df_unique (pd.DataFrame): DataFrame after removing duplicates
        subset (list[str] | None): Columns to consider for identifying duplicates

    Returns:
        mapping (dict): key = index of a unique row (df_unique),
                        value = list of indices of removed duplicates (df_original)
    """
    if subset is None:
        subset = df_original.columns.tolist()

    mapping = {}
    for u_idx, u_row in df_unique.iterrows():
        # Find all matching rows in df_original
        matches = df_original[(df_original[subset] == u_row[subset]).all(axis=1)]

        # Exclude the unique row itself, keep only duplicates
        removed_indices = matches.index.difference([u_idx]).tolist()

        if removed_indices:  # Only store if duplicates exist
            mapping[u_idx] = removed_indices

    return mapping

# rbps_uniqe, removed_rbps = drop_duplicates_return(rbps)
# print("RBPs len ", len(rbps))
# print("UNIQE_RBPs len ", len(rbps_uniqe))
# print("REMOVED_RBPs len ", len(removed_rbps))
# print(removed_rbps)

# dict = map_unique_to_removed(rbps, rbps_uniqe)
# print(dict)

# rbps_uniqe.to_csv('./Data_sets/training_RBPs_UNIQE.csv', index=False, header=None)



def save_dict_to_csv(mapping, filename):
    """
    Save dictionary (key -> list of values) to CSV file.
    
    Args:
        mapping (dict): dictionary where values are lists
        filename (str): path to save the CSV
    """
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["original_index", "duplicate_indices"])  # header
        
        for key, values in mapping.items():
            writer.writerow([key, ";".join(map(str, values))])  # store list as "a;b;c"
# save_dict_to_csv(dict, './Data_sets/idx_remove_matches,csv')


def csv_to_dict(file_path):
    df = pd.read_csv(file_path)

    result = {}
    for _, row in df.iterrows():
        key = int(row["original_index"])
        values = [int(v) for v in str(row["duplicate_indices"]).split(";")]
        result[key] = values

    return result


def save_correlations(file_path, index_dict, sep="\t", out_dir="."):
    """
    file_path : str - קובץ טקסט עם עמודות כאינדקסים
    index_dict : dict - {key: [indices]}
    sep : str - מפריד עמודות בקובץ (ברירת מחדל טאב)
    out_dir : str - תיקייה לשמירת הקבצים
    """

    # קריאת הקובץ
    df = pd.read_csv(file_path, sep=sep, header=None)
    df.columns = range(df.shape[1])


    for key, idx_list in index_dict.items():
        n = len(idx_list)
        fig, axes = plt.subplots(1, n, figsize=(5*n, 4), squeeze=False)
        fig.suptitle(f"Correlations for protein {key}", fontsize=16)

        x = df[key].values  # עמודת המפתח

        for j, idx in enumerate(idx_list):
            y = df[idx].values

            # חישוב מתאם פירסון
            r, p = pearsonr(x, y)

            ax = axes[0, j]
            ax.scatter(x, y, alpha=0.6)
            ax.set_title(f"{key} vs {idx}\n r={r:.2f}, p={p:.2e}", fontsize=10)
            ax.set_xlabel(str(key))
            ax.set_ylabel(str(idx))

        plt.tight_layout()
        # שמירה לקובץ
        out_path = f"{out_dir}/correlations_protein{key}.png"
        plt.savefig(out_path, dpi=300)
        plt.close(fig)


dup_dict = csv_to_dict('Data_sets/idx_remove_matches,csv')
save_correlations('Data_sets/training_data2.txt',dup_dict, sep='\t', out_dir=Figures)
