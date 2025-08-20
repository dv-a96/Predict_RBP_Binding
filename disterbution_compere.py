import numpy as np
from scipy.stats import wasserstein_distance
import pandas as pd

data = pd.read_csv("Data_sets/training_data2.txt", delimiter='\t', header=None)
data.columns = range(data.shape[1])
col_names = data.columns
num_cols = len(col_names)


def overlap_coefficient_from_data(data1, data2, num_bins=50):
    """
    Compute overlap coefficient from raw data columns.
    
    Parameters:
    data1, data2 : arrays of raw data points
    num_bins : number of bins to discretize the data
    
    Returns:
    overlap : float between 0 and 1
    """
    # Define bin edges covering both datasets
    bins = np.linspace(min(data1.min(), data2.min()), max(data1.max(), data2.max()), num_bins)
    
    # Compute histograms
    hist1, _ = np.histogram(data1, bins=bins)
    hist2, _ = np.histogram(data2, bins=bins)
    
    # Normalize to probabilities
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    
    # Sum the minimum probabilities per bin
    overlap = np.sum(np.minimum(hist1, hist2))
    return overlap



# Initialize matrices as DataFrames
overlap_df = pd.DataFrame(index=col_names, columns=col_names, dtype=float)
wasserstein_df = pd.DataFrame(index=col_names, columns=col_names, dtype=float)

# Compute pairwise metrics
for i in col_names:
    for j in col_names:
        col_i = data[i].values
        col_j = data[j].values
        print("compute pair {} out of {}". format(i*200+j, num_cols*num_cols))
        overlap_df.loc[i, j] = overlap_coefficient_from_data(col_i, col_j)
        wasserstein_df.loc[i, j] = wasserstein_distance(col_i, col_j)


# Save to CSV with row and column labels
overlap_df.to_csv("Data_sets/protein_overlap_matrix.csv")
wasserstein_df.to_csv("Data_sets/protein_wasserstein_matrix.csv")