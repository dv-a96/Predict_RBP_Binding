'''
Module to conduct a simple data anaylsis on the RBPS, RNAS, and their intensity values'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import entropy
import seaborn as sns
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from data_processing import quantile_normalize, clamp_by_precentile



# rbps = 'Data_sets/training_RBPs2.txt'
# intensities = 'Data_sets/training_data2.txt.gz'
# rnas = 'Data_sets/training_seqs.txt'
Figures = 'Figures'
# # intensities
# intensities = pd.read_csv(intensities,sep='\t',header=None)
# rbps = pd.read_csv(rbps,header=None)
# rbps['Lengths'] = rbps[0].apply(lambda x: len(x))
# #intensities = clamp_by_precentile(intensities)
# #intensities = quantile_normalize(intensities)

def overlay_label_feature_histograms(df, label_col, feature_cols=None, bins=30, figsize=(12, 8),corr_values = None):
    """
    For each feature column, create a subplot with two histograms:
    - histogram of the feature's values
    - histogram of the label column's values (overlayed)

    Args:
        df (pd.DataFrame): dataset
        label_col (str): name of label column (continuous)
        feature_cols (list[str] or None): feature columns to compare against the label.
                                          If None, use all numeric except label_col.
        bins (int): number of bins
        figsize (tuple): figure size
    """
    if isinstance(df, str):
        df = pd.read_csv(df)
    if feature_cols is None:
        feature_cols = df.select_dtypes(include="number").columns.tolist()
        if label_col in feature_cols:
            feature_cols.remove(label_col)

    n = len(feature_cols)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    label_values = df[label_col].dropna()

    for i, col in enumerate(feature_cols):
        ax = axes[i]

        feat_values = df[col].dropna()
        uniq_count = feat_values.nunique()   # number of unique values

        # get correlation value if provided
        if corr_values is not None and col in corr_values:
            corr_text = f"R={corr_values[col]:.3f}, unique={uniq_count}"
        else:
            corr_text = f"unique={uniq_count}"
        # compute common bin edges across both distributions
        all_vals = np.concatenate([label_values, feat_values])
        bin_edges = np.linspace(all_vals.min(), all_vals.max(), bins+1)
        feat_label = f"{col}\n{corr_text}"
        ax.hist(feat_values, bins=bin_edges, alpha=0.6, color="steelblue", label=feat_label)
        ax.hist(label_values, bins=bin_edges, alpha=0.4, color="orange", label=label_col)
       
        ax.set_title(col)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.legend()

    # Hide unused axes
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    temp_path = os.path.join(Figures, f"overlay_label_histograms_{label_col}.png")
    plt.savefig(temp_path, dpi=300)
    plt.close(fig)

def get_corr(file='Evaluation/summ.csv',cols = None):
    df = pd.read_csv(file)
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()
    else:
        # convert integer indices to column names if given
        cols = [df.columns[c] if isinstance(c, int) else c for c in cols]
    print("Correlation matrix for selected columns:")
    corr = df[cols].corr()['quantile_labels']
    print(corr)
    return corr

def sub_plot_intensities_histo(data_frame, cols = None, bins = 500, name = '',corr_values = None):
    """Plot histograms of intensity distributions for specified columns.

    Args:
        data_frame (pd.DataFrame): DataFrame containing intensity data.
        cols (list, optional): List of column names/indexes of columns to plot. Defaults to None.
    """
    if isinstance(data_frame,str):
        data_frame = pd.read_csv(data_frame)
    if cols is None:
        cols = data_frame.select_dtypes(include="number").columns.tolist()
    else:
        # convert integer indices to column names if given
        cols = [data_frame.columns[c] if isinstance(c, int) else c for c in cols]

    n_cols = len(cols)
    nrows = math.ceil(n_cols / 3)    # up to 3 plots per row
    ncols = min(3, n_cols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*5))
    axes = np.atleast_1d(axes).ravel()  # flatten to 1D array for easy indexing

    for i, col in enumerate(cols):
        axes[i].hist(data_frame[col].dropna(), bins=bins, color="steelblue", edgecolor="black")
        axes[i].set_title(col, fontsize=10)
        axes[i].text(
            0.5, 0.9,                                      # (x,y) in axes coords
            f"R: {corr_values[col]:.3f}" if corr_values is not None else "",
            ha="center", va="top",                         # centered horizontally, anchored at top
            transform=axes[i].transAxes,                   # <- use axis-relative coords
            fontsize=9, color="darkred"
        )        
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")

    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    path = os.path.join(Figures,f'{name}_historgram.png')
    plt.savefig(path, dpi=300)




def plot_intenseties_per_indice(indice =132):
    global intensities
    # intensities = clamp_by_precentile(intensities)
    # intensities = quantile_normalize(intensities)
    intensities_ = intensities.iloc[:,indice]
    plot_histo(intensities_,f'indice_{indice}')
def plot_histo(df,name):
    plt.hist(df, bins=200)   # adjust bins as you like
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of values")
    path = os.path.join(Figures,f'{name}_historgram.png')
    plt.savefig(path,dpi=300)
    plt.close()

def plot_rbps_length_histogram(rbps_df):
    """
    Calculate the lengths of each Rbp and plot a histogram of the varoius lengths
    Args:
        rbps_df (data_frame): data frame with rbps sequences
    """
    
    bins = len(rbps['Lengths'].unique())
    rbps['Lengths'].plot.hist(bins=bins)
    plt.xlabel('Rbps sequence length')
    plt.ylabel('Amount in data')
    path = os.path.join(Figures,'rbps_length_historgram.png')
    plt.savefig(path,dpi=300)
    plt.close()

def plot_min_max_intensities_histograms(intensities):
    """Plot the intensities between rbps and rna histograms as follows:
    1. Histograms of the max values per protein
    2. Plot per rna

    Args:
        intensities (_type_): _description_
    """
    max_values = [max(intensities[i]) for i in intensities.columns]
    min_values = [min(intensities[i]) for i in intensities.columns]
    indices = np.arange(len(min_values))
    width = 0.8

    plt.bar(indices, max_values, width=width, color='lightcoral', label='Max values')
    plt.bar(indices, min_values, width=width, color='darkred', label='Min values')

    plt.xlabel('Protien')
    plt.ylabel('Min/Max intensity')
    plt.legend()
    path = os.path.join(Figures,'Protein_min_max_intensity.png')
    plt.savefig(path,dpi=300)

def plot_intensities_historgram(intensities):
    """Plot historgram of the intensities values

    Args:
        intensities (Data frame): RNAxRBP
    """
    plt.hist(intensities.values.flatten(), bins=100, color='skyblue', edgecolor='black')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram of All Values in DataFrame')
    path = os.path.join(Figures,'Intensities_histogram.png')
    plt.savefig(path,dpi=300)

### CHECK CORELATION FOR:
''' Rbp length vs intensity
Rna secondary structure to intensities disterbution'''

def gini_coefficient(x):
    """Compute Gini index from array x."""
    x = np.sort(np.abs(x))  # ensure non-neg values
    n = len(x)
    if np.sum(x) == 0:
        return 0.0
    return (2 * np.sum((np.arange(1, n+1) * x)) / (n * np.sum(x))) - (n + 1) / n
def compute_rbp_binding_metrics(matrix, threshold=0.5, top_n=10):
    """
    Compute summary statistics per RBP column.

    Parameters:
        matrix (np.ndarray or pd.DataFrame): Shape (num_probes, num_rbps)
        threshold (float): Threshold for counting strong binders
        top_n (int): Number of top values to average
    
    Returns:
        pd.DataFrame: Shape (num_rbps, num_metrics)
    """
    if isinstance(matrix, pd.DataFrame):
        rbp_names = matrix.columns
        matrix = matrix.values
    else:
        rbp_names = [f"RBP_{i}" for i in range(matrix.shape[1])]

    metrics = []

    for col in matrix.T:
        values = np.array(col)
        sorted_vals = np.sort(values)[-top_n:] if len(values) >= top_n else values
        norm_vals = values / np.sum(values) if np.sum(values) > 0 else values

        metrics.append({
            'mean': np.mean(values),
            'std': np.std(values),
            'max': np.max(values),
            'count_above_0.5': np.sum(values > threshold),
            f'top{top_n}_mean': np.mean(sorted_vals),
            'gini': gini_coefficient(values),
            'entropy': entropy(norm_vals + 1e-12)  
        })

    return pd.DataFrame(metrics, index=rbp_names)



def plot_len_correlations(df, len_col='RBP length', kind='scatter'):
    metrics = [col for col in df.columns if col != len_col]
    n = len(metrics)
    ncols = 3  # adjust this for layout
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten()

    for i, col in enumerate(metrics):
        ax = axes[i]
        sns.scatterplot(x=df[len_col], y=df[col], ax=ax)
        corr = df[[len_col, col]].corr().iloc[0, 1]
        ax.set_title(f"Pearson r = {corr:.3f}")
        ax.set_xlabel(len_col)
        ax.set_ylabel(col)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    path = os.path.join(Figures,'Length_correlations.png')
    plt.tight_layout()
    plt.savefig(path,dpi=300)
    plt.close()

def scale_data(intensities_df):
    """Scale and center the data for pca,tsne,umap anaylsis

    Args:
        intensities_df (pd.dataframe): rbpsXrna

    """
    return StandardScaler().fit_transform(intensities_df)

def run_pca(intensities_df, return_pca = False):
    """Conduct pca analysis on the intensities data frame.
    Rbps as samples and RNA intensities as features.
    Calculate what variance is captured by minimum amount of rnas

    Args:
        intensities_df (pd.DataFrame): RBPs X Rnas intensities
        return_pca (bool, optional): return the transformed data    
    Returns:
        X_pca: transformed data
    """
    X_scaled = StandardScaler().fit_transform(intensities_df)

    print("NaNs in X_scaled:", np.isnan(X_scaled).any())
    print("Infs in X_scaled:", np.isinf(X_scaled).any())
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Plot cumulative variance
    cum_var = np.cumsum(pca.explained_variance_ratio_)
   
    X_pca_2D = X_pca[:, :2]

    # Create figure and 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Subplot 1: Cumulative Explained Variance ---
    axes[0].plot(range(1, len(cum_var)+1), cum_var, marker='o')
    axes[0].axhline(y=0.95, color='r', linestyle='--')
    axes[0].set_xlabel('Number of components')
    axes[0].set_ylabel('Cumulative explained variance')
    axes[0].set_title('Explained Variance vs. Components')
    axes[0].grid(True)

    # --- Subplot 2: Scree Plot ---
    axes[1].plot(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, marker='o')
    axes[1].set_xlabel('Component number')
    axes[1].set_ylabel('Explained variance ratio')
    axes[1].set_title('Scree Plot')
    axes[1].grid(True)

    # --- Subplot 3: 2D PCA Scatter Plot ---
    axes[2].scatter(X_pca_2D[:, 0], X_pca_2D[:, 1], s=60, alpha=0.8)
    axes[2].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[2].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[2].set_title('2D PCA Plot')
    axes[2].grid(True)

    # Save and close
    plt.tight_layout()
    path = os.path.join(Figures, 'PCA_summary_plots.png')
    plt.savefig(path, dpi=300)
    plt.close()
    if return_pca:
        return X_pca

def plot_tsne_vs_umap(intensities_df, processed = False):
        # --- Step 1: UMAP ---
    if not processed:
        X_scaled=scale_data(intensities_df)
    else : X_scaled = intensities_df
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    X_umap = umap_reducer.fit_transform(X_scaled)

    # --- Step 2: t-SNE ---
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_scaled)

    # --- Step 3: Plot side-by-side ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # UMAP plot
    axes[0].scatter(X_umap[:, 0], X_umap[:, 1], s=60, alpha=0.8)
    axes[0].set_title('UMAP projection')
    axes[0].set_xlabel('UMAP 1')
    axes[0].set_ylabel('UMAP 2')
    axes[0].grid(True)

    # t-SNE plot
    axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], s=60, alpha=0.8)
    axes[1].set_title('t-SNE projection')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].grid(True)

    plt.tight_layout()
    if not processed:
        path = os.path.join(Figures, 'umap_vs_tsne.png')
    else: path = os.path.join(Figures,'umap_vs_tsne_processed.png')
    plt.savefig(path, dpi=300)
    plt.close()
if __name__ == '__main__':
    #plot_rbps_length_histogram(rbps)
    #plot_min_max_intensities_histograms(intensities)
    #plot_intensities_historgram(intensities)
    # intensities_metrics_df = compute_rbp_binding_metrics(intensities)
    # intensities_metrics_df['RBP length'] = rbps['Lengths'] 
    # plot_len_correlations(intensities_metrics_df)
    # intensities_transformed = intensities.T
    # pca_data = run_pca(intensities_transformed,return_pca=True)
    # X_pca_150 = pca_data[:, :150]

    # plot_tsne_vs_umap(X_pca_150,processed=True)
    cols = ["quantile_labels","ESM_CNN_Guas","esm_cnn_MSE","esm_cnn_logcosh","esm_cnn_logcosh_RMSprop","esm_cnn_MAE",
            "only_rna64_MSE_Adam","only_rna_7_964_MSE_Adam","only_rna_sec64_MSE_Adam",
            "only_rna_sec64_MAE_Adam","only_rna_sec64_logcosh_RMSprop"]
    corr = get_corr(cols=cols)
    
    # sub_plot_intensities_histo("Evaluation/summ.csv",cols=cols,name="Histo_ESM_OPT_LOSS",corr_values = corr)
    cols_ = ["ESM_CNN_Guas","esm_cnn_MSE","esm_cnn_logcosh","esm_cnn_logcosh_RMSprop","esm_cnn_MAE",
            "only_rna64_MSE_Adam","only_rna_7_964_MSE_Adam","only_rna_sec64_MSE_Adam",
            "only_rna_sec64_MAE_Adam","only_rna_sec64_logcosh_RMSprop"]
    overlay_label_feature_histograms(df="Evaluation/summ.csv", label_col="quantile_labels", 
                                     feature_cols=cols_, bins=500, figsize=(14, 8), corr_values=corr)