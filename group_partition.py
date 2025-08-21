import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS

def overlap_to_distance(overlap_matrix_path):
    """
    Convert an overlap (similarity) matrix to a distance matrix.

    Parameters
    ----------
    overlap_matrix_path : str
        Path to a CSV file containing a square overlap matrix.
        The CSV is assumed to have:
        - The first row as column headers.
        - The first column as row labels (which will be ignored).

    Returns
    -------
    dist : numpy.ndarray
        A distance matrix of the same shape as the input.
        Each element is computed as: distance = 1 - overlap.

    Notes
    -----
    - The input overlap matrix should have values in [0,1].
      0 = no overlap, 1 = identical.
    - The returned distance matrix will also be in [0,1].
    - The function ignores the original row and column names.
    """
    
    # Read CSV as DataFrame (first row = column names, first column = row labels)
    overlap_df = pd.read_csv(overlap_matrix_path, header=0, index_col=0)
    
    # Extract raw values as a NumPy array
    overlap_matrix = overlap_df.values  # shape: (n, n)
    
    # Convert similarity (overlap) to distance
    dist = 1 - overlap_matrix
    
    return dist



def hierarchical_clustering_from_csv(dist_matrix_path, num_clusters=4, method='average', plot_dendrogram=False):
    """
    Perform hierarchical clustering from a distance matrix CSV.
    
    Parameters
    ----------
    dist_matrix_path : str
        Path to CSV containing the distance matrix (square, symmetric).
        First row/column can be labels (they will be ignored).
    num_clusters : int
        Number of clusters to form after cutting the dendrogram.
    method : str
        Linkage method: 'single', 'complete', 'average', 'ward', etc.
    plot_dendrogram : bool
        Whether to plot the dendrogram.
        
    Returns
    -------
    clusters : pandas.Series
        Cluster assignment for each row (index = row number, value = cluster label)
    """
    # 1️⃣ Load distance matrix
    df = pd.read_csv(dist_matrix_path, header=0, index_col=0)
    dist_matrix = df.values
    
    # 2️⃣ Compute hierarchical linkage
    Z = linkage(dist_matrix, method=method)
    
    # 3️⃣ Assign clusters
    cluster_labels = fcluster(Z, t=num_clusters, criterion='maxclust')
    
    # 4️⃣ Convert to pandas Series for easy use
    clusters = pd.Series(cluster_labels, index=df.index)
    
    # 5️⃣ Optional: plot dendrogram
    if plot_dendrogram:
        plt.figure(figsize=(10, 6))
        dendrogram(Z, labels=df.index.tolist())
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Item')
        plt.ylabel('Distance')
        plt.show()

        sns.clustermap(dist_matrix, row_linkage=Z, col_linkage=Z, figsize=(8,8))
        plt.show()

    
    return clusters



def plot_cluster_counts_side_by_side(csv1, csv2, labels=('Method1','Method2'), out_file="cluster_counts.png"):
    """
    Plot number of proteins per cluster for two cluster assignments in side-by-side subplots.
    Saves the figure to a file.
    """
    # Load cluster assignments
    clusters1 = pd.read_csv(csv1, index_col=0)
    clusters2 = pd.read_csv(csv2, index_col=0)

     # Assume the cluster column is the first (only) column
    clusters1 = clusters1.iloc[:,0]
    clusters2 = clusters2.iloc[:,0]

    counts1 = clusters1.value_counts().sort_index()
    counts2 = clusters2.value_counts().sort_index()

    clusters = sorted(set(counts1.index).union(counts2.index))

    fig, axes = plt.subplots(1, 2, figsize=(14,6))

    # Plot first CSV
    axes[0].bar(clusters, [counts1.get(c,0) for c in clusters])
    axes[0].set_title(labels[0])
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("Number of proteins")

    # Plot second CSV
    axes[1].bar(clusters, [counts2.get(c,0) for c in clusters])
    axes[1].set_title(labels[1])
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Number of proteins")

    plt.suptitle("Cluster Counts Comparison")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_file, dpi=300)
    plt.close()




def plot_two_clusterings_mds(csv1, dist_csv1, csv2, dist_csv2, labels=('Method1','Method2'), out_file="cluster_mds.png"):
    """
    Project proteins into 2D via MDS using two different distance matrices, 
    and plot the clusters from two methods side by side.

    Parameters
    ----------
    csv1, csv2 : str
        Paths to CSV files with cluster assignments (first column: protein, second: cluster)
    dist_csv1, dist_csv2 : str
        Paths to CSV files with precomputed distance matrices for each clustering
    labels : tuple
        Labels for the two clusterings
    out_file : str
        File path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14,6))

    for ax, clusters_csv, dist_csv, label in zip(axes, [csv1, csv2], [dist_csv1, dist_csv2], labels):
        # Load distance matrix
        df_dist = pd.read_csv(dist_csv, header=0, index_col=0)
        proteins = df_dist.index.tolist()

        # Compute 2D MDS projection
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        coords = mds.fit_transform(df_dist.values)

        # Load cluster assignments
        clusters_df = pd.read_csv(clusters_csv, index_col=0)
        clusters = clusters_df.iloc[:,0]  # assume cluster is in first column

        # Scatter plot colored by cluster
        for cluster in clusters.unique():
            idx = [proteins.index(p) for p in clusters[clusters==cluster].index]
            ax.scatter(coords[idx,0], coords[idx,1], label=f'Cluster {cluster}', alpha=0.7)
        ax.set_title(label)
        ax.set_xlabel("MDS1")
        ax.set_ylabel("MDS2")
        ax.legend()

    plt.suptitle("MDS Projection of Proteins by Cluster")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_file, dpi=300)
    plt.close()


plot_two_clusterings_mds('Data_sets/protein_overlap_clusters.csv','Data_sets/protein_overlap_dist_matrix.csv', 'Data_sets/protein_wasserstein_clusters.csv', 'Data_sets/protein_wasserstein_matrix.csv',  labels=('overlap', 'wasserstein'), out_file='Figures/cluster_pca.png')