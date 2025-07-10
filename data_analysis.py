'''
Module to conduct a simple data anaylsis on the RBPS, RNAS, and their intensity values'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

rbps = 'Data_sets/training_RBPs2.txt'
intensities = 'Data_sets/training_data2.txt'
rnas = 'Data_sets/training_seqs.txt'
Figures = 'Figures'
# intensities
intensities = pd.read_csv(intensities,sep='\t',header=None)
rbps = pd.read_csv(rbps,header=None)

def plot_rbps_length_histogram(rbps_df):
    """
    Calculate the lengths of each Rbp and plot a histogram of the varoius lengths
    Args:
        rbps_df (data_frame): data frame with rbps sequences
    """
    rbps['Lengths'] = rbps[0].apply(lambda x: len(x))
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
    plt.hist(intensities.values.flatten(), bins=100, color='skyblue', edgecolor='black')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram of All Values in DataFrame')
    path = os.path.join(Figures,'Intensities_histogram.png')
    plt.savefig(path,dpi=300)



if __name__ == '__main__':
    #plot_rbps_length_histogram(rbps)
    #plot_min_max_intensities_histograms(intensities)
    plot_intensities_historgram(intensities)

    