import sys
from tensorflow.keras.models import load_model
from model_utilities import *
from train_test_utilities import PairDatasetFactory
from data_processing import rna_one_hot, validate_rna_sequences
import pandas as pd
BATCH_SIZE = 4096
MODELS = []


def load_data(rbp_file_path, rna_file_path):
    rna_df = pd.read_csv(rna_file_path, header=None)
    rbp_df = pd.read_csv(rbp_file_path, header=None)
    rna_df, _ = validate_rna_sequences(rna_df)
    return rbp_df, rna_df
def main():
    output_file = sys.argv[1]
    rbp_file_path = sys.argv[2]
    rna_file_path = sys.argv[3]
    rbp_df, rna_df = load_data(rbp_file_path, rna_file_path)
    rnas = rna_one_hot(rna_df)
    rnas = rnas[:,:,:4] # keep only the first 4 bits.
    rbps_indices = rbp_df.index.tolist()
    
if __name__ == "__main__":
    main()