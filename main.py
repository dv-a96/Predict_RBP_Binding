import sys
from tensorflow.keras.models import load_model
from model_utilities import *
from train_test_utilities import PairDatasetFactory
from data_processing import rna_one_hot, validate_rna_sequences, get_ESM_prot_vecs
import pandas as pd
# import torch
# import esm

BATCH_SIZE = 4096
MODELS = []


def embedd_prot_vector(prot_vector):
    """Create the protein embedding on the fly using ESM2 model

    Args:
        prot_vector (str): sequence of the protein

    Returns:
        embedde array: np.array
    """
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    data = [("prot", prot_vector)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6])
        token_representations = results["representations"][6]
    tokens = token_representations[0]  
    truncate_len = tokens.shape[0] - 1  # Exclude BOS token
    mean_representation = tokens[1:truncate_len].mean(0) # Exlucde CLS token and mean
    return mean_representation.numpy()
def get_prot(prot_sequence=None):
    """get the protein embedding from the precomputed embeddings or create it on the fly

    Args:
        prot_sequence (str, optional): protein sequence. Defaults to None.

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        embedded vector: _description_
    """
    if prot_sequence is None:
        raise ValueError("Protein sequence must be provided")
    if not isinstance(prot_sequence, str):
        raise ValueError("Protein sequence must be a string")
    prot_sequence = prot_sequence.upper()
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
    if not set(prot_sequence).issubset(valid_aas):
        print("Invalid sequence, contains:", set(prot_sequence) - valid_aas)
        raise ValueError("Protein sequence contains invalid amino acids")
    rbp_esm_path = 'Data_sets/rbps_2_embeding_esm.csv'
    rbps_vectors = get_ESM_prot_vecs(rbp_esm_path)
    rbp_sequences = 'Data_sets/test_RBPs2.txt'
    rbp_sequences = pd.read_csv(rbp_sequences, header=None)
    prot_index = rbp_sequences[rbp_sequences[0] == prot_sequence].index
    if prot_index.empty:
        print("Protein sequence not found in the dataset: create embedding on the fly")
        return embedd_prot_vector(prot_sequence)
    return rbps_vectors[prot_index].reshape(-1)

def load_data(rbp_file_path, rna_file_path):
    rna_df = pd.read_csv(rna_file_path, header=None)
    rbp_df = pd.read_csv(rbp_file_path, header=None)
    rna_df, _ = validate_rna_sequences(rna_df)
    rnas = rna_one_hot(rna_df)
    rnas = rnas[:,:,:4] # keep only the first 4 bits.
    rbp_seq = get_prot(rbp_df.iloc[0,0])
    rbp_seq = rbp_seq.reshape(1, -1)
    return rnas, rbp_seq

def load_models():
    path = 'Models/Checkpoints/esm_cnn_cluster_all/esm_cnn_regression_quantile_MSE_Adam_2025-08-25_15-11-50.keras'
    model = load_model(path,custom_objects = CUSTOM_OBJECTS)
    return model

def main():
    output_file = sys.argv[1]
    rnas, rbp = load_data(sys.argv[2], sys.argv[3])
    factory =  PairDatasetFactory(rbp,rnas)
    test_dataset = factory.make_train(prot_ids=1, batch_size=BATCH_SIZE, shuffle=False)
    model = load_models()
    preds = model.predict(test_dataset)
    print('done')

    
if __name__ == "__main__":
    main()