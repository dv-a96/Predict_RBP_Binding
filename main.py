import sys
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from model_utilities import *
from train_test_utilities import PairDatasetFactory, get_clusteres_indices
from data_processing import validate_rna_sequences, get_ESM_prot_vecs, rna_one_hot
from evaluate_models import prot_model_to_idx, knn_models_predict, get_knn_emb_indices
import pandas as pd
import re
import numpy as np
import torch
import esm

BATCH_SIZE = 4096

def parse_args():
    if len(sys.argv) > 5:
        print("Usage: python main.py <output_file> <rbp.txt> <rna.txt> <optional: Batch_size>")
        sys.exit(1)
    output_file = sys.argv[1]
    rbp_file = sys.argv[2]
    rna_file = sys.argv[3]
    if len(sys.argv) == 5:
        global BATCH_SIZE
        BATCH_SIZE = int(sys.argv[4])
        if BATCH_SIZE <= 0:
            print("Batch size must be a positive integer.")
            sys.exit(1)
    print(f"Using batch size: {BATCH_SIZE}")
    return output_file, rbp_file, rna_file

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
    return rbps_vectors[prot_index].reshape(-1), prot_index[0]

def load_data(rbp_file_path, rna_file_path):
    rna_df = pd.read_csv(rna_file_path, header=None)
    rbp_df = pd.read_csv(rbp_file_path, header=None)
    rna_df, _ = validate_rna_sequences(rna_df)
    rnas = rna_one_hot(rna_df)
    rnas = rnas[:,:,:4] # keep only the first 4 bits.
    rbp_seq, rbp_index = get_prot(rbp_df.iloc[0,0])
    rbp_seq = rbp_seq.reshape(1, -1)
    return rnas, rbp_seq ,rbp_index




def save_to_output_file(file_name, preds, rbp_index=0):
    re_patters = re.compile(r'RBP(\d+)')
    match = re_patters.search(file_name)
    if not match:
        file_name = f'RBP{rbp_index+201}.txt'
        print(f"Invalid file name format. Expected format: 'RBP<index>.txt'\nSaving to: {file_name}")
    if not file_name.endswith('.txt'):
        file_name += '.txt'
    with open(file_name, 'w') as f:
        for pred in preds:
            f.write(f"{pred}\n")
    print(f"Predictions saved to {file_name}")
    f.close()

def main():
    output_file, rbp_file, rna_file = parse_args()
    rnas, rbp_seq, rbp_index = load_data(rbp_file, rna_file)
    training_rbps = get_ESM_prot_vecs("Data_sets/all_proteins_emb_vectors.csv.gz")
    cluster_idx = get_clusteres_indices(cluster_id='all')
    all_rbps_indices = np.concatenate([val for val in cluster_idx.values()])
    training_rbps = training_rbps[all_rbps_indices]
    knn_indices = get_knn_emb_indices(rbp_seq, K=3, df=training_rbps, metric="cosine")
    original_prots_indexes = all_rbps_indices[knn_indices]
    models = prot_model_to_idx(indices=original_prots_indexes,folder='esm_cnn_Baseline_perProt')
    factory =  PairDatasetFactory(rbp_seq,rnas)
    test_dataset = factory.make_train( batch_size=BATCH_SIZE, shuffle=False)
    prediction = knn_models_predict(test_dataset=test_dataset, models=models, numer_of_samples=rnas.shape[0], batch_size=BATCH_SIZE)
    save_to_output_file(output_file, prediction,rbp_index)
# def main_2():
#     #rnas, rbp_seq, rbp_index = load_data(sys.argv[2], sys.argv[3])
#     testing_rbps = get_ESM_prot_vecs("Data_sets/rbps_2_embeding_esm.csv")
#     rnas = pd.read_csv("Data_sets/test_seqs.txt",header=None)
#     rnas = rna_one_hot(rnas)
#     rnas = rnas[:,:,:4] 
#     factory =  PairDatasetFactory(testing_rbps,rnas)
#     training_rbps = get_ESM_prot_vecs()
#     cluster_idx = get_clusteres_indices(cluster_id='all')
#     all_rbps_indices = np.concatenate([val for val in cluster_idx.values()])
#     training_rbps = training_rbps[all_rbps_indices]
#     for i,rbp_seq in enumerate(testing_rbps):
#         knn_indices = get_knn_emb_indices(rbp_seq, K=3, df=training_rbps, metric="cosine")
#         original_prots_indexes = all_rbps_indices[knn_indices]
#         models = prot_model_to_idx(indices=original_prots_indexes)
#         test_dataset = factory.make_train( batch_size=BATCH_SIZE, shuffle=False,prot_ids=[i])
#         prediction = knn_models_predict(test_dataset=test_dataset, models=models, numer_of_samples=rnas.shape[0], batch_size=BATCH_SIZE)
#         save_to_output_file(f'RBP{201+i}',prediction)
    
if __name__ == "__main__":
    main()