import os
import torch
import pandas as pd
import argparse

def extract_embeddings(embedding_dir, output_csv, layer):
    data_rows = []

    for filename in os.listdir(embedding_dir):
        if filename.endswith(".pt"):
            filepath = os.path.join(embedding_dir, filename)
            data = torch.load(filepath)

            # Use the protein label from the ESM output (taken from the FASTA header)
            protein_id = data["label"]

            # Get the mean representation from the specified layer
            vec = data["mean_representations"][layer]
            vec_np = vec.numpy().tolist()

            # Combine protein ID and vector
            row = [protein_id] + vec_np
            data_rows.append(row)

    # Create column names: "protein_id", "f0", "f1", ..., "fn"
    embedding_dim = len(data_rows[0]) - 1
    columns = ["protein_id"] + [f"f{i}" for i in range(embedding_dim)]

    # Build and save DataFrame
    df = pd.DataFrame(data_rows, columns=columns)
    df.to_csv(output_csv, index=False)

    print(f"Saved {len(df)} protein embeddings to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Extract mean protein embeddings from .pt files")
    parser.add_argument("embedding_dir", help="Directory containing .pt files")
    parser.add_argument("output_csv", help="Path to output CSV file")
    parser.add_argument("layer", type=int, help="Layer number to extract from mean_representations")

    args = parser.parse_args()
    extract_embeddings(args.embedding_dir, args.output_csv, args.layer)

if __name__ == "__main__":
    main()
