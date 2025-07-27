First make sure you have PyTorch installed.
Then install the `fair-esm` a lybrary for downloading and using ESM models by running:
`pip install fair-esm`.

Then to compute embedding for protein in `FASTA` file run the `extract.py` file with the following arguments:

`model_file` - path to pre-trained ESM model

`fasta_file` - path to FASTA file on which to extract representations

`output_dr` - path to the directory wehere the representaions will be saved

`--repr_layers` - layers indices from which to extract representations (0 to num_layers, inclusive)

`--include` - specify which representations to return:

               `per_tok` includes the full sequence, with an embedding per amino acid (seq_len x hidden_dim).
               
               `mean` includes the embeddings averaged over the full sequence, per layer.
               
               `bos` includes the embeddings from the beginning-of-sequence token. (NOTE: Don't use with the pre-trained models)
