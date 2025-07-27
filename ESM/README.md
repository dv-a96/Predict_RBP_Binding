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

For example to extrcat embeddings from sample FASTA file run the following script:
`python extract.py esm2_t6_8M_UR50D some_proteins.fasta ProteinEmbeding --repr_layers 0 5 6 --include mean per_tok`

This will create a directory named `ProteinEmbeding` and will use the pre-trained `esm2_t6_8M_UR50D` (wich has 6 layers) to extract the embedding representations from layers 0, 5 and 6 of the models. In the `PriteinEmbeding` dir you can find a `.pt` file for each protein from th FASTA file. The `.pt` file contain few representatins for each provtein (according to the `--include` parameter).

Now, in order to extract the mean embedding from the final layer of the model for each protein use the `emb_to_csv` file to get a `.csv` file with one representation for each protein (each row contain the protein id from the FASTA file following by the representation). The usage should include the following arguments:

* root-dir - the directory of the `.pt` files
* output-file - path to `.csv` output file
* last-layer - the number of the last layer of the model

  For example. To get the embedding file from our `ProteinEmbeding` directory you should run:
  `python emb_to_csv.py ProteinEmbeding/ embeding_esm2_t6_8M_UR50D.csv 6`
