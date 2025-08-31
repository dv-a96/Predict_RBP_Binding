# Predict_RBP_Binding

The code uses the following packages, which can be installed with pip install requirements.txt
esm==3.2.1.post1
keras==3.10.0
matplotlib==3.8.1
numpy==2.3.2
pandas==2.3.2
scikit_learn==1.4.2
scipy==1.16.1
seaborn==0.13.2
tensorflow==2.20.0
torch==2.8.0




## Objective
The project goal is to develop a model capable of predicting the binding values of RNA sequences to proteins. Given a set of RNAs, RBPs, and their binding values from RNA-Compete experiments, the objective is to train a model that can predict the binding values for new proteins (not yet tested in experiments) with new RNA sequences.

## Usage
### Input and Output
The program that runs the model takes as input two text files and an output file. The execution is done as follows:
`python main.py <ofile> <RBP> <RNA> <optional: Batch_size>`
- ofile – path to the output file where the predicted binding values are stored.
-	RBP – a text file containing amino acid sequences of proteins, one sequence per line.
-	RNA – a text file containing RNA nucleotide sequences, one sequence per line.
- Batch_size – optional int for predicting batch size (defaults to 4096). If running causes OOM (it shouldn't) reduce batch size to 1024.
The output of the program is a separate file for each protein (e.g., RBP201.txt, RBP202.txt, …), where each file contains the predicted binding values for the RNAs. The order of the predicted values corresponds to the order of the RNA sequences (each line in the output represents the predicted binding value for a single RNA).


**Download the models from: https://drive.google.com/file/d/1t_HAtE_assPcQJ8juxXf0mDEohAwHv-N/view?usp=sharing**
Unzip the folder at the working space. The output should be esm_cnn_Baseline_perProt\modelN.keras with 176 models.

