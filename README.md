# Predict_RBP_Binding

##PreRequiermants
* Python 2.7 (also work for Python 3 with few updates in the syntax)
* numpy 1.15.2
* pandas 0.23.3
* scipy 1.1.0
* biopython 1.72
* gensim 3.6.0
* memory-profiler 0.54.0
* guppy 0.1.10




## Using other models:
Im not realy read off all this papers but some sound interesting:
1. https://www.nature.com/articles/s41467-019-12920-0 not realy sure what it predicts, but something rna interactions and their effect on protein sequence. we can use this output as additional feature, and it wasnt trained on invivo\vitro data but the pdb so another level of learining.
2. using DSSP to predict alpha helix, pockets, beta sheets
3. using APBS to predict electrostatic charge.
**this values should be checked with corelations to the rna intensities**

