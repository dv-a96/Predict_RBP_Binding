Similarity scores obtained using MMseq2.
Following the `https://github.com/soedinglab/mmseqs2/wiki` to install and run all protiens agiasnt each other.

we run the following command:
`mmseqs easy-search Predict_RBP_Binding/Data_sets/training_RBPs2.fa Predict_RBP_Binding/Data_sets/training_RBPs2.fa results.tsv tmp   --format-output "query,target,pident,alnlen,qlen,tlen,qcov,tcov,evalue,bits"`