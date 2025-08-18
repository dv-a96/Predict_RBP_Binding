import pandas as pd
import os
import sys

def clean(embedded_proteins, pt_folder):
    not_deleted = []
    data = pd.read_csv(embedded_proteins)
    ids = data['protein_id']
    # Save ids that need to be exlucded from next embedding
    
        
    # add .pt ending
    #ids = ids.apply(lambda x: os.path.join(pt_folder,(x +'.pt')))
    # remove files from pt_folder
    for seq in ids:
        temp_path = os.path.join(pt_folder,f'{seq}.pt')
        if os.path.isfile(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                not_deleted.append(seq)
                print(e)
        else:
            print(f'no file {temp_path} found')
            not_deleted.append(seq)
    with open('miss_delete.txt','a') as f:
        for seq in not_deleted:
            f.write(f'{seq}\n')
    ids = data['protein_id']
    ids = ids[~ids.isin(not_deleted)]
    with open('embedded_sequence.txt','a') as f:
        for id in ids:
            f.write(f'{id}\n')
if __name__ == "__main__":
    clean(sys.argv[1],sys.argv[2])