import os
from datetime import datetime
from tensorflow.keras.models import load_model
from model_utilities import *
from train_test_utilities import *
from data_processing import *
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_distances
import re
base_dir = "Models"
checkpoint_dir = os.path.join(base_dir, "Checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
EVAL_FOLDER = 'Evaluation'
BATCH_SIZE = 4096


def get_knn_emb_indices(prot_vec, K, df, metric="cosine"):
    """
    Return the indices of the K closest protein embeddings to a given vector.

    Parameters
    ----------
    prot_vec : array-like (list, Series, or numpy array)
        Protein embedding vector to compare against.
    K : int
        Number of nearest neighbors to return.
    df : pd.DataFrame
        DataFrame where each row is a protein embedding.
    metric : str
        Distance metric to use ("cosine" or "euclidean").

    Returns
    -------
    indices : list
        List of indices of the K nearest proteins.
    """
    # Convert DataFrame to numpy array
    X = np.array(df) 
    # Ensure prot_vec is a numpy array (no pandas index alignment issues)
    prot_vec = np.asarray(prot_vec)
    if prot_vec.ndim > 1:
        prot_vec = prot_vec.flatten()
    # Compute distances
    if metric == "cosine":
        dists = cosine_distances([prot_vec], X)[0]  
    elif metric == "euclidean":
        dists = np.linalg.norm(X - prot_vec, axis=1)
    else:
        raise ValueError("metric must be 'cosine' or 'euclidean'")
    # Get indices of K nearest neighbors
    nearest_indices = np.argsort(dists)[:K]
    return nearest_indices.tolist()


def knn_models_predict(test_dataset, models, numer_of_samples=None,batch_size=BATCH_SIZE):
    """Given a test dataset and a list of models, predict the outputs for each model and average them.

    Args:
        test_dataset (tf.data.Dataset): dataset to predict on.
        models (list): list of keras models.

    Returns:
        np.array: averaged predictions.
    """
    
    models = [load_model(m,custom_objects=CUSTOM_OBJECTS) for m in models if isinstance(m, str) and os.path.exists(m)] + [m for m in models if not isinstance(m, str)]
    K = len(models)
    if numer_of_samples is not None:
        predictions = np.zeros(shape=(numer_of_samples, K))
        for i, model in enumerate(models):
            preds = model.predict(test_dataset,batch_size=batch_size)
            predictions[:,i] = preds.reshape(-1)
    else:
        predictions = []
        for i, model in enumerate(models):
            preds = model.predict(test_dataset,batch_size=batch_size)
            predictions.append(preds.reshape(-1))
        predictions = np.array(predictions).T
    mean_preds = predictions.mean(axis=1).reshape(-1)
    return mean_preds





def prot_model_to_idx(indices, folder = 'esm_cnn_Baseline_perProt'):
    """search for a model for a given protein index or list of indices in a folder.
    The model files should have the format *_prot{index}_*.keras to be found.

    Args:
        indices (int\list): indice to look for.
        folder (str, optional): path. Defaults to 'Models/Checkpoints/esm_cnn_Baseline_perProt'.

    Returns:
        models: path to models with indices
    """
    if isinstance(indices, int):
        indices = [indices]
    models = os.listdir(folder) 
    idx2file = {int(m.group(1)): f for f in models if (m := re.search(r'_prot(\d+)_', f))}
    picked_models = [idx2file[i] for i in indices if i in idx2file]
    full_paths = [os.path.join(folder, m) for m in picked_models]
    return full_paths

def evaluate_perprot_knn_models(K=5):
    rnas, rbps, intensities, sample_w_np, edges_np, bin_w_np, test_rnas, test_intensities = prepare_baseline_data(if_remove_rna_duplicates=True,test=True)
    test_intensities = np.array(test_intensities)
    rbps = get_ESM_prot_vecs()
    test_rnas = rna_one_hot(test_rnas)
    test_rnas = test_rnas[:,:,:4] # keep only the first 4
    cluster_idx = get_clusteres_indices(cluster_id='all')
    all_rbps_indices = np.concatenate([val for val in cluster_idx.values()])
    train_indices = all_rbps_indices[all_rbps_indices <= 189]
    rbps_trains_vectors = rbps[train_indices]
    rbps_test_vectors = rbps[190:]
    factory = PairDatasetFactory(rbps_test_vectors,test_rnas,test_intensities)
    all_predictions = []
    for test_idx,test_vector in enumerate(rbps_test_vectors):
        knn_indices = get_knn_emb_indices(test_vector, K, rbps_trains_vectors, metric="cosine")
        print(f"Test RBP index: {test_idx+190}, KNN indices in train set: {train_indices[knn_indices]}")
        original_prots_indexes = train_indices[knn_indices]
        models = prot_model_to_idx(original_prots_indexes)
        if models == []:
            print(f"No models found for RBP {test_idx+190}, skipping.")
            all_predictions.append(np.zeros(test_rnas.shape[0]))
            continue
        models = [load_model(m,custom_objects=CUSTOM_OBJECTS) for m in models]
        predictions = np.zeros((test_rnas.shape[0], K))
        test_dataset = factory.make_train(batch_size=BATCH_SIZE, shuffle=False, prot_ids=[test_idx])
        for i, model in enumerate(models):
            preds = model.predict(test_dataset,batch_size=BATCH_SIZE)
            predictions[:,i] = preds.reshape(-1)
        mean_preds = predictions.mean(axis=1).reshape(-1)
        all_predictions.append(mean_preds)
        #print(f'correlation for RBP {test_idx+190}:', pearsonr(mean_preds, test_intensities[:,test_idx+190])[0])
    true_intensities = test_intensities[:,190:]
    return pearson_stats(true_intensities,np.array(all_predictions).T)[1]


def evaluate_base_line(model):
    
    rnas = pd.read_csv("Data_sets/training_seqs.txt",header=None)
    rbps = get_ESM_prot_vecs()
    rnas = rna_one_hot(rnas)
    rnas = rnas[:,:,:4] # keep only the first 4 bits.
    factory = PairDatasetFactory(rbps,rnas,place_on_cpu=True)
    rbp_test_indices = list(range(190,200))
    rna_test_indices = list(range(100000,rnas.shape[0]))
    

    test_dataset_all = factory.make_train(batch_size=BATCH_SIZE, shuffle=False, prot_ids=rbp_test_indices, rna_ids=rna_test_indices)
    model = load_model(model,custom_objects=CUSTOM_OBJECTS)
    intensities = np.array(pd.read_csv("Data_sets/training_data2.txt.gz",header=None,sep="\t"))
    factory_preds = model.predict(test_dataset_all,batch_size=BATCH_SIZE)
    factory_preds = factory_preds.reshape(-1)
    true_intensities = intensities[100000:,rbp_test_indices]
    
    factory_preds = factory_preds.reshape(len(rbp_test_indices), -1).T
    
    correlations = pearson_stats(true_intensities,factory_preds)
    print('\nstats factory:',correlations)
    

def add_preds_to_eval_file(preds, model_name):
    eval_file_path = os.path.join(EVAL_FOLDER, f'summ.csv')
    os.makedirs(EVAL_FOLDER, exist_ok=True)
    if not os.path.exists(eval_file_path):
        pd.DataFrame({model_name: preds}).to_csv(eval_file_path, index=False)
    else:
         
        data = pd.read_csv(eval_file_path)
        data[model_name] = preds
        data.to_csv(eval_file_path, index=False)


def evaluate_pident_knn(pident='Data_sets/pident.csv', K =3 , test_indice = None):
    pass


def evalute_cluster_models(folder, cluster_id,seed=42, model_constrain = None):
    """Givan a folder with models, evalaute them on a specific cluster.
    Outputs a csv file with predictions for each model and each RBP in the validation and test set.
    With prefix 'predictions_' and 'labels_' for true intensities.
    If validation/test sets are bigger than 10 RBPS, only 10 random RBPS are taken.
    If model_contrain is given, only models with this string in their name are evaluated.
    Args:
        folder (str): path
        cluster_id (str/int): 1-3, 'all'
        seed (int, optional): seed used to split the orignal data. Defaults to 42.
        model_constrain (str, optional): if given, only models with this string in their name are evaluated. Defaults to None.
    """
    models = [os.path.join(folder,f) for f in os.listdir(folder) if f.endswith('.keras')]
    if model_constrain:
        print(f"Evaluating only models with '{model_constrain}' in their name.")
        models = [m for m in models if model_constrain in m]
    model_names = [m.split("/")[-1].rsplit("_", 2)[:-2][0] for m in models]
    summary_path = os.path.join('Evaluation','Clusters',cluster_id)
    os.makedirs(summary_path, exist_ok=True)
    summary_data = os.path.join(summary_path, 'predictions_summary.csv')
    if os.path.exists(summary_data):
        # load existing file
        data = pd.read_csv(summary_data)
    else:
        # create empty DataFrame and save it
        data = pd.DataFrame()
    rnas, rbps, intensities,sample_w, edges, bin_w = prepare_training_data()
    rnas = rna_one_hot(rnas)
    rbps = get_ESM_prot_vecs()
    rnas = rnas[:,:,:4] # keep only the first 4 bits.
    factory = PairDatasetFactory(rbps,rnas,intensities,place_on_cpu=True)
    cluster_idx = get_clusteres_indices(cluster_id=cluster_id)
    rbps_train_indices, rbps_validation_indices, rbps_test_indices = split_rbs_to_train_val_test(cluster_idx, val_ratio=0.2, test_ratio=0.1, random_state=seed)
    def predict_and_add_to_summary(model_path, model_name, indice_type, indices):
        """given model path, name, indice type (val/test) and indice, predict and add to summary file.

        Args:
            model_path (str): path to model
            model_name (str): model name
            indice_type (str): val/test
            indices (list): rbp indices list ints.
        """
        if model_name in data.columns:
            return
        model = load_model(model_path,custom_objects=CUSTOM_OBJECTS)
        for indice in indices:
            if f'predictions_{indice_type}_{model_name}_{indice}' in data.columns:
                continue
            print(f"Predicting for model {model_name}, RBP {indice} ({indice_type})")
            #single_data = factory.make_train(batch_size=BATCH_SIZE, shuffle=False, prot_ids=[indice])
            rbp_indice = rbps[indice]  
            rbp_indice = rbp_indice[None,:]
            rbp_indice = rbp_indice.repeat(rnas.shape[0],axis=0)
            pred = model.predict([rbp_indice,rnas],batch_size=BATCH_SIZE)
            if any(sub in model_name for sub in ['asymmetric_t','gaussian','asymmetric_gaussian','asymmetric_laplace']):
                pred = pred[:,:1]
            pred = pred.reshape(-1)
            data[f'predictions_{indice_type}_{model_name}_{indice}'] = pred
            if f'labels_{indice}' not in data.columns:
                data[f'labels_{indice}'] = intensities[:,indice]
    if len(rbps_validation_indices) > 10:
        np.random.seed(seed)
        rbps_validation_indices = np.random.choice(rbps_validation_indices, size=10, replace=False)
    if len(rbps_test_indices) > 10:
        np.random.seed(seed)
        rbps_test_indices = np.random.choice(rbps_test_indices, size=10, replace=False)
    for model_path, model_name in zip(models, model_names):
        predict_and_add_to_summary(model_path, model_name, 'val', rbps_validation_indices)
        predict_and_add_to_summary(model_path, model_name, 'test', rbps_test_indices)
            
    data.to_csv(summary_data, index=False)





def evaluate_cluster_predictions(cluster_id, norm_method, model_constrain = None, only_ensmeble = False):
    data_path = os.path.join('Evaluation','Clusters',str(cluster_id),'predictions_summary.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No predictions summary found for cluster {cluster_id}. Please run evalute_cluster_models first.")
    data = pd.read_csv(data_path)
    if norm_method not in ['quantile','meannorm','robust','zscore','quantile_meannorm']:
        raise ValueError(f"Unknown normalization method {norm_method}. Choose from ['quantile','meannorm','robust','zscore','quantile_meannorm']")
    columns_with_norm = [col for col in data.columns if norm_method in col or col.startswith('labels_')]
    # Extarct val and test indices from columns names
    val_pattern = re.compile(r'predictions_(val)_(.+?)_(\d+)$')
    validation_indices = [val_pattern.match(col).group(3) for col in columns_with_norm if val_pattern.match(col)]
    test_pattern = re.compile(r'predictions_(test)_(.+?)_(\d+)$')
    test_indices = [test_pattern.match(col).group(3) for col in columns_with_norm if test_pattern.match(col)]
    validation_indices = list(set(validation_indices))
    test_indices = list(set(test_indices))
    # Get model names
    model_pattern = re.compile(r'predictions_(val|test)_(.+?)_(\d+)$')
    model_names = list(set([model_pattern.match(col).group(2) for col in columns_with_norm if model_pattern.match(col)]))
    data_path = os.path.join('Evaluation','Clusters',str(cluster_id),f'{norm_method}')
    os.makedirs(data_path, exist_ok=True)
    if model_constrain:
        model_names = [m for m in model_names if model_constrain in m]
        data_path = os.path.join(data_path, model_constrain)
        os.makedirs(data_path, exist_ok=True)
    # Get rna train, val, test indices
    ##########
    indices_data = pd.read_csv(os.path.join('Data_sets','Index_sets',f'{norm_method}_rnas_indices.csv'))
    rna_validation_indices = indices_data[f'cluster{cluster_id}_validation'].dropna()
    rna_test_indices = indices_data[f'cluster{cluster_id}_test'].dropna()
    
    rna_validation_data = data.iloc[rna_validation_indices]
    rna_test_data = data.iloc[rna_test_indices]
    validation_test_indices = rna_validation_indices.tolist() + rna_test_indices.tolist()
    rna_val_test_data = data.iloc[validation_test_indices]
    ##########
    # For every model, calculate pearson r for val and test sets and their mean.
    models_val_corrs = {}
    models_test_corrs = {}
    models_test_val_test_indices = {} # evaluate the test set on the validation and test indices
    models_test_all_indices = {} # evaluate the test set on all indices
    
    def evaluate_cluster_predictions_as_ensemble(test_indices, models):
        # pick the mean predictions of all model combinations and evaluate their pearson r corelation.
        with open(os.path.join(data_path,'ensemble_dict.txt'),'w') as f:
            for i,model in enumerate(models):
                f.write(f"Model {i}: {model}\n")
        num_models = len(models)
        # Enumrate models
        from itertools import combinations
        
        for k in range(2, num_models+1):
            
            df = pd.DataFrame(index=test_indices + ['mean'])
            for combo in combinations(range(num_models), k):
                combo_models = [models[i] for i in combo]
                for indice in test_indices:
                    combo_pred_cols = [f'predictions_test_{model}_{indice}' for model in combo_models if f'predictions_test_{model}_{indice}' in data.columns]
                    if not combo_pred_cols:
                        continue
                    ensemble_preds = data[combo_pred_cols].mean(axis=1)
                    true_labels = data[f'labels_{indice}'].values
                    ensemble_corr = pearsonr(ensemble_preds, true_labels)[0]
                    df.loc[indice, f'Ensemble_{combo}'] = ensemble_corr
                df.loc['mean', f'Ensemble_{combo}'] = df.loc[test_indices, f'Ensemble_{combo}'].mean()
            df_path = os.path.join(data_path,f'ensemble_size_{k}.csv')
            df.to_csv(df_path)
    
    #evaluate_cluster_predictions_as_ensemble(test_indices, model_names)
    if only_ensmeble:
        return
    for model in model_names:
        val_cols = [col for col in columns_with_norm if f'predictions_val_{model}_' in col] + [f'labels_{val_indice}' for val_indice in validation_indices]
        test_cols = [col for col in columns_with_norm if f'predictions_test_{model}_' in col] + [f'labels_{test_indice}' for test_indice in test_indices]
        val_corr = rna_validation_data[val_cols].corr().loc[[f'labels_{idx}' for idx in validation_indices],[f'predictions_val_{model}_{idx}' for idx in validation_indices]].values.diagonal()
        test_corr = rna_test_data[test_cols].corr().loc[[f'labels_{idx}' for idx in test_indices],[f'predictions_test_{model}_{idx}' for idx in test_indices]].values.diagonal()
        val_test_corr = rna_val_test_data[test_cols].corr().loc[[f'labels_{idx}' for idx in test_indices],[f'predictions_test_{model}_{idx}' for idx in test_indices]].values.diagonal()
        all_test_corr = data[test_cols].corr().loc[[f'labels_{idx}' for idx in test_indices],[f'predictions_test_{model}_{idx}' for idx in test_indices]].values.diagonal()
        models_val_corrs[model] = np.append(val_corr, np.mean(val_corr))
        models_test_corrs[model] = np.append(test_corr, np.mean(test_corr))
        models_test_val_test_indices[model] = np.append(val_test_corr, np.mean(val_test_corr))
        models_test_all_indices[model] = np.append(all_test_corr, np.mean(all_test_corr))
    val_summary = pd.DataFrame(models_val_corrs, index = validation_indices + ['mean'])
    test_summary = pd.DataFrame(models_test_corrs, index = test_indices + ['mean'])
    test_val_summary = pd.DataFrame(models_test_val_test_indices, index = test_indices + ['mean'])
    all_test_summary = pd.DataFrame(models_test_all_indices, index = test_indices + ['mean'])
    manual = ''
    val_summary.to_csv(os.path.join(data_path,f'validation_summary{manual}.csv'))
    test_summary.to_csv(os.path.join(data_path,f'test_summary{manual}.csv'))
    test_val_summary.to_csv(os.path.join(data_path,f'test_and_validation_summary{manual}.csv'))
    all_test_summary.to_csv(os.path.join(data_path,f'all_test_summary{manual}.csv'))
    
    print(f"Summary saved to {os.path.join('Evaluation','Clusters',str(cluster_id),norm_method,'validation|test_summary.csv')}")
        


 
               

def create_indices():
    data_path = 'Data_sets/Index_sets'
    os.makedirs(data_path, exist_ok=True)
    norm_methods = ['quantile','meannorm']
    for norm_method in norm_methods:
        rnas, rbps, intensities,sample_w, edges, bin_w = prepare_training_data(normalization_method=norm_method)
        rbp_splits = {}
        rna_splits = {}
        for id in [1,2,3,'all']:
            cluster_idx = get_clusteres_indices(cluster_id=id)
            rbps_train_indices, rbps_validation_indices, rbps_test_indices = split_rbs_to_train_val_test(cluster_idx, val_ratio=0.2, test_ratio=0.1, random_state=42)
            rbp_splits[f"cluster{id}_train"] = pd.Series(rbps_train_indices).astype(int)
            rbp_splits[f"cluster{id}_validation"] = pd.Series(rbps_validation_indices).astype(int)
            rbp_splits[f"cluster{id}_test"] = pd.Series(rbps_test_indices).astype(int)
            all_rbp_indices = np.concatenate([rbps_train_indices, rbps_validation_indices, rbps_test_indices])
            rna_train_indices, rna_validation_indices, rna_test_indices = stratified_split_multi(intensities.T[all_rbp_indices],random_state=42,val_size=0.2)
            rna_splits[f"cluster{id}_train"] = pd.Series(rna_train_indices).astype(int)
            rna_splits[f"cluster{id}_validation"] = pd.Series(rna_validation_indices).astype(int)
            rna_splits[f"cluster{id}_test"] = pd.Series(rna_test_indices).astype(int)
        df_rbps = pd.DataFrame(rbp_splits)
        df_rnas = pd.DataFrame(rna_splits)
        df_rbps.to_csv(os.path.join(data_path,f'{norm_method}_rbps_indices.csv'), index=False)
        df_rnas.to_csv(os.path.join(data_path,f'{norm_method}_rnas_indices.csv'), index=False)




def choose_model(folder_path):
    """Let the user pick which model to load.

    Args:
        folder_path (str): path to folder with .keras models

    Returns:
        keras.Model: ML model.
    """
    keras_files = [f for f in os.listdir(folder_path) if f.endswith('.keras')]
    
    if not keras_files:
        print("No .keras models found in the given folder.")
        return None
    
    # Sort by timestamp extracted from filename (descending)
    def extract_timestamp(filename):
        try:
            timestamp_str = filename.rsplit("_", 2)[-2] + "_" + filename.rsplit("_", 2)[-1].replace(".keras", "")
            return datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            return datetime.min  # put invalid formats at the end
    
    keras_files.sort(key=extract_timestamp, reverse=True)
    
    # Print options
    print("Available .keras models (newest first):")
    for idx, file in enumerate(keras_files, start=1):
        print(f"{idx}. {file}")
    
    # Get user choice
    while True:
        try:
            choice = int(input(f"Select model (1-{len(keras_files)}): "))
            if 1 <= choice <= len(keras_files):
                chosen_file = keras_files[choice - 1]
                break
            else:
                print("Invalid choice. Please select a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Load and return model
    model_path = os.path.join(folder_path, chosen_file)
    print(f"Loading model: {chosen_file}")
    model = load_model(model_path,custom_objects = CUSTOM_OBJECTS)
    print("Model loaded successfully!")
    model_name = chosen_file.rsplit("_", 2)[:-2][0]  # Extract model name from filename
    return model,model_name


def pearson_stats(A, B):
    """
    A, B: numpy arrays of shape (k, m)
    Returns:
        col_rs: list of Pearson r for each column
        mean_r: average of column-wise r's
        flat_r: Pearson r for flattened arrays
    """
    assert A.shape == B.shape, "Arrays must have the same shape"
    k, m = A.shape
    
    # 1. Column-wise Pearson correlations
    col_rs = [pearsonr(A[:, j], B[:, j])[0] for j in range(m)]
    # 2. Mean of column-wise correlations
    mean_r = np.mean(col_rs)
    # 3. Flattened correlation
    flat_r = pearsonr(A.ravel(), B.ravel())[0]
    return col_rs, mean_r, flat_r

def evaluate_model(model_name, exclude_num = 20, seed = 42, batch_size = 2048, mlp_layers=[64]):
    # Init and get data
    rnas, rbps, intensities = prepare_training_data(normalization_method='quantile')
    rbps_number = len(rbps)
    

     # load model
    global checkpoint_dir
    checkpoint_dir = os.path.join(checkpoint_dir, create_model_name(model_name, mlp_layers))
    model,model_name = choose_model(checkpoint_dir)
    pred_intensities = []
    if "combined_cnn" in model_name.lower():
        rbps = rbp_one_hot(rbps)
        rnas = rna_one_hot(rnas)
        for indice in test_indices:
            rbp_indice = rbps[indice]
            rbp_indice = rbp_indice[None,:,:] 
            rbp_indice = rbp_indice.repeat(rnas.shape[0],axis=0)
            combined = np.concatenate((rbp_indice,rnas),axis=1)
            preds = model.predict(combined)
            pred_intensities.append(preds.reshape(-1))
    elif "separate_cnn" in model_name.lower():
        rbps = rbp_one_hot(rbps)
        rnas = rna_one_hot(rnas)
        rnas = rnas[:,:,:4] # keep only the first 4 bits.
        for indice in test_indices:
            rbp_indice = rbps[indice]
            rbp_indice = rbp_indice[None,:,:] 
            rbp_indice = rbp_indice.repeat(rnas.shape[0],axis=0)
            preds = model.predict([rbp_indice,rnas])
            pred_intensities.append(preds.reshape(-1))
    elif "mlp" in model_name.lower():
        rbps = rbp_one_hot(rbps)
        rnas = rna_one_hot(rnas)
        rnas = rnas[:,:,:4] # keep only the first 4 bits.
        rnas = rnas.reshape(rnas.shape[0],-1)
        rbps = rbps.reshape(rbps.shape[0],-1)
        
        for indice in test_indices:
            rbp_indice = rbps[indice]
            rbp_indice = rbp_indice[None,:] 
            rbp_indice = rbp_indice.repeat(rnas.shape[0],axis=0)
            combined = np.concatenate((rbp_indice,rnas),axis=1)
            preds = model.predict(combined,batch_size=batch_size)
            preds = preds.reshape(-1)
            pred_intensities.append(preds.reshape(-1))
    elif 'esm_cnn' in model_name.lower():
        rnas = rna_one_hot(rnas)
        rbps = get_ESM_prot_vecs()
        rnas = rnas[:,:,:4] # keep only the first 4 bits.
        #NOTE : change train to test!
        for indice in train_indices:
            rbp_indice = rbps[train_indices]  
            rbp_indice = rbp_indice[None,:]
            rbp_indice = rbp_indice.repeat(rnas.shape[0],axis=0)
            rbp_indice=rbp_indice.reshape(rbp_indice.shape[0],rbp_indice.shape[2])
            preds = model.predict([rbp_indice,rnas],batch_size=batch_size)
            if model_name.lower() == "esm_cnn_guas":
                preds = preds[:,:1]
            pred_intensities.append(preds.reshape(-1))
    elif 'only_rna' in model_name.lower():
        rnas = rna_one_hot(rnas)
        rnas = rnas[:,:,:4]
        preds = model.predict(rnas,batch_size=batch_size)
        pred_intensities.append(preds.reshape(-1))
    elif 'probe_rating' in model_name.lower():
        rbps = get_ESM_prot_vecs()
        rnas = get_ESM_rna_vecs()
        fold_rbps_validation = rbps[test_indices]
        intensities_fold_train = intensities[:,train_indices]
        intensities_fold_validation = intensities[:,test_indices]
        YTD=np.dot(intensities_fold_train.T, rnas)  
        # make input to nn
        rnaNum=YTD.shape[0]
        protTestIN=fold_rbps_validation.repeat(rnaNum, axis=0)
        protTestNum=fold_rbps_validation.shape[0]
        rnaTestIN=np.tile(YTD,(protTestNum,1))

        
        predictedSimilarity=model.predict([protTestIN, rnaTestIN])
        predictedSimilarity=predictedSimilarity.reshape((rnaNum,-1),order='F')  
        # option1: Weighted sum reconstruction
        intensityPred=np.dot(intensities_fold_train, predictedSimilarity)
        # option2:  Moore-Penrose pseudo inverse reconstruction:
        intensityPred1=np.dot(np.linalg.pinv(intensities_fold_train.T), predictedSimilarity)
        print(pearson_stats(intensityPred, intensities_fold_validation))
        print(pearson_stats(intensityPred1, intensities_fold_validation))
        return
    add_preds_to_eval_file(np.array(pred_intensities).T, model_name)
    # pred_intensities = np.array(pred_intensities).T
    
    # true_intensities = intensities[:,test_indices]
    # correlations = pearson_stats(true_intensities,pred_intensities)
    # print(correlations)
#compare_models_in_folder("Predict_RBP_Binding/Models/Checkpoints/esm_loss_bin")
#evaluate_model('Only_RNA_sec',exclude_num=199)
#evaluate_base_line("Models/Checkpoints/esm_cnn_Baseline/esm_cnn_regression_v1_quantile_MSE_Adam_2025-08-27_16-46-24.keras")
# evalute_cluster_models('Models/Checkpoints/esm_cnn_cluster_all',
#                         cluster_id='all',model_constrain='v')
# evaluate_cluster_predictions('all', norm_method='quantile',model_constrain='v',only_ensmeble=False)
#