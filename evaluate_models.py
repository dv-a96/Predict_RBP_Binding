import os
from datetime import datetime
from tensorflow.keras.models import load_model
from model_utilities import correlation_coefficient_loss
from train_test_utilities import *
from data_processing import *
from naming_utilities import create_model_name
from scipy.stats import pearsonr,spearmanr
base_dir = "Models"
checkpoint_dir = os.path.join(base_dir, "Checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

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
    model = load_model(model_path,custom_objects = {"correlation_coefficient_loss":correlation_coefficient_loss})
    print("Model loaded successfully!")
    return model

def return_metrics(y_true, y_pred):
    r,p = pearsonr(y_pred,y_true)
    s_r, s_p = spearmanr(y_pred,y_true)
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
    logger = create_logger(f'predict_{model_name}')
    rnas, rbps, intensities = prepare_training_data(logger=logger,normalization_method='quantile')
    rbps_number = len(rbps)
    if exclude_num:
        test_indices = exclude_indices(samples_num=rbps_number, exclude_num=exclude_num, random_state=seed)
        if len(test_indices) == 0:
            raise ValueError('Error excluding testing indices')
        train_indices = list(set(range(rbps_number)).difference(set(test_indices)))

     # load model
    global checkpoint_dir
    checkpoint_dir = os.path.join(checkpoint_dir, create_model_name(model_name, mlp_layers))
    model = choose_model(checkpoint_dir)
    pred_intensities = []
    if model_name == "Combined_CNN":
        rbps,rnas,intensities= process_for_cnn(rbps,rnas,intensities)
        for indice in test_indices:
            rbp_indice = rbps[indice]
            rbp_indice = rbp_indice[None,:,:] 
            rbp_indice = rbp_indice.repeat(rnas.shape[0],axis=0)
            combined = np.concatenate((rbp_indice,rnas),axis=1)
            preds = model.predict(combined)
            pred_intensities.append(preds.reshape(-1))
    elif model_name == "separate_cnn":
        rbps,rnas,intensities= process_for_cnn(rbps,rnas,intensities)
        rnas = rnas[:,:,:4] # keep only the first 4 bits.
        for indice in test_indices:
            rbp_indice = rbps[indice]
            rbp_indice = rbp_indice[None,:,:] 
            rbp_indice = rbp_indice.repeat(rnas.shape[0],axis=0)
            preds = model.predict([rbp_indice,rnas])
            pred_intensities.append(preds.reshape(-1))
    elif model_name == "MLP":
        rbps,rnas,intensities= process_for_cnn(rbps,rnas,intensities)
        rnas = rnas[:,:,:4] # keep only the first 4 bits.
        rnas = rnas.reshape(rnas.shape[0],-1)
        rbps = rbps.reshape(rbps.shape[0],-1)
        
        for indice in test_indices:
            rbp_indice = rbps[indice]
            rbp_indice = rbp_indice[None,:] 
            rbp_indice = rbp_indice.repeat(rnas.shape[0],axis=0)
            combined = np.concatenate((rbp_indice,rnas),axis=1)
            preds = model.predict(combined)
            preds = preds.reshape(-1)
            pred_intensities.append(preds.reshape(-1))
    elif model_name == 'probe_rating':
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

    pred_intensities = np.array(pred_intensities).T
    true_intensities = intensities[:,test_indices]
    correlations = pearson_stats(true_intensities,pred_intensities)
    print(correlations)

evaluate_model('probe_rating')