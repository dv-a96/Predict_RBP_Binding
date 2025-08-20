import os
from datetime import datetime
from tensorflow.keras.models import load_model
from model_utilities import correlation_coefficient_loss, gaussian_nll,mse_from_mu,mae_from_mu
from train_test_utilities import *
from data_processing import *
from naming_utilities import create_model_name
from scipy.stats import pearsonr,spearmanr
from logger_utils import create_logger

base_dir = "Models"
checkpoint_dir = os.path.join(base_dir, "Checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
EVAL_FOLDER = 'Evaluation'


CUSTOM_OBJECTS = {"correlation_coefficient_loss":correlation_coefficient_loss,
                  "gaussian_nll":gaussian_nll,
                  "mse_from_mu":mse_from_mu,
                  "mae_from_mu":mae_from_mu}


def add_preds_to_eval_file(preds, model_name):
    eval_file_path = os.path.join(EVAL_FOLDER, f'summ.csv')
    os.makedirs(EVAL_FOLDER, exist_ok=True)
    if not os.path.exists(eval_file_path):
        pd.DataFrame({model_name: preds}).to_csv(eval_file_path, index=False)
    else:
         
        data = pd.read_csv(eval_file_path)
        data[model_name] = preds
        data.to_csv(eval_file_path, index=False)

def compare_models_in_folder(folder="/home/dsi/lubosha/Predict_RBP_Binding/Models/Checkpoints/esm_cnn"):
    logger = create_logger(f'scaling')
    models = [f for f in os.listdir(folder) if f.endswith('.keras')]
    model_names = [f.rsplit("_", 2)[:-2][0]for f in models]
    summary_data = 'Evaluation/summ.csv'
    data =pd.read_csv(summary_data)
    rnas, rbps, intensities = prepare_training_data(logger=logger,normalization_method='quantile')
    rnas = rna_one_hot(rnas)
    rbps = get_ESM_prot_vecs()
    rnas = rnas[:,:,:4] # keep only the first 4 bits.
    rbp_indice = rbps[[132]]  
    rbp_indice = rbp_indice[None,:]
    rbp_indice = rbp_indice.repeat(rnas.shape[0],axis=0)
    rbp_indice=rbp_indice.reshape(rbp_indice.shape[0],rbp_indice.shape[2])
    for model_name,model_path in zip(model_names,models):
        if model_name in data.columns:
            continue
        else:
            model = load_model(os.path.join(folder,model_path),custom_objects=CUSTOM_OBJECTS)
            preds = model.predict([rbp_indice,rnas],batch_size=4096)
            data[model_name] = preds.reshape(-1)
    data.to_csv(summary_data, index=False)



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
#compare_models_in_folder()
evaluate_model('only_rna',exclude_num=199)
