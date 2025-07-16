"""This module is to train a given model"""

from train_test_utilities import *
from data_processing import *
from models import *
from logger_utils import create_logger

def train(model_name, training_type = 'K fold', K = 10, exclude_num = None, seed = 42):
    logger = create_logger(f'train_{model_name}_{training_type}')
    logger.info(f"Starting training for model: {model_name} with training type: {training_type}")
    # Load and prepare training data
    rnas, rbps, intensities = prepare_training_data(logger=logger)
    # Turn sequences into numerical representations
    pass

    # Load model
    pass

    # Data loading and preprocessing
    rbps_number = len(rbps)
    if exclude_num:
        testing_indices = exclude_indices(samples_num=rbps_number, exclude_num=exclude_num, random_state=seed)
    else: testing_indices = None
    # Split data into training and testing sets
    if training_type == 'K fold':
        train_folds, test_folds = split_k_fold(samples_num=rbps_number, K=K, excluded_indices=testing_indices, random_state=seed)
        logger.info(f"Data split into {K} folds for K-Fold cross-validation.")
        for fold_idx, (train_indices, test_indices) in enumerate(zip(train_folds, test_folds)):
            logger.info(f"Training fold {fold_idx + 1}/{K} with {len(train_indices)} training samples and {len(test_indices)} testing samples.")
            fold_rbps_train = rbps[train_indices]
            fold_rbps_validation = rbps[test_indices]
            ## column protein, rows rna binding values
            intensities_fold_train = intensities[:,train_indices]
            intensities_fold_test = intensities[:,test_indices]
            ### ProbeRating:
            if False:
                YTY=np.dot(intensities_fold_train.T, intensities_fold_train)    
                YTD=np.dot(intensities_fold_train.T, rnas)  
                # make input to nn
                rnaNum=YTD.shape[0]
                protTrainIN=fold_rbps_train.repeat(rnaNum, axis=0)
                protTestIN=fold_rbps_validation.repeat(rnaNum, axis=0)
                similarityTrainIN=YTY.reshape((-1,1), order='F') 
                protTrainNum=fold_rbps_train.shape[0]
                protTestNum=fold_rbps_validation.shape[0]
                rnaTrainIN=np.tile(YTD,(protTrainNum,1))  
                rnaTestIN=np.tile(YTD,(protTestNum,1))


                predictedSimilarity=network1.predict([protTestIN, rnaTestIN])
                predictedSimilarity=predictedSimilarity.reshape((rnaNum,-1),order='F')  
                # option1: Weighted sum reconstruction
                intensityPred=np.dot(intensityTrain, predictedSimilarity)
                # option2:  Moore-Penrose pseudo inverse reconstruction:
                intensityPred1=np.dot(np.linalg.pinv(intensityTrain.T), predictedSimilarity)

                network1, callbacksList = probe_rating()
                history=network1.fit([protTrainIN, rnaTrainIN], similarityTrainIN, batch_size=10, epochs=30, verbose=2, callbacks=callbacksList, validation_split=0.1, shuffle=True)
            # Train model on this fold
            pass


