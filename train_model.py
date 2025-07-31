"""This module is to train a given model"""

from train_test_utilities import *
from data_processing import *
from models import *
from logger_utils import create_logger
def train_k_fold(model_name, K = 10, exclude_num = None, seed = 42, batch_size = 512, epochsNum = 1):
    logger = create_logger(f'train_{model_name}_{K}-fold')
    logger.info(f"Starting training for model: {model_name} with training type: {K} - fold")
    # Load and prepare training data
    rnas, rbps, intensities = prepare_training_data(logger=logger)
    if model_name == "Combined_CNN":
        rbps,rnas,intensities= process_for_cnn(rbps,rnas,intensities)
        combined_cnn, call_backs = Combined_CNN(input_shape=(rbps.shape[1]+rnas.shape[1],20))

    
            ### CNN:
    rbps_number = len(rbps)
    if exclude_num:
        testing_indices = exclude_indices(samples_num=rbps_number, exclude_num=exclude_num, random_state=seed)
    else: testing_indices = None
    # Split data into training and testing sets

    train_folds, test_folds = split_k_fold(samples_num=rbps_number, k=K, excluded_indices=testing_indices, random_state=seed)
    logger.info(f"Data split into {K} folds for K-Fold cross-validation.")
    for fold_idx, (train_indices, test_indices) in enumerate(zip(train_folds, test_folds)):
        logger.info(f"Training fold {fold_idx + 1}/{K} with {len(train_indices)} training samples and {len(test_indices)} testing samples.")
        fold_rbps_train = rbps[train_indices]
        fold_rbps_validation = rbps[test_indices]
        ## column protein, rows rna binding values
        intensities_fold_train = intensities[:,train_indices]
        intensities_fold_validation = intensities[:,test_indices]
        
        ### CNN:
        if model_name == 'Combined_CNN':
            train_ds = RBP_RNA_PairDataset(fold_rbps_train, rnas, intensities=intensities_fold_train)
            train_ds = train_ds.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            val_ds = RBP_RNA_PairDataset(fold_rbps_validation, rnas, intensities=intensities_fold_validation)
            val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            steps_per_epoch = fold_rbps_train.shape[0]*rnas.shape[0] // batch_size
            val_steps = fold_rbps_validation.shape[0] * rnas.shape[0]  // batch_size
            combined_cnn.fit(train_ds,validation_data=val_ds,epochs=epochsNum,callbacks=call_backs,steps_per_epoch=steps_per_epoch,validation_steps=val_steps)
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
    
def train_held_out_test(model_name, exclude_num = 20, seed = 42, batch_size = 512, epochsNum = 1):
    logger = create_logger(f'train_{model_name}_heldout')
    logger.info(f"Starting training for model: {model_name} with training type: heldout")
    # Load and prepare training data
    rnas, rbps, intensities = prepare_training_data(logger=logger)
    rbps_number = len(rbps)
    if exclude_num:
        test_indices = exclude_indices(samples_num=rbps_number, exclude_num=exclude_num, random_state=seed)
        if len(test_indices) == 0:
            raise ValueError('Error excluding testing indices')
        train_indices = list(set(range(rbps_number)).difference(set(test_indices)))
    if model_name == "Combined_CNN":
        rbps,rnas,intensities= process_for_cnn(rbps,rnas,intensities)
        combined_cnn, call_backs = Combined_CNN(input_shape=(rbps.shape[1]+rnas.shape[1],20))

    train_ds = RBP_RNA_PairDataset(rbps[train_indices], rnas, intensities=intensities[:,train_indices])
    train_ds = train_ds.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = RBP_RNA_PairDataset(rbps[test_indices], rnas, intensities=intensities[:,test_indices])
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    steps_per_epoch = rbps[train_indices].shape[0]*rnas.shape[0] // batch_size
    val_steps = rbps[test_indices].shape[0] * rnas.shape[0]  // batch_size
    combined_cnn.fit(train_ds,validation_data=val_ds,epochs=epochsNum,callbacks=call_backs,steps_per_epoch=steps_per_epoch,validation_steps=val_steps)

if __name__ =="__main__":
    #train_k_fold("Combined_CNN")
    train_held_out_test("Combined_CNN")