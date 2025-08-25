"""This module is to train a given model"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from train_test_utilities import *
from data_processing import *
from models import *

import time


from logger_utils import create_logger
from model_utilities import LOSSES, init_checkpoint_and_tensorboard
# def train_k_fold(model_name, K = 10, exclude_num = None, seed = 42, batch_size = 512, epochsNum = 1):
#     logger = create_logger(f'train_{model_name}_{K}-fold')
#     logger.info(f"Starting training for model: {model_name} with training type: {K} - fold")
#     # Load and prepare training data
#     rnas, rbps, intensities = prepare_training_data(logger=logger)
#     if model_name == "Combined_CNN":
#         rbps,rnas,intensities= process_for_cnn(rbps,rnas,intensities)
#         combined_cnn, call_backs = Combined_CNN(input_shape=(rbps.shape[1]+rnas.shape[1],20))

    
#             ### CNN:
#     rbps_number = len(rbps)
#     if exclude_num:
#         testing_indices = exclude_indices(samples_num=rbps_number, exclude_num=exclude_num, random_state=seed)
#     else: testing_indices = None
#     # Split data into training and testing sets

#     train_folds, test_folds = split_k_fold(samples_num=rbps_number, k=K, excluded_indices=testing_indices, random_state=seed)
#     logger.info(f"Data split into {K} folds for K-Fold cross-validation.")
#     for fold_idx, (train_indices, test_indices) in enumerate(zip(train_folds, test_folds)):
#         logger.info(f"Training fold {fold_idx + 1}/{K} with {len(train_indices)} training samples and {len(test_indices)} testing samples.")
#         fold_rbps_train = rbps[train_indices]
#         fold_rbps_validation = rbps[test_indices]
#         ## column protein, rows rna binding values
#         intensities_fold_train = intensities[:,train_indices]
#         intensities_fold_validation = intensities[:,test_indices]
        
#         ### CNN:
#         if model_name == 'Combined_CNN':
#             train_ds = RBP_RNA_Combined_Dataset(fold_rbps_train, rnas, intensities=intensities_fold_train)
#             train_ds = train_ds.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
#             val_ds = RBP_RNA_Combined_Dataset(fold_rbps_validation, rnas, intensities=intensities_fold_validation)
#             val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
#             steps_per_epoch = fold_rbps_train.shape[0]*rnas.shape[0] // batch_size
#             val_steps = fold_rbps_validation.shape[0] * rnas.shape[0]  // batch_size
#             combined_cnn.fit(train_ds,validation_data=val_ds,epochs=epochsNum,callbacks=call_backs,steps_per_epoch=steps_per_epoch,validation_steps=val_steps)
#         ### ProbeRating:
#         if False:
#             YTY=np.dot(intensities_fold_train.T, intensities_fold_train)    
#             YTD=np.dot(intensities_fold_train.T, rnas)  
#             # make input to nn
#             rnaNum=YTD.shape[0]
#             protTrainIN=fold_rbps_train.repeat(rnaNum, axis=0)
#             protTestIN=fold_rbps_validation.repeat(rnaNum, axis=0)
#             similarityTrainIN=YTY.reshape((-1,1), order='F') 
#             protTrainNum=fold_rbps_train.shape[0]
#             protTestNum=fold_rbps_validation.shape[0]
#             rnaTrainIN=np.tile(YTD,(protTrainNum,1))  
#             rnaTestIN=np.tile(YTD,(protTestNum,1))


#             predictedSimilarity=network1.predict([protTestIN, rnaTestIN])
#             predictedSimilarity=predictedSimilarity.reshape((rnaNum,-1),order='F')  
#             # option1: Weighted sum reconstruction
#             intensityPred=np.dot(intensityTrain, predictedSimilarity)
#             # option2:  Moore-Penrose pseudo inverse reconstruction:
#             intensityPred1=np.dot(np.linalg.pinv(intensityTrain.T), predictedSimilarity)

#             network1, callbacksList = probe_rating()
#             history=network1.fit([protTrainIN, rnaTrainIN], similarityTrainIN, batch_size=10, epochs=30, verbose=2, callbacks=callbacksList, validation_split=0.1, shuffle=True)
#         # Train model on this fold
#         pass
    
def train_held_out_test(model_name, exclude_num = 20, seed = 42, batch_size = 512, epochsNum = 50, mlp_layers=[64],
                        loss_idx=1,opt_idx=2, plateauPatience=2, earlyStopPatience=5, model_type='regression',
                        if_sample_wieght=False, alpha=0.5, bins=20, if_clamp_by_percentile = False,
                        percentile = 99.5, normalization_method = 'quantile', cluster_id = None, remove_rna_dups = False):
    tf.keras.backend.clear_session()
    model_name = model_name.lower()
    
    # logger = create_logger(f'train_{full_model_name}_heldout')
    # logger.info(f"Starting training for model: {full_model_name} with training type: heldout")
    checkPtFile, tensorBoardDir = init_checkpoint_and_tensorboard(model_name,loss_key=loss_idx, opt_idx=opt_idx,
                                                                  model_type=model_type,mlp_layers=mlp_layers,
                                                                  if_sample_wieght=if_sample_wieght, bins=bins, alpha=alpha,
                                                                  if_clamp_by_percentile=if_clamp_by_percentile,percentile=percentile,
                                                                  cluster_id=cluster_id,remove_rna_dups=remove_rna_dups,
                                                                  norm_method=normalization_method)
    # Load and prepare training data
    rnas, rbps, intensities, sample_w_np, edges_np, bin_w_np = prepare_training_data(logger=None,normalization_method=normalization_method,
                                                                                        if_clamp_by_percentile=if_clamp_by_percentile,percentile=percentile,
                                                                                        if_sample_wieght=if_sample_wieght, alpha=alpha, bins=bins,
                                                                                        if_remove_rna_duplicates=remove_rna_dups)
    

    rbps_number = len(rbps)
    if cluster_id is not None:
        cluster_idx = get_clusteres_indices(cluster_id=cluster_id)
        rbps_train_indices, rbps_validation_indices, rbps_test_indices = split_rbs_to_train_val_test(cluster_idx, val_ratio=0.2, test_ratio=0.1, random_state=seed)
    elif exclude_num:
        test_indices = exclude_indices(samples_num=rbps_number, exclude_num=exclude_num, random_state=seed)
        if len(test_indices) == 0:
            raise ValueError('Error excluding testing indices')
        train_indices = list(set(range(rbps_number)).difference(set(test_indices)))
    all_rbp_indices = np.concatenate([rbps_train_indices, rbps_validation_indices, rbps_test_indices])
    rna_train_indices, rna_validation_indices, rna_test_indices = stratified_split_multi(intensities.T[all_rbp_indices],random_state=seed,val_size=0.2)
    if model_name == "combined_CNN":
        rbps = rbp_one_hot(rbps)
        rnas = rna_one_hot(rnas)
        model, call_backs = Combined_CNN(input_shape=(rbps.shape[1]+rnas.shape[1],20),mlp_layers=mlp_layers)
        
    elif model_name == "separate_cnn":
        rbps = rbp_one_hot(rbps)
        rnas = rna_one_hot(rnas)
        rnas = rnas[:,:,:4] # keep only the first 4 bits.
        model,call_backs = separate_cnn(protein_shape=(rbps.shape[1],20),rna_shape=(rnas.shape[1],4),mlp_layers=mlp_layers)
        
    elif model_name == "mlp":
        rbps = rbp_one_hot(rbps)
        rnas = rna_one_hot(rnas)
        rnas = rnas[:,:,:4] # keep only the first 4 bits.
        rnas = rnas.reshape(rnas.shape[0],-1)
        rbps = rbps.reshape(rbps.shape[0],-1)
        model,call_backs = MLP_Model(input_shape=(rnas.shape[1]+rbps.shape[1],),mlp_layers=mlp_layers)
        
  
    elif 'esm_cnn' in model_name.lower():
        
        rnas = rna_one_hot(rnas)
        rbps = get_ESM_prot_vecs()
        rnas = rnas[:,:,:4] # keep only the first 4 bits.
        if "minmax" in normalization_method:
            sigmoid_head = True
        else: sigmoid_head = False
        model,call_backs = build_ESM_CNN(prot_input=(rbps.shape[1],),rna_input=(41,4),
                                         check_points_folder=checkPtFile, tensorboard_folder=tensorBoardDir,
                                         loss_idx=loss_idx,opt_idx=opt_idx,
                                         plateauPatience=plateauPatience, 
                                         earlyStopPatience=earlyStopPatience,
                                         model_type=model_type,
                                         sigmoid_head=sigmoid_head)
        
        factory = PairDatasetFactory(rbps, rnas, intensities, place_on_cpu=True, sample_weight_array=sample_w_np)
        
    elif model_name == "only_rna":
        rnas = rna_one_hot(rnas)
        rnas = rnas[:,:,:4]
        model,call_backs = Only_RNA(rna_input=(41,4),loss_idx=loss_idx,
                                  check_points_folder=checkPtFile, tensorboard_folder=tensorBoardDir,
                                  opt_idx=opt_idx,plateauPatience=plateauPatience, earlyStopPatience=earlyStopPatience) 
        model.fit(rnas,intensities[:,train_indices],epochs=epochsNum,
                  callbacks=call_backs,validation_data=(rnas,intensities[:,train_indices]),batch_size=batch_size)
        model.save(checkPtFile)
        return
    elif model_name == 'probe_rating':
        rbps = get_ESM_prot_vecs()
        rnas = get_ESM_rna_vecs()
        # dictData=h5py.File('/home/dsi/lubosha/Predict_RBP_Binding/sample4 (1).mat', 'r')
        # label=np.array(dictData['Y']) 
        # label=label.T   
        # RNAf=np.array(dictData['D']) 
        # RNAf=RNAf.T     
        # protf=np.array(dictData['P'])
        # protf=protf.T   
        
        fold_rbps_train = rbps[train_indices]
        fold_rbps_validation = rbps[test_indices]
        ## column protein, rows rna binding values
        
        intensities_fold_train = intensities[:,train_indices]
        intensities_fold_validation = intensities[:,test_indices]
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

        network1, callbacksList = probe_rating(protein_vector_length=rbps.shape[1],rna_vector_length=rnas.shape[1])
        history=network1.fit([protTrainIN, rnaTrainIN], similarityTrainIN, batch_size=10, epochs=30, verbose=2, callbacks=callbacksList, validation_split=0.1, shuffle=True)

        # predictedSimilarity=network1.predict([protTestIN, rnaTestIN])
        # predictedSimilarity=predictedSimilarity.reshape((rnaNum,-1),order='F')  
        # # option1: Weighted sum reconstruction
        # intensityPred=np.dot(intensities_fold_train, predictedSimilarity)
        # # option2:  Moore-Penrose pseudo inverse reconstruction:
        # intensityPred1=np.dot(np.linalg.pinv(intensities_fold_train.T), predictedSimilarity)
        return
    
    
    train_ds = factory.make_train(batch_size=batch_size, shuffle=True,  prot_ids=rbps_train_indices,rna_ids=rna_train_indices)
    val_ds = factory.make_train(batch_size=batch_size, shuffle=False, prot_ids=rbps_validation_indices,rna_ids=rna_validation_indices)
   
    
    start = time.time()
    
    model.fit(train_ds, epochs=epochsNum, callbacks=call_backs, validation_data=val_ds)
    
    end = time.time()
    elapsed = end - start
    print(f"Training time  {elapsed:.2f} seconds")
   
    # model.save(checkPtFile)
    # Callbacks
    
    # for loss_key,loss_name in LOSSES.items():
    #     checkPtFile, tensorBoardDir = init_checkpoint_and_tensorboard(full_model_name,loss_key=loss_key,opt_idx=1)
    #     print(f'evaluating model {model_name} with loss {loss_name}')
    #     model,call_backs = ESM_CNN(prot_input=(rbps.shape[1],),rna_input=(41,4),loss_idx = loss_key, 
    #                                check_points_folder=checkPtFile, tensorboard_folder=tensorBoardDir,opt_idx=1)
    #     start = time.time()
    #     model.fit([rbp_indice,rnas],intensities[:,train_indices],epochs=5)
    #     end = time.time()
    #     elapsed = end - start
    #     print(f"Training time for 5 epoch: {elapsed:.2f} seconds")
        
    #     model.save(checkPtFile)

    #model.fit(train_ds,epochs=epochsNum,callbacks=call_backs,steps_per_epoch=steps_per_epoch)
    
         


"""
Check memory out of alocation

batch_size = 128
    while True:
        try:
            print(f"Trying batch_size={batch_size}")
            model.fit(train_ds.batch(batch_size).take(1))  # just 1 step to test allocation
            batch_size *= 2   # increase
        except tf.errors.ResourceExhaustedError:
            print(f"OOM at batch_size={batch_size}")
            break"""
if __name__ =="__main__":
    #train_k_fold("Combined_CNN")
    #train_held_out_test("ESM_CNN",exclude_num=199,loss_idx=1,opt_idx=2)
    # disterbutions = ['asymmetric_t','gaussian','asymmetric_gaussian','asymmetric_laplace',]
    # for model_type in disterbutions:
    #     train_held_out_test("ESM_CNN",exclude_num=199,loss_idx=1,opt_idx=2,model_type=model_type)
    # norm_methods = ['minmax', 'zscore', 'robust', 'meannorm', 'boxcox', 'quantile', 'log', None,
    #                 'log_zscore','log_minmax','log_meannorm','log_quantile',
    #                 'quantile_minmax','quantile_zscore','quantile_meannorm']
    
    
    remove_rna_dups = True
    alphas = 0.5
    clamp_by_percentile_options = False
    if_sample_weights_options = True
    losses = 1
    opts = 2
    clusters = [1,2,3,'all']
    cluster = 3
    norm_method = 'boxcox'
    train_held_out_test("ESM_CNN",loss_idx=losses,opt_idx=opts,if_sample_wieght=if_sample_weights_options,
                        if_clamp_by_percentile=clamp_by_percentile_options,alpha=alphas,cluster_id=cluster,
                        remove_rna_dups=remove_rna_dups,earlyStopPatience=7,seed=42,normalization_method=norm_method)                        
    
    # for los in losses:
    #     train_held_out_test("ESM_CNN",exclude_num=199,loss_idx=los,opt_idx=opts,if_sample_wieght=if_sample_weights_options,
    #                     if_clamp_by_percentile=clamp_by_percentile_options,alpha=alphas)
    # for loss_idx in losses:
    #     for opt_idx in opts:
    #         pass
            #train_held_out_test("ESM_CNN",exclude_num=199,loss_idx=loss_idx,opt_idx=opt_idx,if_sample_wieght=True,if_clamp_by_percentile=True,alph)
            #train_held_out_test("only_rna",exclude_num=199,loss_idx=loss_idx,opt_idx=opt_idx,if_sample_wieght=False)
    

    

    