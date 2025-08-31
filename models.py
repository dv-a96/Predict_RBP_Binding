'''Module to create models to predict RBP binding intensities to RNA sequences.'''

import tensorflow as tf


from keras import models
from keras import layers
from keras import regularizers

from keras import Input
from model_utilities import *



def probe_rating(activationFunc='tanh', protein_vector_length = 1612, rna_vector_length = 1024, plateauPatience = 3,
                 earlyStopPatience = 10,  l2weight=0.01, l1weight=0.01, dropoutRate=0.5, 
                 lossIdx=1,optimizerIdx=2, lrate=0.1):
    if activationFunc=='selu':
        myInitializer="lecun_normal"
    elif activationFunc=='tanh':
        myInitializer="glorot_uniform" 
    myOptimizer = get_optimizer(optimizerIdx,lrate)
    myLoss = get_loss(lossIdx)
    
    protTensor=Input(shape=(protein_vector_length,), name='FastProt')
    if activationFunc=='selu':
        x1=layers.AlphaDropout(dropoutRate)(protTensor)
    else:
        x1=layers.Dropout(dropoutRate)(protTensor)

    x1=layers.BatchNormalization()(x1)
    x1=layers.Dense(units=32, activation=activationFunc, kernel_initializer=myInitializer, kernel_regularizer=regularizers.l1_l2(l1=0, l2=0.01))(x1)
    x1=layers.BatchNormalization()(x1)
    
    rnaTensor=Input(shape=(rna_vector_length,), name='FastRNA')
    if activationFunc=='selu':
        x2=layers.AlphaDropout(dropoutRate)(rnaTensor)
    else:
        x2=layers.Dropout(dropoutRate)(rnaTensor)
   
    x2=layers.BatchNormalization()(x2)
    x2=layers.Dense(units=32, activation=activationFunc, kernel_initializer=myInitializer, kernel_regularizer=regularizers.l1_l2(l1=0, l2=0.01))(x2)   
    x2=layers.BatchNormalization()(x2)
    merged=layers.dot([x1, x2], -1)    
    #merged=kronecker([x1, x2]) 
    #merged=layers.concatenate([x1, x2]) 
    #merged=layers.multiply([x1, x2]) 
    similarity=layers.Dense(units=1, kernel_regularizer=regularizers.l1_l2(l1=l1weight, l2=l2weight))(merged) 
    network1=models.Model([protTensor, rnaTensor], similarity) 
    network1.compile(optimizer=myOptimizer, loss=myLoss, metrics=[correlation_coefficient_loss])
    checkPtFile, tensorBoardDir = init_checkpoint_and_tensorboard("probe_rating") 
    callbacksList = get_callbacks(checkPtFile,tensorBoardDir,plateauPatience=plateauPatience,earlyStopPatience=earlyStopPatience)
    return network1, callbacksList

def RNA_convolution(input_shape=(41, 4),
                    activationFunc='relu',
                    l2weight=0.01,
                    l1weight=0.01,
                    dropoutRate=0.25):
    """
    Builds an RNA convolutional block for short sequences (e.g. 43nt).
    Returns: (input_tensor, output_tensor) so it can be integrated into a larger model.
    """

    # ---- Input ----
    inputTensor = Input(shape=input_shape, name='RNA_sequence')

    # ---- Branch 1: kernel size 7 ----
    conv_k7 = layers.Conv1D(
        filters=32,
        kernel_size=7,
        padding='same',
        activation=activationFunc,
        kernel_regularizer=regularizers.l1_l2(l1=l1weight, l2=l2weight),
        name="conv_k7"
    )(inputTensor)

    pool_k7 = layers.MaxPooling1D(pool_size=2, strides=2, name="pool_k7")(conv_k7)
    
    # ---- Branch 2: kernel size 3 ----
    conv_k3 = layers.Conv1D(
        filters=32,
        kernel_size=3,
        padding='same',
        activation=activationFunc,
        kernel_regularizer=regularizers.l1_l2(l1=l1weight, l2=l2weight),
        name="conv_k3"
    )(inputTensor)

    pool_k3 = layers.MaxPooling1D(pool_size=2, strides=2, name="pool_k3")(conv_k3)
    

    # ---- Merge branches ----
    merged = layers.Concatenate(name="merge_k3_k7")([pool_k7, pool_k3])

    # ---- Second convolution after merge ----
    conv_merged = layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding='valid',
        activation=activationFunc,
        kernel_regularizer=regularizers.l1_l2(l1=l1weight, l2=l2weight),
        name="rna_conv_merged"
    )(merged)

    merged_pool = layers.MaxPooling1D(pool_size=4, strides=2, name="rna_merged_pool")(conv_merged)
    merged_drop = layers.Dropout(dropoutRate, name="merged_drop")(merged_pool)
    

    # ---- Flatten output for integration ----
    flat_output = layers.Flatten(name="rna_flatten")(merged_drop)
    return inputTensor, flat_output

def Protein_convolution(input_shape=(1000, 20),
                    activationFunc='relu',
                    l2weight=0.01,
                    l1weight=0.01,
                    dropoutRate=0.25):
    inputTensor = Input(shape=input_shape, name='Protein_sequence')
    
    # --- Branch 1: Conv with kernel size 8 ---
    conv_8 = layers.Conv1D(filters=32,kernel_size=8,activation=activationFunc,padding='same',
        kernel_regularizer=regularizers.l1_l2(l1=l1weight, l2=l2weight),name="conv_8_prot")(inputTensor)
    pool_8 = layers.MaxPooling1D(pool_size=4,name='pool_8_prot')(conv_8)
    norm_8 = layers.BatchNormalization()(pool_8)
    drop_8 = layers.Dropout(dropoutRate)(norm_8)
    # --- Branch 2: Conv with kernel size 64 ---
    conv_64 = layers.Conv1D(filters=32,kernel_size=64,activation=activationFunc,padding='same',
        kernel_regularizer=regularizers.l1_l2(l1=l1weight, l2=l2weight),name="conv_64_prot")(inputTensor)
    pool_64 = layers.MaxPooling1D(pool_size=4,name='pool_64_prot')(conv_64)
    norm_64 = layers.BatchNormalization()(pool_64)
    drop_64 = layers.Dropout(dropoutRate)(norm_64)

    # --- Merge both branches ---
    merged = layers.Concatenate(name="merge_conv8_conv64")([drop_8, drop_64])
    x = layers.Conv1D(filters=128, kernel_size=3, activation=activationFunc, padding='valid',
                      kernel_regularizer=regularizers.l1_l2(l1=l1weight, l2=l2weight),name='prot_merged_conv')(merged)
    x = layers.MaxPooling1D(pool_size=4,name='pool_merged_prot_1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropoutRate)(x)
    x = layers.Conv1D(filters=64, kernel_size=3, activation=activationFunc, padding='valid',
                      kernel_regularizer=regularizers.l1_l2(l1=l1weight, l2=l2weight),name='prot_last_conv_1')(x)
    x = layers.MaxPooling1D(pool_size=2,name='pool_merged_prot_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropoutRate)(x)
    x = layers.Conv1D(filters=32, kernel_size=3, strides=2, activation=activationFunc, padding='valid',
                      kernel_regularizer=regularizers.l1_l2(l1=l1weight, l2=l2weight),name='prot_last_conv_2')(x)
    x = layers.Flatten(name='prot_flatten')(x)
    return inputTensor,x

def separate_cnn(protein_shape = (1000,20), rna_shape = (41,4), activationFunc='relu', mlp_layers=[64],
                  plateauPatience=3,
        earlyStopPatience=10, l2weight=0.01, l1weight=0.01, dropoutRate=0.5,
        lossIdx=1, optimizerIdx=2, lrate=0.001):
    # if optimizerIdx == 1:
    #     myOptimizer = optimizers.RMSprop(learning_rate=lrate)
    # elif optimizerIdx == 2:
    #     myOptimizer = optimizers.Adam(learning_rate=lrate)
    myOptimizer = get_optimizer(lrate=lrate)
    myLoss = get_loss(lossIdx)
    protein_tensor, flatten_protein = Protein_convolution(input_shape=protein_shape)
    rna_tensor, flatten_rna = RNA_convolution(input_shape=rna_shape)
    merged_features = layers.Concatenate(name="merge_protein_rna")([flatten_protein, flatten_rna])
   
    x = MLP_block(merged_features, mlp_layers=mlp_layers,
                  activationFunc=activationFunc,
                  l2weight=l2weight, l1weight=l1weight, dropoutRate=dropoutRate)

    # Output layer
    output = layers.Dense(1, activation='linear', name="output")(x)
    
    
    model = models.Model(inputs=[protein_tensor, rna_tensor], outputs=output)
   
    model.compile(optimizer=myOptimizer, loss=myLoss)
    full_name = create_model_name("Separate_cnn",mlp_layers)
    checkPtFile, tensorBoardDir = init_checkpoint_and_tensorboard(full_name)
    callbacksList = get_callbacks(checkPtFile,tensorBoardDir)

    print(model.summary())
    return model, callbacksList


def Combined_CNN(input_shape=(1000, 20), activationFunc='relu', plateauPatience=3,
        earlyStopPatience=10, l2weight=0.0, l1weight=0.01, dropoutRate=0.5,
        lossIdx=1, optimizerIdx=2, lrate=0.001, mlp_layers = [64]):
    
   
    myOptimizer = get_optimizer(lrate=lrate)
    myLoss = get_loss(lossIdx)
    
    inputTensor = Input(shape=input_shape, name='RNA_Protein_Matrix')
    
    x = layers.Conv1D(filters=32, kernel_size=8, activation=activationFunc, padding='same',
                      kernel_regularizer=regularizers.l1_l2(l1=l1weight, l2=l2weight))(inputTensor)
    x = layers.MaxPooling1D(pool_size=4)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropoutRate)(x)
    x = layers.Conv1D(filters=64, kernel_size=3, activation=activationFunc, padding='same',
                      kernel_regularizer=regularizers.l1_l2(l1=l1weight, l2=l2weight))(inputTensor)
    x = layers.MaxPooling1D(pool_size=4)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropoutRate)(x)
    x = layers.Conv1D(filters=128, kernel_size=3, activation=activationFunc, padding='same',
                      kernel_regularizer=regularizers.l1_l2(l1=l1weight, l2=l2weight))(x)
    x = layers.MaxPooling1D(pool_size=4)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropoutRate)(x)
    x = layers.Conv1D(filters=64, kernel_size=3, activation=activationFunc, padding='valid',
                      kernel_regularizer=regularizers.l1_l2(l1=l1weight, l2=l2weight))(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropoutRate)(x)
    x = MLP_block(x, mlp_layers=mlp_layers,
                  activationFunc=activationFunc,
                  l2weight=l2weight, l1weight=l1weight, dropoutRate=dropoutRate)
    output = layers.Dense(1, activation='linear')(x)  # Continuose output
    
    model = models.Model(inputs=inputTensor, outputs=output)
    model.compile(optimizer=myOptimizer, loss=myLoss, metrics=[correlation_coefficient_loss])
    full_name = create_model_name("Combined_cnn",mlp_layers)
    checkPtFile, tensorBoardDir = init_checkpoint_and_tensorboard(full_name)
    callbacksList = get_callbacks(checkPtFile,tensorBoardDir)
    print(model.summary())

    return model, callbacksList



def MLP_block(x, mlp_layers=[64], activationFunc='relu', l2weight=0.01, l1weight=0.01, dropoutRate=0.5):
    """Builds the Dense layers of the MLP, returns the processed tensor (no input, no compile)."""
    x = layers.Flatten()(x)   # flatten if needed (optional if already flat)
    for layer_index, hidden_number in enumerate(mlp_layers):
        x = layers.Dense(hidden_number,
                         activation=activationFunc,
                         kernel_regularizer=regularizers.l1_l2(l1=l1weight, l2=l2weight),
                         name=f'FC_{layer_index}')(x)
        x = layers.Dropout(dropoutRate)(x)
    return x



def Only_RNA(rna_input = (41,4),loss_idx= 1, check_points_folder = None, tensorboard_folder = None,
            opt_idx = 2,plateauPatience=2, earlyStopPatience=5):
    regu = 5.7215002041656515e-06
    dropout =  0.362233801349954
    inputTensor = Input(shape=rna_input, name='RNA_Protein_Matrix')
    conv_kernel_long = layers.Conv1D(51, kernel_size=24, activation='relu', use_bias=True,
                              kernel_regularizer=regularizers.l2(regu))(inputTensor)
    conv_kernel_11 = layers.Conv1D(filters=512, kernel_size=11, activation='relu', use_bias=True,
                           kernel_regularizer=regularizers.l2(regu))(inputTensor)
    conv_kernel_9 = layers.Conv1D(filters=512, kernel_size=9, activation='relu', use_bias=True,
                           kernel_regularizer=regularizers.l2(regu))(inputTensor)  # kernel of 9 nucleotides
    conv_kernel_7 = layers.Conv1D(filters=512, kernel_size=7, activation='relu', use_bias=True,
                           kernel_regularizer=regularizers.l2(regu))(inputTensor)  # kernel of 7 nucleotides
    conv_kernel_5 = layers.Conv1D(filters=512, kernel_size=5, activation='relu', use_bias=True,
                           kernel_regularizer=regularizers.l2(regu))(inputTensor)
    conv_kernel_3 = layers.Conv1D(filters=512, kernel_size=3, activation='relu', use_bias=True,
                           kernel_regularizer=regularizers.l2(regu))(inputTensor)
    conv_kernel_5_sec = layers.Conv1D(filters=151, kernel_size=5, activation='relu', use_bias=True,
                             kernel_regularizer=regularizers.l2(regu))(inputTensor) # kernel of 5 nucleotides - second path

    max_pool_long = layers.MaxPooling1D(pool_size=(10))(conv_kernel_long)
    max_pool_11 = layers.MaxPooling1D(pool_size=(21))(conv_kernel_11)
    max_pool_9 = layers.MaxPooling1D(pool_size=(23))(conv_kernel_9)
    max_pool_7 = layers.MaxPooling1D(pool_size=(25))(conv_kernel_7)
    max_pool_5 = layers.MaxPooling1D(pool_size=(27))(conv_kernel_5)
    max_pool_3 = layers.MaxPooling1D(pool_size=(29))(conv_kernel_3)
    max_pool_5_sec = layers.MaxPooling1D(pool_size=(27))(conv_kernel_5_sec)
    
    
    
    
    merge2 = layers.concatenate([max_pool_11, max_pool_3,  max_pool_5,max_pool_long,max_pool_9,max_pool_7]) #merge first path
    fl_rel = layers.Flatten()(merge2) #Flatten layer
    fl_sec = layers.Flatten()(max_pool_5_sec) #Flatten layer - second path

    drop_flat = layers.Dropout(dropout, name="drop_flat")(fl_rel)
    drop_fl_sec = layers.Dropout(dropout, name="drop_fl_el")(fl_sec) #Dropout

    hidden_dense_relu = layers.Dense(256, activation='relu')(drop_flat)  # 4096
    hidden_dense_sec = layers.Dense(152, activation='relu')(drop_fl_sec)
    drop_hidden_dense_relu = layers.Dropout(dropout, name="drop_hidden_dense_relu")(hidden_dense_relu)
    merge3 = layers.concatenate([hidden_dense_sec, drop_hidden_dense_relu]) #merge first and second path
    
    hidden_dense_relu_2 = layers.Dense(128, activation='relu')(merge3)  # 4096
    output = layers.Dense(1, activation='linear')(hidden_dense_relu_2)
    model = models.Model(inputs=inputTensor, outputs=output)
    myOptimizer = get_optimizer(opt_idx,0.001)
    myLoss = get_loss(loss_idx)
    model.compile(optimizer=myOptimizer, loss=myLoss, metrics=[correlation_coefficient_loss])
    # prot
    
    callbacksList = get_callbacks(check_points_folder,tensorboard_folder,plateauPatience, earlyStopPatience)
    print(model.summary())
    return model, callbacksList

def MLP_Model(input_shape=None,activationFunc='relu', l2weight=0.0, l1weight=0.01, dropoutRate=0.5,
              lossIdx=1, optimizerIdx=2, lrate=0.001, plateauPatience=3,
              earlyStopPatience=10,mlp_layers = [64]):
    myOptimizer = get_optimizer(optimizerIdx,lrate)
    myLoss = get_loss(lossIdx)

    
    # Input and Flatten
    inputTensor = Input(shape=input_shape, name='RNA_Protien_representation')
    x = MLP_block(inputTensor, mlp_layers=mlp_layers,
                  activationFunc=activationFunc,
                  l2weight=l2weight, l1weight=l1weight, dropoutRate=dropoutRate)
    # Output layer for regression
    output = layers.Dense(1, activation='linear')(x)
    # Build and compile
    model = models.Model(inputs=inputTensor, outputs=output)
    model.compile(optimizer=myOptimizer, loss=myLoss, metrics=[correlation_coefficient_loss])
    full_name = create_model_name("MLP",mlp_layers)
    # Callbacks
    checkPtFile, tensorBoardDir = init_checkpoint_and_tensorboard(full_name)
    callbacksList = get_callbacks(checkPtFile,tensorBoardDir)
    print(model.summary())
    return model, callbacksList


def build_esm_CNN_backbone(prot_input=(312,), rna_input=(41,4),
                   regu=5.7215002041656515e-06, dropout=0.362233801349954, dict_version= 1):
    version_dict ={
        1: {'long_filters':51, 'short_filters':512, 'prot_conv': False, 
            'rna_2_conv': False, 'dense1':256, 'dense2':128,
            'dense3': 0, 'batch_norm_rna': False, 'batch_norm_prot':False,'batch_norm_merged': False},
        
        2: {'long_filters':51, 'short_filters':512, 'prot_conv': False, 
            'rna_2_conv': False, 'dense1':256, 'dense2':128,
            'dense3': 0, 'batch_norm_rna': False, 'batch_norm_prot':False,'batch_norm_merged': True},

        3: {'long_filters':51, 'short_filters':512, 'prot_conv': True,
            'rna_2_conv': False, 'dense1':256, 'dense2':128,
            'dense3': 0, 'batch_norm_rna': False, 'batch_norm_prot':False,'batch_norm_merged': True},

        4: {'long_filters':51, 'short_filters':512, 'prot_conv': True,
            'rna_2_conv': False, 'dense1':256, 'dense2':128,
            'dense3': 0, 'batch_norm_rna': True, 'batch_norm_prot':True,'batch_norm_merged': True},
        
        5: {'long_filters':51, 'short_filters':1024, 'prot_conv': False,
            'rna_2_conv': False, 'dense1':512, 'dense2':256,
            'dense3': 128, 'batch_norm_rna': False, 'batch_norm_prot':False,'batch_norm_merged': False},
        
        6: {'long_filters':51, 'short_filters':1024, 'prot_conv': True,
            'rna_2_conv': False, 'dense1':512, 'dense2':256,
            'dense3': 128, 'batch_norm_rna': True, 'batch_norm_prot':True,'batch_norm_merged': True},

        7: {'long_filters':51, 'short_filters':1024, 'prot_conv': True,
            'rna_2_conv': True, 'dense1':512, 'dense2':256,
            'dense3': 128, 'batch_norm_rna': True, 'batch_norm_prot':True,'batch_norm_merged': True}



        
    }
    params_dict = version_dict.get(dict_version, version_dict[1])
    # RNA branch
    rna_in = Input(shape=rna_input, name='RNA_Protein_Matrix')
    conv_long = layers.Conv1D(params_dict['long_filters'], 24, use_bias=True,
                              kernel_regularizer=regularizers.l2(regu))(rna_in)
    conv_11   = layers.Conv1D(params_dict['short_filters'], 11, use_bias=True,
                              kernel_regularizer=regularizers.l2(regu))(rna_in)
    conv_5    = layers.Conv1D(params_dict['short_filters'], 5, use_bias=True,
                              kernel_regularizer=regularizers.l2(regu))(rna_in)
    conv_3    = layers.Conv1D(params_dict['short_filters'], 3, use_bias=True,
                              kernel_regularizer=regularizers.l2(regu))(rna_in)
    if params_dict['batch_norm_rna']:
        conv_long = layers.BatchNormalization()(conv_long)
        conv_11   = layers.BatchNormalization()(conv_11)
        conv_5    = layers.BatchNormalization()(conv_5)
        conv_3    = layers.BatchNormalization()(conv_3)
    conv_long = layers.ReLU()(conv_long)
    conv_11   = layers.ReLU()(conv_11)
    conv_5    = layers.ReLU()(conv_5)
    conv_3    = layers.ReLU()(conv_3)
    if params_dict['rna_2_conv']:
        conv_11 = layers.Conv1D(params_dict['short_filters'], 3, activation='relu',use_bias=True,
                              kernel_regularizer=regularizers.l2(regu))(conv_11)
        conv_5  = layers.Conv1D(params_dict['short_filters'], 3, activation='relu',use_bias=True,
                                kernel_regularizer=regularizers.l2(regu))(conv_5)
        conv_3  = layers.Conv1D(params_dict['short_filters'], 3, activation='relu',use_bias=True,
                                kernel_regularizer=regularizers.l2(regu))(conv_3)
        
    pool_long = layers.MaxPooling1D(10)(conv_long)
    pool_11   = layers.MaxPooling1D(21)(conv_11)
    pool_5    = layers.MaxPooling1D(27)(conv_5)
    pool_3    = layers.MaxPooling1D(29)(conv_3)
    merged_rna = layers.concatenate([pool_11, pool_3, pool_5, pool_long])
    
    flat_rna = layers.Flatten()(merged_rna)
        
    #merged_rna = layers.concatenate([pool_11, pool_3, pool_5, pool_long])
    drop_flat  = layers.Dropout(dropout, name="drop_flat")(flat_rna)

    dense_rna  = layers.Dense(params_dict['dense1'], activation='relu')(drop_flat)
    drop_rna   = layers.Dropout(dropout, name="drop_hidden_dense_relu")(dense_rna)
    
    

    # Protein branch
    prot_in = Input(shape=prot_input, name='Protein_representation')
    if params_dict['prot_conv']:
        prot_conv = layers.Reshape((prot_input[0],1))(prot_in)
        prot_conv = layers.Conv1D(256, 3, activation='relu', use_bias=True,
                              kernel_regularizer=regularizers.l2(regu))(prot_conv)
        prot_conv = layers.GlobalMaxPooling1D()(prot_conv)
        prot_flat = layers.Flatten()(prot_conv)
    else:
        prot_flat = layers.Flatten()(prot_in)
    if params_dict['batch_norm_prot']:
        prot_flat = layers.BatchNormalization()(prot_flat)
    # Merge branches
    merged = layers.concatenate([prot_flat, drop_rna])
    if params_dict['batch_norm_merged']:
        merged = layers.BatchNormalization()(merged)

    features = layers.Dense(params_dict['dense2'], activation='relu', name="shared_features")(merged)
    if params_dict['dense3'] > 0:
        features = layers.Dense(params_dict['dense3'], activation='relu', name="shared_features_2")(features)
        features = layers.Dropout(dropout, name="drop_features")(features)



    return prot_in,rna_in, features

def build_ESM_CNN(prot_input=(312,), rna_input=(41,4),loss_idx= 1, check_points_folder = None, tensorboard_folder = None,
                    opt_idx = 2, plateauPatience=3, earlyStopPatience=5, model_type='regression', sigmoid_head=False, dict_version=1):
    """Build an a model that accpets ESM protein embeddings and RNA one-hot encoding as input.
    The model can be used for regression or uncertainty estimation using different output layers and loss functions.

    Args:
        prot_input (tuple, optional): _description_. Defaults to (312,).
        rna_input (tuple, optional): _description_. Defaults to (41,4).
        loss_idx (int, optional): _description_. Defaults to 1.
        check_points_folder (str, optional): path. Defaults to None.
        tensorboard_folder (str, optional): path. Defaults to None.
        opt_idx (int, optional): _description_. Defaults to 2.
        plateauPatience (int, optional): _description_. Defaults to 3.
        earlyStopPatience (int, optional): _description_. Defaults to 5.
        model_type (str, optional): type of the model output. Defaults to 'regression'. regression, gaussian, asymmetric_gaussian, asymmetric_laplace, asymmetric_t
        sigmoid_head (bool, optional): Whether to use a sigmoid activation in the regression head. Defaults to False.
        dict_version (int, optional): Version of the ESM_CNN architecture to use. Defaults to 1.
    Raises:
        ValueError: _description_

    Returns:
        keras model, callbacks: _description_
    """
    metrics_list = [mse_from_mu, mae_from_mu,pearson_corr_from_mu]
    prot_in, rna_in, features = build_esm_CNN_backbone(prot_input, rna_input,dict_version=dict_version)
    myOptimizer = get_optimizer(1,0.001)
    model_type = 'regression' if model_type is None or model_type == '' else model_type.lower()
    if model_type == 'regression':
        if sigmoid_head:
            output = layers.Dense(1, activation='sigmoid', name="output")(features)
        else:
            output = layers.Dense(1, activation='linear', name="output")(features)
        model = models.Model(inputs=[prot_in,rna_in], outputs=output, name="ESM_CNN_Regression")
        myOptimizer = get_optimizer(opt_idx,0.001)
        myLoss = get_loss(loss_idx)
        model.compile(optimizer=myOptimizer, loss=myLoss, metrics=[pearson_corr])
    
    elif model_type == 'gaussian':
        out = layers.Dense(2, name="gaus_params")(features)  # [mu, log_sigma]
        model = models.Model(inputs=[prot_in,rna_in], outputs=out, name="ESM_CNN_Gausian")
        model.compile(optimizer=myOptimizer, loss=gaussian_nll, metrics=metrics_list)
    elif model_type == 'asymmetric_gaussian':
        out = layers.Dense(3, name="asym_gaus_params")(features)  # [mu, log_sigma_l, log_sigma_r]
        model = models.Model(inputs=[prot_in,rna_in], outputs=out, name="ESM_CNN_Asym_Gausian")
        model.compile(optimizer=myOptimizer, loss=two_piece_normal_nll, metrics=metrics_list)
    elif model_type == 'asymmetric_laplace':
        out = layers.Dense(3, name="laplace_params")(features)  # [mu, log_b_l, log_b_r]
        model = models.Model(inputs=[prot_in,rna_in], outputs=out, name="ESM_CNN_Asym_Laplace")
        model.compile(optimizer=myOptimizer, loss=two_piece_laplace_nll, metrics=metrics_list)
    elif model_type == 'asymmetric_t':
        out = layers.Dense(4, name="laplace_params")(features)  # [mu, log_sigma_l, log_sigma_r, raw_nu]
        model = models.Model(inputs=[prot_in,rna_in], outputs=out, name="ESM_CNN_Asym_T")
        model.compile(optimizer=myOptimizer, loss=two_piece_t_nll, metrics=metrics_list)
    else:
        raise ValueError("Invalid model_type. Choose from 'regression', 'gaussian', 'asymmetric_gaussian', 'asymmetric_laplace', 'asymmetric_t'.")
    callbacksList = get_callbacks(check_points_folder,tensorboard_folder,plateauPatience,earlyStopPatience)
    print(model.summary())
    return model, callbacksList



#
















