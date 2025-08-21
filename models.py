'''Module to create models to predict RBP binding intensities to RNA sequences.'''

import tensorflow as tf


from keras import models
from keras import layers
from keras import regularizers

from keras import Input
from naming_utilities import create_model_name
from model_utilities import correlation_coefficient_loss,get_loss, get_callbacks, gaussian_nll, mse_from_mu,mae_from_mu, get_optimizer

#### NOTE: Future addition:
""" 
1. Early stop based on training steps?
2. Reduce LR based on training steps?
3. Convert model compile and optizmiers so on to externall function avoid recoding.

4. Proberating.
5. ESM model.
6. RNA embedded model.

7. Model evalutions - compare losses, pearson r to test set, split test set.
8. Intensities normalization.

9. Additional features: ...
10. Introduce Blossum bias!
"""





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

# def get_model(input, lossIdx=1, optimizerIdx=2, lrate=0.001):
#     myOptimizer = get_optimizer(lrate=lrate)
#     myLoss = get_loss(lossIdx)
#     model = separate_cnn(input)
#     full_name = "Separate_cnn" + "_".join(str(x) for x in layers)
#     model.compile(optimizer=myOptimizer, loss=myLoss, metrics=[correlation_coefficient_loss])
#     checkPtFile, tensorBoardDir = init_checkpoint_and_tensorboard(full_name)
#     callbacksList = get_callbacks(checkPtFile,tensorBoardDir)

#     print(model.summary())
#     return model, callbacksList

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



def ESM_CNN(prot_input = (312,),rna_input = (41,4),loss_idx= 1, check_points_folder = None, tensorboard_folder = None,
            opt_idx = 2):
    regu = 5.7215002041656515e-06
    dropout =  0.362233801349954
    inputTensor = Input(shape=rna_input, name='RNA_Protein_Matrix')
    conv_kernel_long = layers.Conv1D(51, kernel_size=24, activation='relu', use_bias=True,
                              kernel_regularizer=regularizers.l2(regu))(inputTensor)
    conv_kernel_11 = layers.Conv1D(filters=512, kernel_size=11, activation='relu', use_bias=True,
                           kernel_regularizer=regularizers.l2(regu))(inputTensor)
    conv_kernel_5 = layers.Conv1D(filters=512, kernel_size=5, activation='relu', use_bias=True,
                           kernel_regularizer=regularizers.l2(regu))(inputTensor)
    conv_kernel_3 = layers.Conv1D(filters=512, kernel_size=3, activation='relu', use_bias=True,
                           kernel_regularizer=regularizers.l2(regu))(inputTensor)
    max_pool_long = layers.MaxPooling1D(pool_size=(10))(conv_kernel_long)
    max_pool_11 = layers.MaxPooling1D(pool_size=(21))(conv_kernel_11)
    max_pool_5 = layers.MaxPooling1D(pool_size=(27))(conv_kernel_5)
    max_pool_3 = layers.MaxPooling1D(pool_size=(29))(conv_kernel_3)
    merge2 = layers.concatenate([max_pool_11, max_pool_3,  max_pool_5,max_pool_long]) #merge first path
    fl_rel = layers.Flatten()(merge2) #Flatten layer
    drop_flat = layers.Dropout(dropout, name="drop_flat")(fl_rel)
    hidden_dense_relu = layers.Dense(256, activation='relu')(drop_flat)  # 4096
    drop_hidden_dense_relu = layers.Dropout(dropout, name="drop_hidden_dense_relu")(hidden_dense_relu)
    prot_tensor = Input(shape=prot_input, name='Protein_representation')
    prot_ = layers.Flatten()(prot_tensor)
    merge_4 = layers.concatenate([prot_, drop_hidden_dense_relu])
    hidden_dense_relu_2 = layers.Dense(128, activation='relu')(merge_4)  # 4096
    output = layers.Dense(1, activation='linear')(hidden_dense_relu_2)
    model = models.Model(inputs=[prot_tensor,inputTensor], outputs=output)
    myOptimizer = get_optimizer(opt_idx,0.001)
    myLoss = get_loss(loss_idx)
    model.compile(optimizer=myOptimizer, loss=myLoss, metrics=[correlation_coefficient_loss])
    # prot
    
    callbacksList = get_callbacks(check_points_folder,tensorboard_folder)
    print(model.summary())
    return model, callbacksList

def Only_RNA(rna_input = (41,4),loss_idx= 1, check_points_folder = None, tensorboard_folder = None,
            opt_idx = 2):
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
    
    callbacksList = get_callbacks(check_points_folder,tensorboard_folder)
    print(model.summary())
    return model, callbacksList

def ESM_CNN_Guasian(prot_input = (312,),rna_input = (41,4),loss_idx= 1, check_points_folder = None, tensorboard_folder = None,
                    opt_idx = 2):
    regu = 5.7215002041656515e-06
    dropout =  0.362233801349954
    inputTensor = Input(shape=rna_input, name='RNA_Protein_Matrix')
    conv_kernel_long = layers.Conv1D(51, kernel_size=24, activation='relu', use_bias=True,
                              kernel_regularizer=regularizers.l2(regu))(inputTensor)
    conv_kernel_11 = layers.Conv1D(filters=512, kernel_size=11, activation='relu', use_bias=True,
                           kernel_regularizer=regularizers.l2(regu))(inputTensor)
    
    conv_kernel_5 = layers.Conv1D(filters=512, kernel_size=5, activation='relu', use_bias=True,
                           kernel_regularizer=regularizers.l2(regu))(inputTensor)
    conv_kernel_3 = layers.Conv1D(filters=512, kernel_size=3, activation='relu', use_bias=True,
                           kernel_regularizer=regularizers.l2(regu))(inputTensor)
    max_pool_long = layers.MaxPooling1D(pool_size=(10))(conv_kernel_long)
    max_pool_11 = layers.MaxPooling1D(pool_size=(21))(conv_kernel_11)
    max_pool_5 = layers.MaxPooling1D(pool_size=(27))(conv_kernel_5)
    max_pool_3 = layers.MaxPooling1D(pool_size=(29))(conv_kernel_3)
    merge2 = layers.concatenate([max_pool_11, max_pool_3,  max_pool_5,max_pool_long]) #merge first path
    fl_rel = layers.Flatten()(merge2) #Flatten layer
    drop_flat = layers.Dropout(dropout, name="drop_flat")(fl_rel)
    hidden_dense_relu = layers.Dense(256, activation='relu')(drop_flat)  # 4096
    drop_hidden_dense_relu = layers.Dropout(dropout, name="drop_hidden_dense_relu")(hidden_dense_relu)
    prot_tensor = Input(shape=prot_input, name='Protein_representation')
    prot_ = layers.Flatten()(prot_tensor)
    merge_4 = layers.concatenate([prot_, drop_hidden_dense_relu])
    hidden_dense_relu_2 = layers.Dense(128, activation='relu')(merge_4)  # 4096
    mu = layers.Dense(1, name="mu")(hidden_dense_relu_2)              # linear
    log_var = layers.Dense(1, name="log_var")(hidden_dense_relu_2)    # linear (no activation)   
    y_pred = layers.Concatenate()([mu, log_var]) 
    model = models.Model(inputs=[prot_tensor,inputTensor], outputs=y_pred)
    myOptimizer = get_optimizer(opt_idx,0.001)
    
    model.compile(optimizer=myOptimizer, loss=gaussian_nll, metrics=[mse_from_mu,mae_from_mu])
    # prot
    
    callbacksList = get_callbacks(check_points_folder,tensorboard_folder)
    print(model.summary())
    return model, callbacksList
def ESM_CNN_Oren(prot_input = (312,),rna_input = (41,4)):
    params_dict = {
        "dropout": 0.362233801349954,
        "epochs": 78,
        "batch" : 4096,
        "regu": 5.7215002041656515e-06,
        "hidden1" : 6029,
        "hidden2" : 1168,
        "filters1" : 2376,
        "hidden_sec" : 152,
        "filters_sec" : 151,
        "leaky_alpha" : 0.23149394545024274,
        "filters_long_length" : 24,
        "filters_long" : 51
    }
    inputTensor = Input(shape=rna_input, name='RNA_Protein_Matrix')

    conv_kernel_long = layers.Conv1D(params_dict["filters_long"], kernel_size=params_dict["filters_long_length"], activation='relu', use_bias=True,
                              kernel_regularizer=regularizers.l2(params_dict["regu"]))(inputTensor)  # Long kernel - its purpose is to identify structure preferences
    conv_kernel_11 = layers.Conv1D(filters=params_dict["filters1"], kernel_size=11, activation='relu', use_bias=True,
                            kernel_regularizer=regularizers.l2(params_dict["regu"]))(inputTensor)  # kernel of 11 nucleotides
    conv_kernel_9 = layers.Conv1D(filters=params_dict["filters1"], kernel_size=9, activation='relu', use_bias=True,
                           kernel_regularizer=regularizers.l2(params_dict["regu"]))(inputTensor)  # kernel of 9 nucleotides
    conv_kernel_7 = layers.Conv1D(filters=params_dict["filters1"], kernel_size=7, activation='relu', use_bias=True,
                           kernel_regularizer=regularizers.l2(params_dict["regu"]))(inputTensor)  # kernel of 7 nucleotides
    conv_kernel_5 = layers.Conv1D(filters=params_dict["filters1"], kernel_size=5, activation='relu', use_bias=True,
                           kernel_regularizer=regularizers.l2(params_dict["regu"]))(inputTensor)  # kernel of 5 nucleotides
    conv_kernel_5_sec = layers.Conv1D(filters=params_dict["filters_sec"], kernel_size=5, activation='relu', use_bias=True,
                             kernel_regularizer=regularizers.l2(params_dict["regu"]))(inputTensor) # kernel of 5 nucleotides - second path

    max_pool_long = layers.MaxPooling1D(pool_size=(40 - params_dict["filters_long_length"]))(conv_kernel_long)
    max_pool_11 = layers.MaxPooling1D(pool_size=(31))(conv_kernel_11)
    max_pool_9 = layers.MaxPooling1D(pool_size=(33))(conv_kernel_9)
    max_pool_7 = layers.MaxPooling1D(pool_size=(35))(conv_kernel_7)
    max_pool_5 = layers.MaxPooling1D(pool_size=(37))(conv_kernel_5)
    max_pool_5_sec = layers.MaxPooling1D(pool_size=(37))(conv_kernel_5_sec)
    prot_tensor = Input(shape=prot_input, name='Protein_representation')
    prot_ = layers.Flatten()(prot_tensor)
    merge2 = layers.concatenate([max_pool_11, max_pool_7, max_pool_long, max_pool_9, max_pool_5]) #merge first path
    fl_rel = layers.Flatten()(merge2) #Flatten layer
    fl_sec = layers.Flatten()(max_pool_5_sec) #Flatten layer - second path
    drop_fl_sec = layers.Dropout(params_dict["dropout"], name="drop_fl_el")(fl_sec) #Dropout
    drop_flat = layers.Dropout(params_dict["dropout"], name="drop_flat")(fl_rel)
    hidden_dense_sec = layers.Dense(params_dict["hidden_sec"], activation='relu')(drop_fl_sec)
    hidden_dense_relu = layers.Dense(params_dict["hidden1"], activation='relu')(drop_flat)  # 4096
    drop_hidden_dense_relu = layers.Dropout(params_dict["dropout"], name="drop_hidden_dense_relu")(hidden_dense_relu)
    hidden_dense_relu1 = layers.Dense(params_dict["hidden2"], activation='relu')(drop_hidden_dense_relu)  # 1024 best
    
    merge_4 = layers.concatenate([hidden_dense_sec, hidden_dense_relu1, drop_flat, hidden_dense_relu,prot_])
    output = layers.Dense(1, activation='linear')(merge_4)
    model = models.Model(inputs=[inputTensor,prot_tensor], outputs=output)
    myOptimizer = get_optimizer(2,0.001)
    myLoss = get_loss(1)
    model.compile(optimizer=myOptimizer, loss=myLoss, metrics=[correlation_coefficient_loss])
    full_name = create_model_name("ESM_OREN","")
    # Callbacks
    checkPtFile, tensorBoardDir = init_checkpoint_and_tensorboard(full_name)
    callbacksList = get_callbacks(checkPtFile,tensorBoardDir)
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






















# def correlation_coefficient_loss(y_true, y_pred):
#     '''
#     Use K.epsilon() == 10^-7 to avoid divide by zero error    
#     '''
#     x = y_true
#     y = y_pred
#     mx = K.mean(x)
#     my = K.mean(y)
#     xm, ym = x-mx, y-my
#     r_num = K.sum(tf.multiply(xm,ym))
#     r_den = K.sqrt(tf.multiply(K.maximum(K.sum(K.square(xm)), K.epsilon()), K.maximum(K.sum(K.square(ym)), K.epsilon())))
#     r = r_num / r_den
#     r = K.maximum(K.minimum(r, 1.0), -1.0)
#     return K.square(1 - r)



# class Kron(layers.merge._Merge):
#     """Merge Layer that mimic Kronecker product on a list of 2 inputs
#     """
#     def __init__(self, axis=-1, **kwargs):
#         """
#         **kwargs: standard layer keyword arguments.
#         """
#         super(Kron, self).__init__(**kwargs)
#         self.axis=axis
#         self._reshape_required = False  # to be compatible with super class layers.merge._Merge's call() method

#     def build(self, input_shape):
#         pass

#     def _merge_function(self, inputs):
#         """
#         Do Kronecker product on the last axis for the 2 input tensors. Note inputs tensors should have equal dimension for axis=0 case (ie. batchsize should be equal). 

#         Alternatively, if inputs tensors have equal dimensions, can also use the implementation in outer_product() function below. 
#         """
#         output=K.repeat_elements(inputs[0], K.int_shape(inputs[1])[1], -1)
#         inputs1_tiled=K.tile(inputs[1], [1, K.int_shape(inputs[0])[1]])
#         return output*inputs1_tiled 
        
#     @staticmethod    
#     def outer_product(inputs):
#         """
#         use the implementation in _merge_function() for outer product, since it is more general. This outer_product() function can only deal with inputs of 2 tensors with equal dimensions
#         """
#         inputs0, inputs1 = inputs
#         batchSize = K.shape(inputs0)[0]
#         outerProduct = inputs0[:, :, np.newaxis]*inputs1[:, np.newaxis, :]
#         outerProduct = K.reshape(outerProduct, (batchSize, -1))
#         return outerProduct    

#     def compute_output_shape(self, input_shape):
#         if not isinstance(input_shape, list) or len(input_shape)!=2:
#             raise ValueError('A `Kronecker` layer should be called on a list of 2 inputs.')
#         output_shape=list(input_shape[0])
#         shape=list(input_shape[1])
#         if output_shape[self.axis] is None or shape[self.axis] is None:
#             output_shape[self.axis]=None
#         output_shape[self.axis] *= shape[self.axis]
#         return tuple(output_shape)

#     def get_config(self):
#         base_config = super().get_config()
#         return {**base_config, "axis": self.axis}



# def kronecker(inputs, **kwargs):
#     """Functional interface to the `Kron` layer.
#     # Arguments
#         inputs: A list of input tensors (at least 2).
#         **kwargs: Standard layer keyword arguments.
#     # Returns
#         A tensor, the kronecker product of the inputs.
#     """
#     return Kron(**kwargs)(inputs)




