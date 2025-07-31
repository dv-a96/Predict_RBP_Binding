'''Module to create models to predict RBP binding intensities to RNA sequences.'''

import tensorflow as tf
from tensorflow.keras import backend as K

from keras import models
from keras import layers
from keras import regularizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
import os
from keras import optimizers
from keras import Input
from datetime import datetime
base_dir = "Models"
checkpoint_dir = os.path.join(base_dir, "Checkpoints")
tensorboard_dir = os.path.join(base_dir, "TensorBoard")

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(tensorboard_dir, exist_ok=True)

#### NOTE: Future addition:
""" 
1. Early stop based on training steps?
2. Reduce LR based on training steps?
"""


def get_optimizer(optimizerIdx = 2, lrate = 0.1):
    if optimizerIdx==1:
        myOptimizer=optimizers.RMSprop(learning_rate=lrate, rho=0.9, epsilon=None, decay=0.0) 
    elif optimizerIdx==2:
        myOptimizer = optimizers.Adam(learning_rate=lrate, beta_1=0.9, beta_2=0.999,  amsgrad=False)
    return myOptimizer

def get_loss(lossIdx):
    if lossIdx==1:
        myLoss='mean_squared_error'
    elif lossIdx==2:
        myLoss='mean_absolute_percentage_error'
    elif lossIdx==3:
        myLoss='mean_squared_logarithmic_error'
    elif lossIdx==4:
        myLoss='logcosh'
    return myLoss


def init_checkpoint_and_tensorboard(model_name):
    """Initialize checkpoint and TensorBoard directories with model name and timestamp."""
    # Optional: add timestamp and model name to distinguish runs
    global checkpoint_dir, tensorboard_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(checkpoint_dir, f"{model_name}")
    tensorboard_dir = os.path.join(tensorboard_dir, f"{model_name}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    checkpoint_dir = os.path.join(checkpoint_dir, f"{model_name}_{timestamp}.keras")
    tensorboard_dir = os.path.join(tensorboard_dir, f"{model_name}_{timestamp}")
    return checkpoint_dir, tensorboard_dir

def get_callbacks(checkPtFile, tensorBoardDir, plateauPatience = 3 ):
    """Generated a callback list with checkpoint, reduce lr and tensorboard

    Args:
        checkPtFile (str): path to checkpoint folder
        tensorBoardDir (str): path to tensorboard folder
        plateauPatience (int, optional): number of epo. Defaults to 3.
    """
    #NOTE: No early stoping and reduce plateu due to 1 epoch training.
    # EarlyStopping(monitor="val_loss", patience=earlyStopPatience),
    # ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=plateauPatience, min_lr=1e-6),
    callbacksList = [
        ModelCheckpoint(filepath=checkPtFile, verbose=1, monitor="val_loss", save_best_only=True),
        
        TensorBoard(tensorBoardDir, histogram_freq=0, embeddings_freq=0, update_freq=10)
    ]
    return callbacksList

def probe_rating(activationFunc='tanh', protein_vector_length = 1612, rna_vector_length = 1024, plateauPatience = 3,
                 earlyStopPatience = 10,  l2weight=0, l1weight=0.01, dropoutRate=0.5, 
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
    callbacksList = get_callbacks(checkPtFile,tensorBoardDir)
    return network1, callbacksList

def RNA_convolution(input_shape=(41, 4),
                    activationFunc='relu',
                    l2weight=0.0,
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
        name="conv_merged"
    )(merged)

    merged_pool = layers.MaxPooling1D(pool_size=4, strides=2, name="merged_pool")(conv_merged)
    merged_drop = layers.Dropout(dropoutRate, name="merged_drop")(merged_pool)

    # ---- Flatten output for integration ----
    flat_output = layers.Flatten(name="rna_flatten")(merged_drop)
    return inputTensor, flat_output

def Protein_convolution(input_shape=(1000, 20),
                    activationFunc='relu',
                    l2weight=0.0,
                    l1weight=0.01,
                    dropoutRate=0.25):
    inputTensor = Input(shape=input_shape, name='Protein_sequence')
    
    # --- Branch 1: Conv with kernel size 8 ---
    conv_8 = layers.Conv1D(
        filters=64,
        kernel_size=8,
        activation=activationFunc,
        padding='same',
        kernel_regularizer=regularizers.l1_l2(l1=l1weight, l2=l2weight),
        name="conv_8"
    )(inputTensor)

    pool_8 = layers.MaxPooling1D(pool_size=2)(conv_8)
    norm_8 = layers.BatchNormalization()(pool_8)
    drop_8 = layers.Dropout(dropoutRate)(norm_8)

    # --- Branch 2: Conv with kernel size 64 ---
    conv_64 = layers.Conv1D(
        filters=64,
        kernel_size=64,
        activation=activationFunc,
        padding='same',
        kernel_regularizer=regularizers.l1_l2(l1=l1weight, l2=l2weight),
        name="conv_64"
    )(inputTensor)

    pool_64 = layers.MaxPooling1D(pool_size=2)(conv_64)
    norm_64 = layers.BatchNormalization()(pool_64)
    drop_64 = layers.Dropout(dropoutRate)(norm_64)

    # --- Merge both branches ---
    merged = layers.Concatenate(name="merge_conv8_conv64")([drop_8, drop_64])
    x = layers.Conv1D(filters=128, kernel_size=3, activation=activationFunc, padding='valid',
                      kernel_regularizer=regularizers.l1_l2(l1=l1weight, l2=l2weight))(merged)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropoutRate)(x)
    x = layers.Flatten()(x)
    return inputTensor,x

def separate_cnn(protein_shape = (1000,20), rna_shape = (41,4), activationFunc='relu', plateauPatience=3,
        earlyStopPatience=10, l2weight=0.0, l1weight=0.01, dropoutRate=0.5,
        lossIdx=1, optimizerIdx=2, lrate=0.001):
    if optimizerIdx == 1:
        myOptimizer = optimizers.RMSprop(learning_rate=lrate)
    elif optimizerIdx == 2:
        myOptimizer = optimizers.Adam(learning_rate=lrate)
    myOptimizer = get_optimizer(lrate=lrate)
    myLoss = get_loss(lossIdx)
    protein_tensor, flatten_protein = Protein_convolution(input_shape=protein_shape)
    rna_tensor, flatten_rna = RNA_convolution(input_shape=rna_shape)
    merged_features = layers.Concatenate(name="merge_protein_rna")([flatten_protein, flatten_rna])

    # --- Dense layers after merge ---
    dense1 = layers.Dense(128, activation=activationFunc, name="dense_1")(merged_features)
    dense2 = layers.Dense(64, activation=activationFunc, name="dense_2")(dense1)

    # --- Output layer (example: binary classification) ---
    output = layers.Dense(1, activation='linear', name="output")(dense2)
    
    model = models.Model(inputs=[protein_tensor, rna_tensor], outputs=output)
    model.compile(optimizer=myOptimizer, loss=myLoss, metrics=[correlation_coefficient_loss])
    checkPtFile, tensorBoardDir = init_checkpoint_and_tensorboard("Separate_cnn")
    callbacksList = get_callbacks(checkPtFile,tensorBoardDir)
    print(model.summary())
    return model,callbacksList
def Combined_CNN(input_shape=(1000, 20), activationFunc='relu', plateauPatience=3,
        earlyStopPatience=10, l2weight=0.0, l1weight=0.01, dropoutRate=0.5,
        lossIdx=1, optimizerIdx=2, lrate=0.001):
    
    if optimizerIdx == 1:
        myOptimizer = optimizers.RMSprop(learning_rate=lrate)
    elif optimizerIdx == 2:
        myOptimizer = optimizers.Adam(learning_rate=lrate)
    myOptimizer = get_optimizer(lrate=lrate)
    myLoss = get_loss(lossIdx)
    
    inputTensor = Input(shape=input_shape, name='RNA_Protein_Matrix')
    
    x = layers.Conv1D(filters=64, kernel_size=8, activation=activationFunc, padding='same',
                      kernel_regularizer=regularizers.l1_l2(l1=l1weight, l2=l2weight))(inputTensor)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropoutRate)(x)

    x = layers.Conv1D(filters=128, kernel_size=3, activation=activationFunc, padding='same',
                      kernel_regularizer=regularizers.l1_l2(l1=l1weight, l2=l2weight))(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropoutRate)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation=activationFunc)(x)
    output = layers.Dense(1, activation='linear')(x)  # Continuose output
    
    model = models.Model(inputs=inputTensor, outputs=output)
    model.compile(optimizer=myOptimizer, loss=myLoss, metrics=[correlation_coefficient_loss])
    
    checkPtFile, tensorBoardDir = init_checkpoint_and_tensorboard("CNN_model")
    callbacksList = get_callbacks(checkPtFile,tensorBoardDir)
    print(model.summary())

    return model, callbacksList


def MLP_Model(input_shape=(1050, 20), activationFunc='relu', plateauPatience=3,
              earlyStopPatience=10, l2weight=0.0, l1weight=0.01, dropoutRate=0.5,
              lossIdx=1, optimizerIdx=2, lrate=0.001):
    myOptimizer = get_optimizer(optimizerIdx,lrate)
    myLoss = get_loss(lossIdx)

    
    # Input and Flatten
    inputTensor = Input(shape=input_shape, name='RNA_Protein_Matrix')
    x = layers.Flatten()(inputTensor)

    # Fully connected layers
    x = layers.Dense(128, activation=activationFunc,
                     kernel_regularizer=regularizers.l1_l2(l1=l1weight, l2=l2weight))(x)
    x = layers.Dropout(dropoutRate)(x)

    x = layers.Dense(64, activation=activationFunc,
                     kernel_regularizer=regularizers.l1_l2(l1=l1weight, l2=l2weight))(x)
    x = layers.Dropout(dropoutRate)(x)

    x = layers.Dense(16, activation=activationFunc,
                     kernel_regularizer=regularizers.l1_l2(l1=l1weight, l2=l2weight))(x)
    x = layers.Dropout(dropoutRate)(x)

    # Output layer for regression
    output = layers.Dense(1, activation='linear')(x)

    # Build and compile
    model = models.Model(inputs=inputTensor, outputs=output)
    model.compile(optimizer=myOptimizer, loss=myLoss, metrics=[correlation_coefficient_loss])

    # Callbacks
    checkPtFile, tensorBoardDir = init_checkpoint_and_tensorboard("MLP_model")
    callbacksList = get_callbacks(checkPtFile,tensorBoardDir)

    return model, callbacksList





















def correlation_coefficient_loss(y_true, y_pred):
    '''
    Use K.epsilon() == 10^-7 to avoid divide by zero error    
    '''
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.maximum(K.sum(K.square(xm)), K.epsilon()), K.maximum(K.sum(K.square(ym)), K.epsilon())))
    r = r_num / r_den
    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return K.square(1 - r)



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




