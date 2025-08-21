from tensorflow.keras import backend as K
import tensorflow as tf
import os
from naming_utilities import timestamp
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras import optimizers
LOSSES = {
    1: 'MSE',
    2: 'MAPE',
    3: 'MSLE',
    4: 'logcosh',
    5: 'MAE'
}

OPTS = {
    1: 'RMSprop',
    2: 'Adam'
}

log2pi = tf.math.log(2.0 * tf.constant(3.141592653589793, dtype=tf.float32))
log2pi_05 = 0.5 * log2pi
base_dir = "Models"
checkpoint_dir = os.path.join(base_dir, "Checkpoints")
tensorboard_dir = os.path.join(base_dir, "TensorBoard")

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(tensorboard_dir, exist_ok=True)


def init_checkpoint_and_tensorboard(model_name,loss_key=None , opt_idx = None):
    """Initialize checkpoint and TensorBoard directories with model name and timestamp."""
    # Optional: add timestamp and model name to distinguish runs
    global checkpoint_dir, tensorboard_dir
    
    checkpoint_dir_ = os.path.join(checkpoint_dir, f"{model_name}")
    tensorboard_dir_ = os.path.join(tensorboard_dir, f"{model_name}")
    os.makedirs(checkpoint_dir_, exist_ok=True)
    os.makedirs(tensorboard_dir_, exist_ok=True)
    if loss_key is not None:
        model_name = model_name + f"_{LOSSES[loss_key]}"
    if opt_idx is not None:
        model_name = model_name + f"_{OPTS[opt_idx]}"   
    checkpoint_dir_ = os.path.join(checkpoint_dir_, f"{model_name}_{timestamp}.keras")
    tensorboard_dir_ = os.path.join(tensorboard_dir_, f"{model_name}_{timestamp}")
    return checkpoint_dir_, tensorboard_dir_
# --- Gaussian NLL loss (stable with log-variance) ---
def gaussian_nll(y_true, y_pred):
    # y_true: (B, 1), y_pred: (B, 2) -> [mu, log_var]
    mu = y_pred[:, :1]
    log_var = y_pred[:, 1:]
    log_var = tf.clip_by_value(log_var, -10.0, 10.0)           # numeric stability
    inv_var = tf.exp(-log_var)
    
    # NLL per-sample
    nll = 0.5 * (log_var) + log2pi_05 + 0.5 * tf.square(y_true - mu) * inv_var
    return tf.reduce_mean(nll)
def mse_from_mu(y_true, y_pred):
    mu = y_pred[:, :1]
    return tf.reduce_mean(tf.square(y_true - mu))

def mae_from_mu(y_true, y_pred):
    mu = y_pred[:, :1]
    return tf.reduce_mean(tf.abs(y_true - mu))

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


def get_optimizer(optimizerIdx = 2, lrate = 0.1):
    if optimizerIdx==1:
        myOptimizer=optimizers.RMSprop(learning_rate=lrate, rho=0.9) 
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
        myLoss='log_cosh'
    elif lossIdx==5:
        myLoss= 'mean_absolute_error'
    return myLoss

def get_callbacks(checkPtFile, tensorBoardDir, plateauPatience = 0,earlyStopPatience = 0):
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
    if plateauPatience:
        callbacksList.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=plateauPatience, min_lr=1e-6))
    if earlyStopPatience:
        callbacksList.append(EarlyStopping(monitor="val_loss", patience=earlyStopPatience))
    return callbacksList
