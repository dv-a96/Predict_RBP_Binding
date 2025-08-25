from tensorflow.keras import backend as K
import tensorflow as tf
import os
import math

from datetime import datetime
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


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


log_2pi = tf.constant(math.log(2.0 * math.pi), dtype=tf.float32)
log_pi  = tf.constant(math.log(math.pi), dtype=tf.float32)
log2pi_05 = 0.5 * log_2pi


base_dir = "Models"
checkpoint_dir = os.path.join(base_dir, "Checkpoints")
tensorboard_dir = os.path.join(base_dir, "TensorBoard")

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(tensorboard_dir, exist_ok=True)

def create_model_name(model_name, mlp_layers, model_type='regression'):
    """generate model name with the number of hidden nuerons in each mlp layer.

    Args:
        model_name (str): general name
        mlp_layers (list): list of integers representing the number of neurons in each MLP layer

    Returns:
        str: full name
    """
    
    if model_name.lower() == 'probe_rating' or 'esm_cnn' in model_name.lower():
        mlp_layers = []
    model_name = model_name.lower() + '_' + model_type
    full_name =  model_name.lower() + "_".join(str(x) for x in mlp_layers)
    return full_name

def init_checkpoint_and_tensorboard(model_name,loss_key=None , opt_idx = None, model_type='regression', mlp_layers = [],
                                    if_sample_wieght=False, alpha=None, bins=None, if_clamp_by_percentile = False,
                        percentile = None,cluster_id = None,remove_rna_dups = False,norm_method = None):
    """Initialize checkpoint and TensorBoard directories with model name and timestamp."""
    # Optional: add timestamp and model name to distinguish runs
    global checkpoint_dir, tensorboard_dir
    if cluster_id is not None:
        model_name_ = f"{model_name}_cluster_{cluster_id}"
    else: model_name_ = model_name
    checkpoint_dir_ = os.path.join(checkpoint_dir, f"{model_name_}")
    tensorboard_dir_ = os.path.join(tensorboard_dir, f"{model_name_}")

    os.makedirs(checkpoint_dir_, exist_ok=True)
    os.makedirs(tensorboard_dir_, exist_ok=True)
    
    model_name = create_model_name(model_name, mlp_layers, model_type=model_type)
    if norm_method is not None:
        model_name = f"{model_name}_{norm_method}"
    if if_sample_wieght:
        if alpha is None or bins is None:
            raise ValueError("If if_sample_wieght is True, alpha and bins must be provided.")
        model_name = f"{model_name}_alpha{alpha}_bins{bins}"
    if if_clamp_by_percentile:
        if percentile is None:
            raise ValueError("If if_clamp_by_percentile is True, percentile must be provided.")
        model_name = f"{model_name}_clamp{percentile}"
    if model_type == 'regression':
            
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


def _side_scale(y_true, mu, left_scale, right_scale):
    # side: 1.0 for left (y <= mu), 0.0 for right (y > mu)
    side_left = tf.cast(y_true <= mu, y_true.dtype)
    return side_left * left_scale + (1.0 - side_left) * right_scale, side_left

# ---------- Two-piece Normal ----------
def two_piece_normal_nll(y_true, y_pred):
    """
    y_true: (B, 1)
    y_pred: (B, 3) -> [mu, log_sigma_l, log_sigma_r]
    """
    mu         = y_pred[:, :1]
    log_sig_l  = tf.clip_by_value(y_pred[:, 1:2], -10.0, 10.0)
    log_sig_r  = tf.clip_by_value(y_pred[:, 2:3], -10.0, 10.0)
    sig_l      = tf.exp(log_sig_l)
    sig_r      = tf.exp(log_sig_r)

    s, _       = _side_scale(y_true, mu, sig_l, sig_r)
    z          = (y_true - mu) / s

    # NLL per sample: log((σ_l+σ_r)/2) + 0.5*log(2π) + 0.5*z^2
    log_norm_const = tf.math.log((sig_l + sig_r) * 0.5) + log2pi_05
    nll = log_norm_const + 0.5 * tf.square(z)
    return tf.reduce_mean(nll)

# ---------- Two-piece Laplace ----------
def two_piece_laplace_nll(y_true, y_pred):
    """
    y_true: (B, 1)
    y_pred: (B, 3) -> [mu, log_b_l, log_b_r]
    """
    mu       = y_pred[:, :1]
    log_bl   = tf.clip_by_value(y_pred[:, 1:2], -10.0, 10.0)
    log_br   = tf.clip_by_value(y_pred[:, 2:3], -10.0, 10.0)
    b_l      = tf.exp(log_bl)
    b_r      = tf.exp(log_br)

    s, _     = _side_scale(y_true, mu, b_l, b_r)
    # NLL: log(b_l + b_r) + |y - mu| / s(y)
    nll = tf.math.log(b_l + b_r) + tf.abs(y_true - mu) / s
    return tf.reduce_mean(nll)
def two_piece_t_nll(y_true, y_pred):
    """
    y_true: (B, 1)
    y_pred: (B, 4) -> [mu, log_sigma_l, log_sigma_r, raw_nu]
    """
    mu         = y_pred[:, :1]
    log_sig_l  = tf.clip_by_value(y_pred[:, 1:2], -10.0, 10.0)
    log_sig_r  = tf.clip_by_value(y_pred[:, 2:3], -10.0, 10.0)
    sig_l      = tf.exp(log_sig_l)
    sig_r      = tf.exp(log_sig_r)

    # ν > 0 enforced by softplus; add tiny epsilon for safety
    raw_nu     = y_pred[:, 3:4]
    nu         = tf.nn.softplus(raw_nu) + 1e-3

    s, _       = _side_scale(y_true, mu, sig_l, sig_r)
    z          = (y_true - mu) / s

    # log C_nu = lgamma((ν+1)/2) - lgamma(ν/2) - 0.5*(log ν + log π)
    logC = (tf.math.lgamma((nu + 1.0) * 0.5)
            - tf.math.lgamma(nu * 0.5)
            - 0.5 * (tf.math.log(nu) + log_pi))

    # NLL: log((σ_l+σ_r)/2) - logC + ((ν+1)/2) * log(1 + z^2/ν)
    nll = tf.math.log((sig_l + sig_r) * 0.5) - logC \
          + 0.5 * (nu + 1.0) * tf.math.log1p(tf.square(z) / nu)

    return tf.reduce_mean(nll)
def mse_from_mu(y_true, y_pred):
    mu = y_pred[:, :1]
    return tf.reduce_mean(tf.square(y_true - mu))

def mae_from_mu(y_true, y_pred):
    mu = y_pred[:, :1]
    return tf.reduce_mean(tf.abs(y_true - mu))

def pearson_corr(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    xm = y_true - tf.reduce_mean(y_true)
    ym = y_pred - tf.reduce_mean(y_pred)
    corr = tf.reduce_sum(xm * ym) / (
        tf.sqrt(tf.reduce_sum(tf.square(xm))) *
        tf.sqrt(tf.reduce_sum(tf.square(ym))) + 1e-8
    )
    return corr
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
        callbacksList.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=plateauPatience, min_lr=1e-6,verbose=1))
    if earlyStopPatience:
        callbacksList.append(EarlyStopping(monitor="val_loss", patience=earlyStopPatience,verbose=1,restore_best_weights=True))
    return callbacksList
