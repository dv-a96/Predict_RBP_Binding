from datetime import datetime
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def create_model_name(model_name, mlp_layers):
    """generate model name with the number of hidden nuerons in each mlp layer.

    Args:
        model_name (str): general name
        mlp_layers (list): list of integers representing the number of neurons in each MLP layer

    Returns:
        str: full name
    """
    if model_name.lower() == 'probe_rating' or 'esm_cnn' in model_name.lower():
        mlp_layers = []
    full_name =  model_name.lower() + "_".join(str(x) for x in mlp_layers)
    return full_name