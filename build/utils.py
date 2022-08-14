import numpy as np
import pandas as pd


def normalization(data, norm_params=None):
    N, D = data.shape
    if norm_params is None:
        min_val = np.zeros(D)
        max_val = np.zeros(D)
        norm_data = data.copy()
        for d in range(D):
            m1 = np.nanmin(data[:, d])
            m2 = np.nanmax(data[:, d])
            min_val[d] = m1
            max_val[d] = m2
            norm_data[:, d] = (data[:, d] - m1) / (m2 - m1 + 1e-6)
        norm_params = {"min_val": min_val, "max_val": max_val}
    else:
        min_val = norm_params["min_val"]
        max_val = norm_params["max_val"]
        norm_data = data.copy()
        for d in range(D):
            m1 = min_val[d]
            m2 = max_val[d]
            norm_data[:, d] = (data[:, d] - m1) / (m2 - m1 + 1e-6)
    return norm_data, norm_params



def renormalization(norm_data, norm_params):
    N, D = norm_data.shape
    min_val = norm_params["min_val"]
    max_val = norm_params["max_val"]
    data = norm_data.copy()
    for d in range(D):
        m1 = min_val[d]
        m2 = max_val[d]
        data[:, d] = norm_data[:, d] * (m2 - m1 + 1e-6) + m1
    return data



def reset_weights(model):
    """Completely reinitialize model parameters"""
    for layer in model.layers:
        if layer.name[:5] == "dense":
            nb_in = layer.input_shape[-1]
            nb_out = layer.output_shape[-1]
            limit = np.sqrt(6.0 / (nb_in + nb_out))
            layer_shape = layer.weights[0].numpy().shape
            layer_total = layer_shape[0] * layer_shape[1]
            bias_shape = layer.weights[1].numpy().shape
            r1 = np.random.uniform(-limit, limit, size=layer_total)
            r1 = np.reshape(r1, layer_shape)
            r2 = np.zeros(shape=bias_shape)
            layer.set_weights([r1, r2])
        elif layer.name[:19] == "batch_normalization":
            layer_shape = layer.weights[0].numpy().shape
            r1 = np.ones(shape=layer_shape)
            r2 = np.zeros(shape=layer_shape)
            r3 = np.zeros(shape=layer_shape)
            r4 = np.ones(shape=layer_shape)
            layer.set_weights([r1, r2, r3, r4])



def maskDistribution(dataset):
    """unique_masks: list of unique NaN masks found in the dataset
    count_masks: corresponding number of occurrences (the probability distrib.)"""
    mask = (1.0 - np.isnan(dataset)).astype("int")
    unique_masks = np.unique(mask, axis=0)
    count_masks = np.zeros(len(unique_masks), dtype="int")
    for i1 in range(mask.shape[0]):
        current_mask = mask[i1]
        i2 = np.where((unique_masks == current_mask).all(axis=1))[0][0]
        count_masks[i2] += 1
    return unique_masks, count_masks



def drawMasks(unique_masks, probs, N):
    """unique_masks: list of unique masks from which to choose
    probs: vector of probability (should sum up to one)
    N: number of samples to draw
    masks: list of size N containing one mask per row drawn from the desired distribution"""
    multinom = np.random.multinomial(n=1, pvals=probs, size=N)
    indices = np.where(multinom==1)[1]
    masks = unique_masks[indices]
    return masks



def drawHintMatrix(p, nb_rows, nb_cols):
    """p: probability of ones
    nb_rows: number of desired rows in the hint matrix H
    nb_cols: number of desired columns in the hint matrix H
    H: hint matrix"""
    H = np.random.uniform(0., 1., size=(nb_rows, nb_cols))
    H = 1.0 * (H < p)
    return H


def load_dataset(name):
    """name: name of the dataset
    data: properly loaded dataset"""
    if name=="breast":
        data = np.genfromtxt("datasets/breast/wdbc.data", delimiter=",")
        data = data[:, 2:]  # We only keep the last 30 columns
    elif name=="spam":
        data = np.genfromtxt("datasets/spam/spambase.data", delimiter=",")
        data = data[:, :-1]  # Get rid of the last column (the label: spam or not)
    elif name=="letter":
        data = np.genfromtxt("datasets/letter/letter-recognition.data", delimiter=",")
        data = data[:, 1:]  # Get rid of the first column (the actual letter)
    elif name=="credit":
        data = pd.read_excel("datasets/credit/default_clients.xls")
        data = np.array(data.iloc[1:, 1:-1], dtype="float32")  # Get rid of the labels and the irrelevant columns
        data = np.delete(data, [1, 2, 3], axis=1)  # Remove the socio-demographics categorical variables
        data = np.delete(data, [2, 3, 4, 5, 6, 7], axis=1)  # Remove the past payment records (categorical)
    elif name=="news":
        data = np.genfromtxt("datasets/news/OnlineNewsPopularity.csv", delimiter=",")
        data = data[1:, 2:-1]  # Remove labels and irrelevant columns
        data = np.delete(data, [11, 12, 13, 14, 15, 16], axis=1)  # Remove data channel (binary)
        data = np.delete(data, [23, 24, 25, 26, 27, 28, 29, 30], axis=1)  # Remove what day of the week (binary)
    elif name=="wine_red":
        data = np.genfromtxt("datasets/wine/winequality-red.csv", delimiter=";")
        data = data[1:, :]  # Remove labels
    elif name=="wine_white":
        data = np.genfromtxt("datasets/wine/winequality-white.csv", delimiter=";")
        data = data[1:, :]  # Remove labels
    elif name=="mydata1":
        data = np.genfromtxt("datasets/mydata1/multivariate_gauss.csv", delimiter=",")
    elif name=="mydata2":
        data = np.genfromtxt("datasets/mydata2/gaussian_mixture.csv", delimiter=",")
    else:
        print("Error: dataset name not valid")
        data = None
    data = np.array(data, dtype="float32")
    return data


def create_MARprob1(name):
    """name: name of the dataset
    mask: properly loaded dataset"""
    data = load_dataset(name)
    norm_data, _ = normalization(data)
    if name=="spam":  # spam is a particular case
        idx_column = 54
        temp1 = np.log(1e-6 + norm_data[:, idx_column])  # column_idx=54
        m1 = np.min(temp1)
        m2 = np.max(temp1)
        temp2 = (temp1 - m1) / (m2 - m1)
        MARvar = temp2 * 0.20 / np.mean(temp2)
    else:
        if name=="breast":
            idx_column = 0
        elif name=="letter":
            idx_column = 4
        elif name=="credit":
            idx_column = 1
        elif name=="news":
            idx_column = 0
        elif name=="wine_red":
            idx_column = 0
        elif name=="wine_white":
            idx_column = 0
        elif name=="mydata1":
            idx_column = 0
        elif name=="mydata2":
            idx_column = 0
        else:
            print("Error: dataset name not valid")
            idx_column = "error"
        MARvar = norm_data[:, idx_column] * 0.20 / np.mean(norm_data[:, idx_column])
    return MARvar, idx_column


def create_MARprob1bis(name):
    """name: name of the dataset
    mask: properly loaded dataset"""
    data = load_dataset(name)
    if name=="breast":
        idx_column = 0
    elif name=="spam":
        idx_column = 54
    elif name=="letter":
        idx_column = 4
    elif name=="credit":
        idx_column = 1
    elif name=="news":
        idx_column = 0
    elif name=="wine_red":
        idx_column = 0
    elif name=="wine_white":
        idx_column = 0
    elif name=="mydata1":
        idx_column = 0
    elif name=="mydata2":
        idx_column = 0
    else:
        print("Error: dataset name not valid")
        idx_column = "error"
    MARvar = np.argsort(np.argsort(data[:, idx_column])) / data.shape[0] * 0.4
    return MARvar, idx_column


def create_MARprob2(name):
    """name: name of the dataset
    mask: properly loaded dataset"""
    data = load_dataset(name)
    if name=="breast":
        idx_column = 0
    elif name=="spam":
        idx_column = 54
    elif name=="letter":
        idx_column = 4
    elif name=="credit":
        idx_column = 1
    elif name=="news":
        idx_column = 0
    elif name=="wine_red":
        idx_column = 0
    elif name=="wine_white":
        idx_column = 0
    elif name=="mydata1":
        idx_column = 0
    elif name=="mydata2":
        idx_column = 0
    else:
        print("Error: dataset name not valid")
        idx_column = "error"
    MARvar = np.argsort(np.argsort(data[:, idx_column])) / data.shape[0] * 0.9
    return MARvar, idx_column