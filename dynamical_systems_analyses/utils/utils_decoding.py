import ipdb
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression



def rmse(y_true, y_pred, axis=None):
    
    if axis is not None:
        return np.sqrt(np.mean((y_true - y_pred)**2, axis=axis))
    else:
        return np.sqrt(np.mean((y_true - y_pred)**2))


def r2(y_true, y_pred, axis=None):
    """
    Row- or column-wise R².

    Parameters
    ----------
    y_true, y_pred : array-like, same shape
    axis           : int or tuple or None
        Axis along which to compute R² (just like in np.mean / np.sum).

    Returns
    -------
    r2 : ndarray
        Shape is y_true.shape without the chosen `axis`.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # residual sum of squares
    ss_res = np.sum((y_true - y_pred) ** 2, axis=axis)

    # total sum of squares – keep `axis` so broadcasting works
    mean   = np.mean(y_true, axis=axis, keepdims=True)
    ss_tot = np.sum((y_true - mean) ** 2, axis=axis)

    # protect against division-by-zero
    ss_tot = np.where(ss_tot == 0, np.finfo(float).eps, ss_tot)

    return 1.0 - ss_res / ss_tot


def calculate_absolute_angle_error(predictions, ground_truth):

    # Normalize the vectors
    predictions_norm  = predictions  / np.linalg.norm(predictions,  axis=1, keepdims=True)
    ground_truth_norm = ground_truth / np.linalg.norm(ground_truth, axis=1, keepdims=True)

    # Compute the dot product between corresponding vectors
    dot_products = np.einsum('ij,ij->i', predictions_norm, ground_truth_norm)

    # Clamp the values to avoid numerical issues with arccos
    dot_products = np.clip(dot_products, -1.0, 1.0)

    # Compute the angle error in radians
    angle_errors = np.arccos(dot_products)

    # Convert radians to degrees if needed (optional)
    angle_errors_degrees = np.degrees(angle_errors)

    return np.abs(angle_errors_degrees)
    

def calculate_relative_magnitude_difference(predictions, ground_truth):

    # Compute vector norms
    predictions_norm  = np.linalg.norm(predictions,  axis=1)
    ground_truth_norm = np.linalg.norm(ground_truth, axis=1)

    # Compute the relative difference
    relative_diff = np.abs(predictions_norm - ground_truth_norm) / ground_truth_norm

    return np.abs(relative_diff)


def LR(
    X_train, 
    X_test, 
    y_train, 
    y_test, 
    metric,
    n_components=None,
    standardize=False):

    # Standardize the data
    if standardize:
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_test_std  = scaler.transform(X_test)
    else:
        X_train_std = X_train
        X_test_std  = X_test

    if n_components is not None:
        ## Fit PCA
        pca = PCA(n_components=n_components, svd_solver='full')
        X_train_pca = pca.fit_transform(X_train_std)
        X_test_pca  = pca.transform(X_test_std)
    else:
        X_train_pca = X_train_std
        X_test_pca  = X_test_std

    ## Train LR on training data
    reg = LinearRegression()
    reg.fit(X_train_pca, y_train)

    ## Predict using LR
    y_pred_train = reg.predict(X_train_pca)
    y_pred_test  = reg.predict(X_test_pca)

    if metric == 'rmse':
        rmse_train = np.sqrt(np.mean((y_train - y_pred_train)**2, axis=0))
        rmse_test  = np.sqrt(np.mean((y_test  - y_pred_test)**2,  axis=0))

        abse_train = np.abs(y_train - y_pred_train)
        abse_test  = np.abs(y_test - y_pred_test)

        return rmse_train, rmse_test, abse_train, abse_test, y_pred_train, y_pred_test
    
    elif metric == 'abs_angle':
        abs_angle_train = calculate_absolute_angle_error(y_pred_train, y_train)
        abs_angle_test  = calculate_absolute_angle_error(y_pred_test,  y_test)

        return np.mean(abs_angle_train), np.mean(abs_angle_test), abs_angle_train, abs_angle_test, y_pred_train, y_pred_test
    
    elif metric == 'rel_mag':
        rel_mag_train = calculate_relative_magnitude_difference(y_pred_train, y_train)
        rel_mag_test  = calculate_relative_magnitude_difference(y_pred_test,  y_test)

        return np.mean(rel_mag_train), np.mean(rel_mag_test), rel_mag_train, rel_mag_test, y_pred_train, y_pred_test

    elif metric == 'r2':
        r2_train = r2(y_train, y_pred_train, axis=0)
        r2_test  = r2(y_test,  y_pred_test,  axis=0)

        return np.mean(r2_train), np.mean(r2_test), r2_train, r2_test, y_pred_train, y_pred_test
    
    else:
        raise ValueError('Invalid metric')
