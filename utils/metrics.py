import tensorflow as tf
import numpy as np
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import math
from sklearn.metrics import mean_squared_error
from scipy.signal import find_peaks

def calculate_euclid(point_a, point_b):
    """
    Args:
        point_a: a data point of curve_a
        point_b: a data point of curve_b
    Return:
        The Euclid distance between point_a and point_b
    """
    return math.sqrt((point_a - point_b) ** 2)


def calculate_frechet_distance(dp, i, j, curve_a, curve_b):
    """
    Args:
        dp: The distance matrix
        i: The index of curve_a
        j: The index of curve_b
        curve_a: The data sequence of curve_a
        curve_b: The data sequence of curve_b
    Return:
        The frechet distance between curve_a[i] and curve_b[j]
    """
    if dp[i][j] > -1:
        return dp[i][j]
    elif i == 0 and j == 0:
        dp[i][j] = calculate_euclid(curve_a[0], curve_b[0])
    elif i > 0 and j == 0:
        dp[i][j] = max(calculate_frechet_distance(dp, i - 1, 0, curve_a, curve_b),
                       calculate_euclid(curve_a[i], curve_b[0]))
    elif i == 0 and j > 0:
        dp[i][j] = max(calculate_frechet_distance(dp, 0, j - 1, curve_a, curve_b),
                       calculate_euclid(curve_a[0], curve_b[j]))
    elif i > 0 and j > 0:
        dp[i][j] = max(min(calculate_frechet_distance(dp, i - 1, j, curve_a, curve_b),
                           calculate_frechet_distance(dp, i - 1, j - 1, curve_a, curve_b),
                           calculate_frechet_distance(dp, i, j - 1, curve_a, curve_b)),
                       calculate_euclid(curve_a[i], curve_b[j]))
    else:
        dp[i][j] = float("inf")
    return dp[i][j]


def get_similarity(curve_a, curve_b):
    dp = [[-1 for _ in range(len(curve_b))] for _ in range(len(curve_a))]
    similarity = calculate_frechet_distance(dp, len(curve_a) - 1, len(curve_b) - 1, curve_a, curve_b)
    # return max(np.array(dp).reshape(-1, 1))[0]
    return similarity


def compute_metrics(real, pred):
    # Convert to a NumPy arrays if they are TensorFlow tensors
    if tf.is_tensor(real): real = real.numpy()
    if tf.is_tensor(pred): pred = pred.numpy()
    
    # Flattening
    real = np.squeeze(real).ravel().astype(np.float64)
    pred = np.squeeze(pred).ravel().astype(np.float64)
    
    # MSE和PCC
    mse = np.mean((real - pred) ** 2)
    pcc, _ = pearsonr(real, pred)
    
    # DTW
    def custom_euclidean(x, y):
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        return np.abs(x.item() - y.item()) if x.size == 1 else euclidean(x, y)
    dtw_dist, _ = fastdtw(real, pred, dist=custom_euclidean)
    
    
    # Freshe distance
    frechet = get_similarity(real, pred)
    
    return mse, pcc, dtw_dist, frechet


def compute_metrics_with_rpeak(real, pred, sampling_rate=125, r_peak_tolerance=0.05):
    real = np.squeeze(real).ravel().astype(np.float32)
    pred = np.squeeze(pred).ravel().astype(np.float32)

    pcc, _ = pearsonr(real, pred)
    rmse = np.sqrt(mean_squared_error(real, pred))

    min_distance = int(0.3 * sampling_rate)
    tol_samples = int(r_peak_tolerance * sampling_rate)

    real_peaks, _ = find_peaks(real, distance=min_distance, prominence=np.std(real) * 0.5)
    pred_peaks, _ = find_peaks(pred, distance=min_distance, prominence=np.std(pred) * 0.5)

    matched = 0
    used = set()
    for rp in real_peaks:
        for i, pp in enumerate(pred_peaks):
            if i in used:
                continue
            if abs(rp - pp) <= tol_samples:
                matched += 1
                used.add(i)
                break

    r_peak_accuracy = matched / max(1, len(real_peaks))

    return {
        'PCC': pcc,
        'RMSE': rmse,
        'R_peak_accuracy': r_peak_accuracy
    }

def integrate_rpeak_into_training(test_ecg_batch, pred_ecg_batch, sampling_rate=125):
    mse_list, pcc_list, rpeak_list = [], [], []

    for i in range(test_ecg_batch.shape[0]):
        real = test_ecg_batch[i]
        pred = pred_ecg_batch[i]
        metrics = compute_metrics_with_rpeak(real, pred, sampling_rate=sampling_rate)
        mse_list.append(metrics['RMSE'] ** 2)
        pcc_list.append(metrics['PCC'])
        rpeak_list.append(metrics['R_peak_accuracy'])

    return {
        'MSE': float(np.mean(mse_list)),
        'PCC': float(np.mean(pcc_list)),
        'R_peak_accuracy': float(np.mean(rpeak_list))
    }