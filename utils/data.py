import numpy as np
import os
import glob
import scipy.signal as signal
from scipy import interpolate
import tensorflow as tf

def load_and_merge_data(data_dir):
    # Load all ECG data
    ecg_files = glob.glob(os.path.join(data_dir, 'ecg', '*.npy'))
    ecg_data = []
    for file in ecg_files:
        ecg_data.append(np.load(file))
    merged_ecg = np.concatenate(ecg_data, axis=0)
    
    # Load all PPG data
    ppg_files = glob.glob(os.path.join(data_dir, 'ppg', '*.npy'))
    ppg_data = []
    for file in ppg_files:
        ppg_data.append(np.load(file))
    merged_ppg = np.concatenate(ppg_data, axis=0)
    
    return merged_ecg, merged_ppg

def filter_data(val, Fs):
    ecg = val
    if ecg.ndim == 1:
        ecg = ecg.reshape(1, -1)
    
    # Low-pass filtering
    Fcutoff_low = 0.5
    Wn_low = (2 * Fcutoff_low) / Fs
    b_low, a_low = signal.butter(2, Wn_low, 'low')
    xn_filtered_LF = signal.filtfilt(b_low, a_low, ecg)
    
    # High-pass filtering
    Fcutoff_high = 40
    Wn_high = (2 * Fcutoff_high) / Fs
    b_high, a_high = signal.butter(2, Wn_high, 'high')
    xn_filtered_HF = signal.filtfilt(b_high, a_high, ecg)
    
    return ecg - xn_filtered_HF - xn_filtered_LF

def normalize(data):
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max == data_min:
        return np.zeros_like(data)
    return (data - data_min) / (data_max - data_min) * 2 - 1

def split_signal(signal, window_size=375):
    return [signal[i*window_size : (i+1)*window_size] 
            for i in range(len(signal)//window_size)]


# interpolate
def interpolate_ecg(ecg, new_length=100):
    original_length = ecg.shape[0]
    x_original = np.linspace(0, 1, original_length)
    x_new = np.linspace(0, 1, new_length)
    
    f = interpolate.interp1d(x_original, ecg, kind='linear')
    return f(x_new)

# Use the Pan-Tompkins algorithm for R peak detection
def detect_r_peaks(ecg_signal, fs=100):
   
    ecg_signal = ecg_signal.flatten()

    sos = signal.butter(2, [0.5, 40], 'bandpass', fs=fs, output='sos')
    filtered_ecg = signal.sosfilt(sos, ecg_signal)

    differentiated_ecg = np.gradient(filtered_ecg)

    squared_ecg = differentiated_ecg ** 2
 
    window_size = int(0.15 * fs)
    if window_size % 2 == 0:
        window_size += 1
    moving_average = np.convolve(squared_ecg, np.ones(window_size)/window_size, mode='same')
   
    threshold = 0.1 * np.max(moving_average)
    
    r_peaks = []
    in_search = False
    search_start = 0
    for i in range(len(moving_average)):
        if moving_average[i] > threshold and not in_search:
            in_search = True
            search_start = i
        elif moving_average[i] < threshold and in_search:
            in_search = False
            peak_index = np.argmax(ecg_signal[search_start:i+1])
            r_peaks.append(search_start + peak_index)
    
    r_peaks = np.array(r_peaks)
    return r_peaks


def load_and_process_data(root_dir):
    all_ecg = []
    all_labels = []
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.npz'):
                data = np.load(os.path.join(root, file))
                ecg_segments = data['segments']
                labels = data['labels']  
                
                
                for i, ecg in enumerate(ecg_segments):
                    
                    if labels[i] == 6:
                        continue
                    
                    processed_ecg = interpolate_ecg(ecg)
                    all_ecg.append(processed_ecg)
                    all_labels.append(labels[i])  
    
    return np.array(all_ecg), np.array(all_labels)


def preprocess_data(ecg_data, labels):
    
    ecg_data = np.expand_dims(ecg_data, axis=-1)
    
    mean = np.mean(ecg_data, axis=(0, 1))
    std = np.std(ecg_data, axis=(0, 1))
    ecg_data = (ecg_data - mean) / std
        
    num_classes = len(np.unique(labels))
    labels = tf.keras.utils.to_categorical(labels, num_classes)
    
    return ecg_data, labels