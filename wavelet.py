import pywt
import numpy as np
import cv2
import os
import pandas as pd

dir_organik = 'D:/Grayscale_Beras_Organik'
dir_nonorganik = 'D:/Grayscale_Beras_NonOrganik'

def extract_wavelet_features(image_path):
    img = cv2.imread(image_path, 0)
    
    wavelet = 'db2'
    coeffs = pywt.wavedec2(img, wavelet, level=1)
    
    LL, (LH, HL, HH) = coeffs
    
    features = []
    
    features += [np.mean(LL), np.std(LL), np.max(LL), np.min(LL)]
    
    features += [np.mean(LH), np.std(LH), np.max(LH), np.min(LH)]
    
    features += [np.mean(HL), np.std(HL), np.max(HL), np.min(HL)]
    
    features += [np.mean(HH), np.std(HH), np.max(HH), np.min(HH)]
    
    return features

def process_wavelet_images(directory, label):
    features = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(directory, filename)
            wavelet_features = extract_wavelet_features(path)
            features.append([filename] + wavelet_features + [label])
    return features

organik_features = process_wavelet_images(dir_organik, 1)
nonorganik_features = process_wavelet_images(dir_nonorganik, 0)

all_features = organik_features + nonorganik_features

column_names = [
    'Filename',
    'LL_mean', 'LL_std', 'LL_max', 'LL_min',
    'LH_mean', 'LH_std', 'LH_max', 'LH_min',
    'HL_mean', 'HL_std', 'HL_max', 'HL_min',
    'HH_mean', 'HH_std', 'HH_max', 'HH_min',
    'Label'
]

df = pd.DataFrame(all_features, columns=column_names)
df.to_csv('wavelet_features.csv', index=False)

print("Ekstraksi fitur wavelet selesai. Data tersimpan di wavelet_features.csv")
