import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
import os
import pandas as pd

dir_organik = 'Grayscale_Beras_Organik/'
dir_nonorganik = 'Grayscale_Beras_NonOrganik/'

def extract_glcm_features(image_path):
    img = cv2.imread(image_path, 0) 
    
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    glcm = graycomatrix(img, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    
    return [contrast, homogeneity, energy, correlation]

def process_images(directory, label):
    features = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(directory, filename)
            glcm_features = extract_glcm_features(path)
            features.append([filename] + glcm_features + [label])
    return features

organik_features = process_images(dir_organik, 1) 
nonorganik_features = process_images(dir_nonorganik, 0)

all_features = organik_features + nonorganik_features
df = pd.DataFrame(all_features, columns=['Filename', 'Contrast', 'Homogeneity', 'Energy', 'Correlation', 'Label'])
df.to_csv('glcm_features.csv', index=False)