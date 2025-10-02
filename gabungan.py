import pandas as pd

# Baca file CSV fitur yang sudah diekstrak
df_glcm = pd.read_csv('glcm_features.csv')
df_wavelet = pd.read_csv('wavelet_features.csv')

# Gabungkan kedua DataFrame berdasarkan Filename dan Label
df_merged = pd.merge(df_glcm, df_wavelet, on=['Filename', 'Label'], how='inner')

# Simpan hasil gabungan ke CSV baru
df_merged.to_csv('combined_features.csv', index=False)

print("Fitur GLCM dan Wavelet berhasil digabungkan dan disimpan di combined_features.csv")
