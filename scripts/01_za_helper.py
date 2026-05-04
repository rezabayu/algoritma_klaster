import pandas as pd

harian = pd.read_csv("outputs/processed/data_harian_bersih.csv")
dasarian = pd.read_csv("outputs/processed/data_dasarian_tahunan.csv")
fitur = pd.read_csv("outputs/processed/data_fitur_harian_r1_r3_r5_r7_r10.csv")

print("Data harian bersih:", harian.shape)
print("Data dasarian:", dasarian.shape)
print("Data fitur harian:", fitur.shape)

print("\nJumlah tahun data dasarian:")
print(dasarian["Tahun"].nunique())

print("\nJumlah dasarian per tahun:")
print(dasarian.groupby("Tahun")["Dasarian_Tahunan"].nunique().describe())

print("\nKolom data fitur:")
print(fitur.columns.tolist())

dasa_tahun_wide = pd.read_csv("outputs/processed/data_dasarian_tahunan_wide.csv")

print(dasa_tahun_wide.head())
print(dasa_tahun_wide.shape)
print(dasa_tahun_wide.columns.tolist())