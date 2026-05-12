import pandas as pd

total = pd.read_csv("outputs/tables/tabel_4_5_total_curah_hujan_tahunan.csv")

print("Tahun terbasah:")
print(total.sort_values("Total_Curah_Hujan", ascending=False).head(5))

print("\nTahun terkering:")
print(total.sort_values("Total_Curah_Hujan", ascending=True).head(5))

print("\nJumlah kategori awal:")
print(total["Kategori_Awal"].value_counts())