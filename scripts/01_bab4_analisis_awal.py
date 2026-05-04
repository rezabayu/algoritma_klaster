import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# KONFIGURASI
# ============================================================

RAW_FILE = "data/raw/data_ch_serang.xlsx"

OUT_PROCESSED = "outputs/processed"
OUT_TABLES = "outputs/tables"
OUT_FIGURES = "outputs/figures"

os.makedirs(OUT_PROCESSED, exist_ok=True)
os.makedirs(OUT_TABLES, exist_ok=True)
os.makedirs(OUT_FIGURES, exist_ok=True)

plt.rcParams["figure.dpi"] = 150
sns.set_theme(style="whitegrid")


# ============================================================
# 1. BACA DATA
# ============================================================

df_raw = pd.read_excel(RAW_FILE)

print("Kolom data:")
print(df_raw.columns.tolist())
print("\nContoh data awal:")
print(df_raw.head())


# ============================================================
# 2. NORMALISASI NAMA KOLOM
# ============================================================

df = df_raw.copy()
df.columns = [str(c).strip() for c in df.columns]

required_cols = ["Tahun", "Bulan", "Tanggal", "Rainfall"]
missing_cols = [c for c in required_cols if c not in df.columns]

if missing_cols:
    raise ValueError(f"Kolom wajib tidak ditemukan: {missing_cols}")

df = df[required_cols].copy()

# Pastikan numerik
df["Tahun"] = pd.to_numeric(df["Tahun"], errors="coerce")
df["Bulan"] = pd.to_numeric(df["Bulan"], errors="coerce")
df["Tanggal"] = pd.to_numeric(df["Tanggal"], errors="coerce")
df["Rainfall"] = pd.to_numeric(df["Rainfall"], errors="coerce")

# Buat tanggal observasi
df["Tanggal_Observasi"] = pd.to_datetime(
    dict(year=df["Tahun"], month=df["Bulan"], day=df["Tanggal"]),
    errors="coerce"
)

# Urutkan
df = df.sort_values("Tanggal_Observasi").reset_index(drop=True)

# Hapus baris tanpa tanggal valid
df = df.dropna(subset=["Tanggal_Observasi"]).copy()

# Jika Rainfall negatif, anggap missing karena curah hujan tidak mungkin negatif
df.loc[df["Rainfall"] < 0, "Rainfall"] = np.nan

# Susun ulang kolom
df = df[["Tanggal_Observasi", "Tahun", "Bulan", "Tanggal", "Rainfall"]]

df.to_csv(f"{OUT_PROCESSED}/data_harian_bersih.csv", index=False)


# ============================================================
# 3. TABEL 4.1 STRUKTUR DATASET
# ============================================================

tabel_4_1 = pd.DataFrame({
    "No": [1, 2, 3, 4],
    "Nama Kolom": ["Tahun", "Bulan", "Tanggal", "Rainfall"],
    "Keterangan": [
        "Tahun pengamatan",
        "Bulan pengamatan",
        "Tanggal pengamatan dalam bulan",
        "Curah hujan harian"
    ],
    "Satuan / Format": ["YYYY", "1–12", "1–31", "mm"]
})

tabel_4_1.to_csv(f"{OUT_TABLES}/tabel_4_1_struktur_dataset.csv", index=False)


# ============================================================
# 4. TABEL 4.2 KELENGKAPAN DATA TAHUNAN
# ============================================================

tanggal_min = df["Tanggal_Observasi"].min()
tanggal_max = df["Tanggal_Observasi"].max()

full_dates = pd.DataFrame({
    "Tanggal_Observasi": pd.date_range(tanggal_min, tanggal_max, freq="D")
})

df_full = full_dates.merge(
    df[["Tanggal_Observasi", "Rainfall"]],
    on="Tanggal_Observasi",
    how="left"
)

df_full["Tahun"] = df_full["Tanggal_Observasi"].dt.year
df_full["Bulan"] = df_full["Tanggal_Observasi"].dt.month
df_full["Tanggal"] = df_full["Tanggal_Observasi"].dt.day

kelengkapan = (
    df_full
    .groupby("Tahun")
    .agg(
        Jumlah_Hari_Seharusnya=("Tanggal_Observasi", "count"),
        Jumlah_Data_Tersedia=("Rainfall", lambda x: x.notna().sum()),
        Data_Hilang=("Rainfall", lambda x: x.isna().sum())
    )
    .reset_index()
)

kelengkapan["Persentase_Kelengkapan"] = (
    kelengkapan["Jumlah_Data_Tersedia"] /
    kelengkapan["Jumlah_Hari_Seharusnya"] * 100
).round(2)

kelengkapan.to_csv(
    f"{OUT_TABLES}/tabel_4_2_kelengkapan_data_tahunan.csv",
    index=False
)


# ============================================================
# 5. TABEL 4.3 STATISTIK DESKRIPTIF HARIAN
# ============================================================

rain = df_full["Rainfall"]

statistik = pd.DataFrame({
    "Statistik": [
        "Tanggal awal",
        "Tanggal akhir",
        "Jumlah hari kalender",
        "Jumlah data tersedia",
        "Jumlah data hilang",
        "Persentase kelengkapan (%)",
        "Minimum curah hujan (mm)",
        "Maksimum curah hujan (mm)",
        "Rata-rata curah hujan (mm)",
        "Median curah hujan (mm)",
        "Standar deviasi curah hujan (mm)",
        "Jumlah hari tanpa hujan",
        "Jumlah hari hujan",
        "Persentase hari hujan (%)"
    ],
    "Nilai": [
        tanggal_min.strftime("%Y-%m-%d"),
        tanggal_max.strftime("%Y-%m-%d"),
        len(df_full),
        rain.notna().sum(),
        rain.isna().sum(),
        round(rain.notna().sum() / len(df_full) * 100, 2),
        round(rain.min(), 2),
        round(rain.max(), 2),
        round(rain.mean(), 2),
        round(rain.median(), 2),
        round(rain.std(), 2),
        int((rain == 0).sum()),
        int((rain > 0).sum()),
        round((rain > 0).sum() / rain.notna().sum() * 100, 2)
    ]
})

statistik.to_csv(
    f"{OUT_TABLES}/tabel_4_3_statistik_deskriptif_harian.csv",
    index=False
)


# ============================================================
# 6. GAMBAR 4.1 DERET WAKTU HUJAN HARIAN
# ============================================================

plt.figure(figsize=(14, 5))
plt.plot(df_full["Tanggal_Observasi"], df_full["Rainfall"], linewidth=0.6)
plt.title("Deret Waktu Curah Hujan Harian Stasiun Meteorologi Maritim Serang")
plt.xlabel("Tahun")
plt.ylabel("Curah Hujan Harian (mm)")
plt.tight_layout()
plt.savefig(f"{OUT_FIGURES}/gambar_4_1_deret_waktu_hujan_harian.png")
plt.close()


# ============================================================
# 7. GAMBAR 4.2 KELENGKAPAN DATA TAHUNAN
# ============================================================

plt.figure(figsize=(12, 5))
plt.bar(kelengkapan["Tahun"], kelengkapan["Persentase_Kelengkapan"])
plt.title("Persentase Kelengkapan Data Curah Hujan Harian per Tahun")
plt.xlabel("Tahun")
plt.ylabel("Kelengkapan Data (%)")
plt.ylim(0, 105)
plt.tight_layout()
plt.savefig(f"{OUT_FIGURES}/gambar_4_2_kelengkapan_data_tahunan.png")
plt.close()


# ============================================================
# 8. PEMBENTUKAN DASARIAN
# ============================================================

def get_dasarian(day):
    if day <= 10:
        return 1
    elif day <= 20:
        return 2
    else:
        return 3

df_full["Dasarian_Ke_Bulan"] = df_full["Tanggal"].apply(get_dasarian)
df_full["Dasarian_Tahunan"] = (df_full["Bulan"] - 1) * 3 + df_full["Dasarian_Ke_Bulan"]

dasarian = (
    df_full
    .groupby(["Tahun", "Dasarian_Tahunan"])
    .agg(
        Curah_Hujan_Dasarian=("Rainfall", "sum"),
        Jumlah_Data_Tersedia=("Rainfall", lambda x: x.notna().sum()),
        Jumlah_Hari=("Rainfall", "size")
    )
    .reset_index()
)

dasarian["Persentase_Kelengkapan"] = (
    dasarian["Jumlah_Data_Tersedia"] / dasarian["Jumlah_Hari"] * 100
).round(2)

dasarian.to_csv(
    f"{OUT_PROCESSED}/data_dasarian_tahunan.csv",
    index=False
)

tabel_4_4 = dasarian.head(20).copy()
tabel_4_4.to_csv(
    f"{OUT_TABLES}/tabel_4_4_contoh_agregasi_dasarian.csv",
    index=False
)


# ============================================================
# 9. DATA WIDE 36 DASARIAN PER TAHUN
# ============================================================

dasarian_wide = dasarian.pivot(
    index="Tahun",
    columns="Dasarian_Tahunan",
    values="Curah_Hujan_Dasarian"
).reset_index()

dasarian_wide.columns = ["Tahun"] + [f"D{int(c):02d}" for c in dasarian_wide.columns[1:]]
dasarian_wide.to_csv(
    f"{OUT_PROCESSED}/data_dasarian_tahunan_wide.csv",
    index=False
)


# ============================================================
# 10. POLA RATA-RATA DASARIAN
# ============================================================

rata_dasarian = (
    dasarian
    .groupby("Dasarian_Tahunan")
    .agg(
        Rata_Rata_Curah_Hujan=("Curah_Hujan_Dasarian", "mean"),
        Median_Curah_Hujan=("Curah_Hujan_Dasarian", "median"),
        Standar_Deviasi=("Curah_Hujan_Dasarian", "std")
    )
    .reset_index()
)

rata_dasarian = rata_dasarian.round(2)

rata_dasarian.to_csv(
    f"{OUT_TABLES}/tabel_4_6_rata_rata_dasarian.csv",
    index=False
)

plt.figure(figsize=(13, 5))
plt.plot(
    rata_dasarian["Dasarian_Tahunan"],
    rata_dasarian["Rata_Rata_Curah_Hujan"],
    marker="o"
)
plt.title("Pola Rata-rata Curah Hujan Dasarian")
plt.xlabel("Dasarian ke-")
plt.ylabel("Rata-rata Curah Hujan Dasarian (mm)")
plt.xticks(range(1, 37))
plt.tight_layout()
plt.savefig(f"{OUT_FIGURES}/gambar_4_3_pola_rata_rata_dasarian.png")
plt.close()


# ============================================================
# 11. TOTAL CURAH HUJAN TAHUNAN
# ============================================================

total_tahunan = (
    df_full
    .groupby("Tahun")
    .agg(
        Total_Curah_Hujan=("Rainfall", "sum"),
        Jumlah_Data_Tersedia=("Rainfall", lambda x: x.notna().sum()),
        Jumlah_Hari=("Rainfall", "size")
    )
    .reset_index()
)

total_tahunan["Persentase_Kelengkapan"] = (
    total_tahunan["Jumlah_Data_Tersedia"] /
    total_tahunan["Jumlah_Hari"] * 100
).round(2)

q33 = total_tahunan["Total_Curah_Hujan"].quantile(0.33)
q67 = total_tahunan["Total_Curah_Hujan"].quantile(0.67)

def kategori_tahun(x):
    if x <= q33:
        return "Relatif kering"
    elif x >= q67:
        return "Relatif basah"
    else:
        return "Relatif normal"

total_tahunan["Kategori_Awal"] = total_tahunan["Total_Curah_Hujan"].apply(kategori_tahun)
total_tahunan["Total_Curah_Hujan"] = total_tahunan["Total_Curah_Hujan"].round(2)

total_tahunan.to_csv(
    f"{OUT_TABLES}/tabel_4_5_total_curah_hujan_tahunan.csv",
    index=False
)

plt.figure(figsize=(13, 5))
plt.bar(total_tahunan["Tahun"], total_tahunan["Total_Curah_Hujan"])
plt.title("Total Curah Hujan Tahunan")
plt.xlabel("Tahun")
plt.ylabel("Total Curah Hujan (mm)")
plt.tight_layout()
plt.savefig(f"{OUT_FIGURES}/gambar_4_4_total_curah_hujan_tahunan.png")
plt.close()


# ============================================================
# 12. HEATMAP DASARIAN TAHUNAN
# ============================================================

heatmap_data = dasarian_wide.set_index("Tahun")

plt.figure(figsize=(16, 8))
sns.heatmap(
    heatmap_data,
    cmap="YlGnBu",
    linewidths=0.2,
    linecolor="white"
)
plt.title("Heatmap Curah Hujan Dasarian per Tahun")
plt.xlabel("Dasarian")
plt.ylabel("Tahun")
plt.tight_layout()
plt.savefig(f"{OUT_FIGURES}/gambar_4_5_heatmap_dasarian_tahunan.png")
plt.close()


# ============================================================
# 13. FITUR HARIAN R1, R3, R5, R7, R10
# ============================================================

fitur = df_full.copy()
fitur["R1"] = fitur["Rainfall"].rolling(window=1, min_periods=1).sum()
fitur["R3"] = fitur["Rainfall"].rolling(window=3, min_periods=1).sum()
fitur["R5"] = fitur["Rainfall"].rolling(window=5, min_periods=1).sum()
fitur["R7"] = fitur["Rainfall"].rolling(window=7, min_periods=1).sum()
fitur["R10"] = fitur["Rainfall"].rolling(window=10, min_periods=1).sum()

fitur.to_csv(
    f"{OUT_PROCESSED}/data_fitur_harian_r1_r3_r5_r7_r10.csv",
    index=False
)


# ============================================================
# 14. RINGKASAN UNTUK DITULIS KE BAB IV
# ============================================================

print("\n============================================================")
print("RINGKASAN HASIL AWAL")
print("============================================================")
print(f"Periode data: {tanggal_min.date()} sampai {tanggal_max.date()}")
print(f"Jumlah hari kalender: {len(df_full)}")
print(f"Jumlah data tersedia: {rain.notna().sum()}")
print(f"Jumlah data hilang: {rain.isna().sum()}")
print(f"Kelengkapan data: {round(rain.notna().sum() / len(df_full) * 100, 2)}%")
print(f"Curah hujan maksimum harian: {round(rain.max(), 2)} mm")
print(f"Rata-rata curah hujan harian: {round(rain.mean(), 2)} mm")
print(f"Median curah hujan harian: {round(rain.median(), 2)} mm")
print(f"Jumlah hari tanpa hujan: {int((rain == 0).sum())}")
print(f"Jumlah hari hujan: {int((rain > 0).sum())}")
print("\nOutput selesai dibuat.")