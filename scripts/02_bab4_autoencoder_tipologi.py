import os
import json
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers


warnings.filterwarnings("ignore")


# ============================================================
# KONFIGURASI
# ============================================================

SEED = 42
INPUT_FILE = "outputs/processed/data_dasarian_tahunan_wide.csv"

OUT_BASE = "outputs/autoencoder"
OUT_TABLES = f"{OUT_BASE}/tables"
OUT_FIGURES = f"{OUT_BASE}/figures"
OUT_PROCESSED = f"{OUT_BASE}/processed"

for d in [OUT_TABLES, OUT_FIGURES, OUT_PROCESSED]:
    os.makedirs(d, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

LATENT_DIMS = [2, 3, 4]
K_VALUES = [2, 3, 4, 5, 6]

EPOCHS = 2000
PATIENCE = 150
LEARNING_RATE = 0.01
N_BOOTSTRAP = 200


# ============================================================
# FUNGSI MODEL AUTOENCODER
# ============================================================

def build_autoencoder(input_dim=36, latent_dim=2):
    """
    Autoencoder sederhana untuk data kecil:
    input 36 dasarian -> encoder -> latent_dim -> decoder -> rekonstruksi 36 dasarian.
    """

    input_layer = layers.Input(shape=(input_dim,), name="input_36_dasarian")

    x = layers.Dense(16, activation="relu", name="encoder_dense_16")(input_layer)
    x = layers.Dense(8, activation="relu", name="encoder_dense_8")(x)
    latent = layers.Dense(latent_dim, activation="linear", name="latent_vector")(x)

    x = layers.Dense(8, activation="relu", name="decoder_dense_8")(latent)
    x = layers.Dense(16, activation="relu", name="decoder_dense_16")(x)
    output_layer = layers.Dense(input_dim, activation="linear", name="reconstruction")(x)

    autoencoder = models.Model(input_layer, output_layer, name=f"autoencoder_latent_{latent_dim}")
    encoder = models.Model(input_layer, latent, name=f"encoder_latent_{latent_dim}")

    autoencoder.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse"
    )

    return autoencoder, encoder


# ============================================================
# FUNGSI STABILITAS BOOTSTRAP
# ============================================================

def bootstrap_cluster_stability(Z, reference_labels, k, n_bootstrap=200):
    """
    Stabilitas klaster:
    - ambil sampel bootstrap dari ruang laten,
    - fit KMeans pada sampel bootstrap,
    - prediksi label seluruh data asli,
    - bandingkan dengan label referensi menggunakan ARI.
    """

    rng = np.random.default_rng(SEED)
    n = Z.shape[0]
    ari_scores = []

    for _ in range(n_bootstrap):
        sample_idx = rng.choice(np.arange(n), size=n, replace=True)
        Z_boot = Z[sample_idx]

        km_boot = KMeans(n_clusters=k, random_state=SEED, n_init=50)
        km_boot.fit(Z_boot)

        boot_labels_full = km_boot.predict(Z)
        ari = adjusted_rand_score(reference_labels, boot_labels_full)
        ari_scores.append(ari)

    return float(np.mean(ari_scores)), float(np.std(ari_scores))


# ============================================================
# 1. BACA DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

feature_cols = [c for c in df.columns if c.startswith("D")]
years = df["Tahun"].values
X = df[feature_cols].values.astype(float)

print("Input shape:", X.shape)
print("Jumlah tahun:", len(years))
print("Jumlah fitur dasarian:", len(feature_cols))

if X.shape[0] != 30:
    print("PERINGATAN: jumlah tahun tidak sama dengan 30. Cek ulang data input.")

if X.shape[1] != 36:
    raise ValueError("Jumlah fitur dasarian bukan 36. Cek kolom D01 sampai D36.")


# ============================================================
# 2. NORMALISASI
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ============================================================
# 3. TRAIN AUTOENCODER UNTUK BEBERAPA LATENT DIMENSION
# ============================================================

ae_results = []
trained_models = {}

for latent_dim in LATENT_DIMS:
    print(f"\nTraining Autoencoder latent_dim={latent_dim}")

    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)

    autoencoder, encoder = build_autoencoder(
        input_dim=X_scaled.shape[1],
        latent_dim=latent_dim
    )

    early_stop = callbacks.EarlyStopping(
        monitor="loss",
        patience=PATIENCE,
        restore_best_weights=True,
        min_delta=1e-6
    )

    history = autoencoder.fit(
        X_scaled,
        X_scaled,
        epochs=EPOCHS,
        batch_size=8,
        shuffle=True,
        verbose=0,
        callbacks=[early_stop]
    )

    loss_history = history.history["loss"]
    final_loss_scaled = float(loss_history[-1])

    Z = encoder.predict(X_scaled, verbose=0)
    X_recon_scaled = autoencoder.predict(X_scaled, verbose=0)
    X_recon = scaler.inverse_transform(X_recon_scaled)

    mse_original = float(np.mean((X - X_recon) ** 2))
    mae_original = float(np.mean(np.abs(X - X_recon)))

    ae_results.append({
        "Latent_Dim": latent_dim,
        "Final_MSE_Scaled": round(final_loss_scaled, 6),
        "MSE_Original": round(mse_original, 4),
        "MAE_Original": round(mae_original, 4),
        "Epoch_Terpakai": len(loss_history)
    })

    trained_models[latent_dim] = {
        "autoencoder": autoencoder,
        "encoder": encoder,
        "Z": Z,
        "X_recon": X_recon,
        "loss_history": loss_history,
        "mse_original": mse_original,
        "mae_original": mae_original
    }

ae_eval = pd.DataFrame(ae_results)
ae_eval.to_csv(f"{OUT_TABLES}/tabel_4_7_evaluasi_autoencoder.csv", index=False)

print("\nEvaluasi Autoencoder:")
print(ae_eval)


# ============================================================
# 4. EVALUASI KLASTER UNTUK SETIAP LATENT_DIM DAN K
# ============================================================

cluster_results = []

for latent_dim in LATENT_DIMS:
    Z = trained_models[latent_dim]["Z"]

    for k in K_VALUES:
        if k >= len(Z):
            continue

        km = KMeans(n_clusters=k, random_state=SEED, n_init=50)
        labels = km.fit_predict(Z)

        sil = silhouette_score(Z, labels)
        dbi = davies_bouldin_score(Z, labels)
        ch = calinski_harabasz_score(Z, labels)

        ari_mean, ari_std = bootstrap_cluster_stability(
            Z,
            labels,
            k,
            n_bootstrap=N_BOOTSTRAP
        )

        cluster_results.append({
            "Latent_Dim": latent_dim,
            "Jumlah_Klaster": k,
            "Silhouette": round(sil, 4),
            "Davies_Bouldin": round(dbi, 4),
            "Calinski_Harabasz": round(ch, 4),
            "Bootstrap_ARI_Mean": round(ari_mean, 4),
            "Bootstrap_ARI_Std": round(ari_std, 4)
        })

cluster_eval = pd.DataFrame(cluster_results)

# Ranking sederhana:
# Silhouette tinggi bagus, DBI rendah bagus, ARI tinggi bagus.
cluster_eval["Rank_Silhouette"] = cluster_eval["Silhouette"].rank(ascending=False)
cluster_eval["Rank_DBI"] = cluster_eval["Davies_Bouldin"].rank(ascending=True)
cluster_eval["Rank_ARI"] = cluster_eval["Bootstrap_ARI_Mean"].rank(ascending=False)

cluster_eval["Total_Rank"] = (
    cluster_eval["Rank_Silhouette"] +
    cluster_eval["Rank_DBI"] +
    cluster_eval["Rank_ARI"]
)

cluster_eval = cluster_eval.sort_values("Total_Rank").reset_index(drop=True)

cluster_eval.to_csv(
    f"{OUT_TABLES}/tabel_4_8_evaluasi_jumlah_klaster.csv",
    index=False
)

print("\nEvaluasi Cluster:")
print(cluster_eval.head(15))


# ============================================================
# 5. PILIH KONFIGURASI TERBAIK
# ============================================================

best = cluster_eval.iloc[0].to_dict()
best_latent_dim = int(best["Latent_Dim"])
best_k = int(best["Jumlah_Klaster"])

print("\nKonfigurasi terpilih:")
print(best)

with open(f"{OUT_PROCESSED}/selected_config.json", "w", encoding="utf-8") as f:
    json.dump(best, f, indent=4, ensure_ascii=False)


# ============================================================
# 6. FINAL CLUSTERING
# ============================================================

Z_best = trained_models[best_latent_dim]["Z"]
X_recon_best = trained_models[best_latent_dim]["X_recon"]
loss_best = trained_models[best_latent_dim]["loss_history"]

km_final = KMeans(n_clusters=best_k, random_state=SEED, n_init=100)
cluster_labels = km_final.fit_predict(Z_best)

# Urutkan nama tipologi berdasarkan rata-rata total curah hujan tahunan.
tmp = pd.DataFrame({
    "Tahun": years,
    "Cluster_Raw": cluster_labels,
    "Total_CH": X.sum(axis=1)
})

cluster_order = (
    tmp.groupby("Cluster_Raw")["Total_CH"]
    .mean()
    .sort_values()
    .index
    .tolist()
)

mapping = {old: new + 1 for new, old in enumerate(cluster_order)}
cluster_labels_ordered = np.array([mapping[x] for x in cluster_labels])

assignment = pd.DataFrame({
    "Tahun": years,
    "Tipologi": cluster_labels_ordered,
    "Total_Curah_Hujan_Tahunan": X.sum(axis=1).round(2)
})

assignment = assignment.sort_values(["Tipologi", "Tahun"]).reset_index(drop=True)

assignment.to_csv(f"{OUT_PROCESSED}/cluster_assignment.csv", index=False)
assignment.to_csv(
    f"{OUT_TABLES}/tabel_4_9_keanggotaan_tipologi_tahunan.csv",
    index=False
)


# ============================================================
# 7. SIMPAN LATENT VECTOR DAN REKONSTRUKSI
# ============================================================

latent_cols = [f"Z{i+1}" for i in range(best_latent_dim)]

latent_df = pd.DataFrame(Z_best, columns=latent_cols)
latent_df.insert(0, "Tahun", years)
latent_df["Tipologi"] = cluster_labels_ordered

latent_df.to_csv(f"{OUT_PROCESSED}/latent_vectors.csv", index=False)

recon_df = pd.DataFrame(X_recon_best, columns=feature_cols)
recon_df.insert(0, "Tahun", years)
recon_df["Tipologi"] = cluster_labels_ordered

recon_df.to_csv(f"{OUT_PROCESSED}/reconstructed_dasarian.csv", index=False)


# ============================================================
# 8. PROFIL KLASTER
# ============================================================

wide_with_cluster = df.copy()
wide_with_cluster["Tipologi"] = cluster_labels_ordered

cluster_profiles = (
    wide_with_cluster
    .groupby("Tipologi")[feature_cols]
    .mean()
    .reset_index()
)

cluster_profiles.to_csv(f"{OUT_PROCESSED}/cluster_profiles.csv", index=False)

cluster_profiles_long = cluster_profiles.melt(
    id_vars="Tipologi",
    value_vars=feature_cols,
    var_name="Dasarian",
    value_name="Rata_Rata_Curah_Hujan"
)

cluster_profiles_long["Dasarian_Ke"] = (
    cluster_profiles_long["Dasarian"].str.replace("D", "", regex=False).astype(int)
)

cluster_profiles_long.to_csv(
    f"{OUT_PROCESSED}/cluster_profiles_long.csv",
    index=False
)

profile_summary = []

for tipologi, group in wide_with_cluster.groupby("Tipologi"):
    mean_profile = group[feature_cols].mean()
    peak_d = mean_profile.idxmax()
    min_d = mean_profile.idxmin()

    profile_summary.append({
        "Tipologi": tipologi,
        "Jumlah_Tahun": len(group),
        "Rata_Rata_Total_CH": round(group[feature_cols].sum(axis=1).mean(), 2),
        "Dasarian_Puncak": peak_d,
        "Nilai_Puncak_mm": round(mean_profile.max(), 2),
        "Dasarian_Terendah": min_d,
        "Nilai_Terendah_mm": round(mean_profile.min(), 2),
        "Tahun_Anggota": ", ".join(group["Tahun"].astype(str).tolist())
    })

profile_summary = pd.DataFrame(profile_summary)
profile_summary.to_csv(
    f"{OUT_TABLES}/tabel_4_10_profil_tipologi_dasarian.csv",
    index=False
)

stability_final = cluster_eval[
    (cluster_eval["Latent_Dim"] == best_latent_dim) &
    (cluster_eval["Jumlah_Klaster"] == best_k)
][[
    "Latent_Dim",
    "Jumlah_Klaster",
    "Bootstrap_ARI_Mean",
    "Bootstrap_ARI_Std"
]]

stability_final.to_csv(
    f"{OUT_TABLES}/tabel_4_11_stabilitas_klaster_bootstrap.csv",
    index=False
)


# ============================================================
# 9. GAMBAR 4.6 LOSS AUTOENCODER
# ============================================================

plt.figure(figsize=(9, 5))
plt.plot(loss_best)
plt.title(f"Kurva Loss Autoencoder (Latent Dim = {best_latent_dim})")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.tight_layout()
plt.savefig(f"{OUT_FIGURES}/gambar_4_6_loss_autoencoder.png")
plt.close()


# ============================================================
# 10. GAMBAR 4.7 REKONSTRUKSI
# ============================================================

total_by_year = pd.Series(X.sum(axis=1), index=years)

example_years = [
    int(total_by_year.idxmin()),
    int(total_by_year.sort_values().index[len(total_by_year) // 2]),
    int(total_by_year.idxmax())
]

for yr in example_years:
    idx = np.where(years == yr)[0][0]

    plt.figure(figsize=(12, 5))
    plt.plot(range(1, 37), X[idx], marker="o", label="Aktual")
    plt.plot(range(1, 37), X_recon_best[idx], marker="x", label="Rekonstruksi AE")
    plt.title(f"Perbandingan Pola Dasarian Aktual dan Rekonstruksi Autoencoder Tahun {yr}")
    plt.xlabel("Dasarian ke-")
    plt.ylabel("Curah Hujan Dasarian (mm)")
    plt.xticks(range(1, 37))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_FIGURES}/gambar_4_7_rekonstruksi_pola_dasarian_{yr}.png")
    plt.close()

# Simpan satu gambar utama memakai tahun median.
yr = example_years[1]
idx = np.where(years == yr)[0][0]

plt.figure(figsize=(12, 5))
plt.plot(range(1, 37), X[idx], marker="o", label="Aktual")
plt.plot(range(1, 37), X_recon_best[idx], marker="x", label="Rekonstruksi AE")
plt.title(f"Perbandingan Pola Dasarian Aktual dan Rekonstruksi Autoencoder Tahun {yr}")
plt.xlabel("Dasarian ke-")
plt.ylabel("Curah Hujan Dasarian (mm)")
plt.xticks(range(1, 37))
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_FIGURES}/gambar_4_7_rekonstruksi_pola_dasarian.png")
plt.close()


# ============================================================
# 11. GAMBAR 4.8 RUANG LATEN
# ============================================================

if best_latent_dim == 2:
    Z_plot = Z_best
    x_label = "Z1"
    y_label = "Z2"
else:
    pca = PCA(n_components=2, random_state=SEED)
    Z_plot = pca.fit_transform(Z_best)
    x_label = "PC1 dari ruang laten"
    y_label = "PC2 dari ruang laten"

plt.figure(figsize=(9, 7))

for tipologi in sorted(np.unique(cluster_labels_ordered)):
    mask = cluster_labels_ordered == tipologi
    plt.scatter(
        Z_plot[mask, 0],
        Z_plot[mask, 1],
        label=f"Tipologi {tipologi}",
        s=80
    )

for i, yr in enumerate(years):
    plt.text(Z_plot[i, 0], Z_plot[i, 1], str(yr), fontsize=8)

plt.title("Visualisasi Ruang Laten Autoencoder dan Hasil Klastering")
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_FIGURES}/gambar_4_8_ruang_laten_klaster.png")
plt.close()


# ============================================================
# 12. GAMBAR 4.9 PROFIL TIPOLOGI
# ============================================================

plt.figure(figsize=(13, 6))

for tipologi in sorted(cluster_profiles["Tipologi"].unique()):
    row = cluster_profiles[cluster_profiles["Tipologi"] == tipologi]
    y = row[feature_cols].values.flatten()

    plt.plot(
        range(1, 37),
        y,
        marker="o",
        label=f"Tipologi {tipologi}"
    )

plt.title("Profil Rata-rata Curah Hujan Dasarian per Tipologi")
plt.xlabel("Dasarian ke-")
plt.ylabel("Rata-rata Curah Hujan Dasarian (mm)")
plt.xticks(range(1, 37))
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_FIGURES}/gambar_4_9_profil_tipologi_dasarian.png")
plt.close()


# ============================================================
# 13. RINGKASAN TERMINAL
# ============================================================

print("\n============================================================")
print("SELESAI AUTOENCODER + CLUSTERING")
print("============================================================")
print(f"Latent dim terpilih: {best_latent_dim}")
print(f"Jumlah klaster terpilih: {best_k}")
print("\nProfil tipologi:")
print(profile_summary)
print("\nOutput tersimpan di:")
print(OUT_BASE)