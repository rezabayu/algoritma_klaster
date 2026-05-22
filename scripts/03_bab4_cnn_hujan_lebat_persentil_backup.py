import os
import json
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers


warnings.filterwarnings("ignore")


# ============================================================
# KONFIGURASI
# ============================================================

SEED = 42

INPUT_FILE = "outputs/processed/data_fitur_harian_r1_r3_r5_r7_r10.csv"

OUT_BASE = "outputs/cnn"
OUT_TABLES = f"{OUT_BASE}/tables"
OUT_FIGURES = f"{OUT_BASE}/figures"
OUT_PROCESSED = f"{OUT_BASE}/processed"

for d in [OUT_TABLES, OUT_FIGURES, OUT_PROCESSED]:
    os.makedirs(d, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

FEATURE_COLS = ["R1", "R3", "R5", "R7", "R10"]
PERCENTILES = [95, 97, 99]

# Split berbasis waktu
TRAIN_END = "2012-12-31"
VAL_END = "2018-12-31"
# Test otomatis 2019-01-01 sampai akhir data

EPOCHS = 300
PATIENCE = 40
BATCH_SIZE = 32
LEARNING_RATE = 0.001


# ============================================================
# FUNGSI BANTU
# ============================================================

def get_dasarian(day):
    if day <= 10:
        return 1
    elif day <= 20:
        return 2
    else:
        return 3


def local_season_group(dasarian_tahunan):
    """
    Kelompok musim lokal berdasarkan hasil analisis pola dasarian:
    - D01-D13 : periode basah awal tahun / awal transisi
    - D14-D20 : periode transisi menurun
    - D21-D27 : periode kering
    - D28-D36 : periode meningkat kembali / akhir tahun basah
    """
    if 1 <= dasarian_tahunan <= 13:
        return "Basah_Awal_Tahun"
    elif 14 <= dasarian_tahunan <= 20:
        return "Transisi_Menurun"
    elif 21 <= dasarian_tahunan <= 27:
        return "Kering"
    else:
        return "Meningkat_Akhir_Tahun"


def safe_div(num, den):
    return num / den if den != 0 else 0.0


def hit_metrics(y_true, y_pred_label, y_prob=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_label, labels=[0, 1]).ravel()

    pod = safe_div(tp, tp + fn)       # Probability of Detection / recall kejadian
    far = safe_div(fp, tp + fp)       # False Alarm Ratio
    csi = safe_div(tp, tp + fp + fn)  # Critical Success Index

    acc = accuracy_score(y_true, y_pred_label)
    precision = precision_score(y_true, y_pred_label, zero_division=0)
    recall = recall_score(y_true, y_pred_label, zero_division=0)
    f1 = f1_score(y_true, y_pred_label, zero_division=0)

    if y_prob is not None and len(np.unique(y_true)) == 2:
        auroc = roc_auc_score(y_true, y_prob)
    else:
        auroc = np.nan

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Accuracy": round(acc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1_Score": round(f1, 4),
        "POD": round(pod, 4),
        "FAR": round(far, 4),
        "CSI": round(csi, 4),
        "AUROC": round(float(auroc), 4) if not np.isnan(auroc) else np.nan,
    }


def choose_threshold_by_validation(y_true, y_prob):
    """
    Ambang keputusan dipilih dari data validasi dengan memaksimalkan CSI.
    Jika terdapat nilai CSI sama, pilih threshold dengan FAR lebih rendah.
    """
    rows = []

    for thr in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_prob >= thr).astype(int)
        m = hit_metrics(y_true, y_pred, y_prob)

        rows.append({
            "Threshold": round(float(thr), 2),
            "CSI": m["CSI"],
            "POD": m["POD"],
            "FAR": m["FAR"],
            "F1_Score": m["F1_Score"],
        })

    df_thr = pd.DataFrame(rows)
    df_thr = df_thr.sort_values(
        ["CSI", "F1_Score", "FAR"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    return float(df_thr.loc[0, "Threshold"]), df_thr


def build_cnn_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(filters=16, kernel_size=2, activation="relu", padding="same"),
        layers.Conv1D(filters=16, kernel_size=2, activation="relu", padding="same"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy")
        ]
    )

    return model


def make_class_weight(y):
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    total = n_pos + n_neg

    if n_pos == 0 or n_neg == 0:
        return None

    return {
        0: total / (2 * n_neg),
        1: total / (2 * n_pos)
    }


# ============================================================
# 1. BACA DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["Tanggal_Observasi"] = pd.to_datetime(df["Tanggal_Observasi"])

required_cols = ["Tanggal_Observasi", "Rainfall"] + FEATURE_COLS
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

df = df.sort_values("Tanggal_Observasi").reset_index(drop=True)

# Target t+1
df["Target_Date"] = df["Tanggal_Observasi"].shift(-1)
df["Rainfall_Target"] = df["Rainfall"].shift(-1)

# Kelompok musim untuk tanggal target
df["Target_Tahun"] = df["Target_Date"].dt.year
df["Target_Bulan"] = df["Target_Date"].dt.month
df["Target_Tanggal"] = df["Target_Date"].dt.day

df["Target_Dasarian_Ke_Bulan"] = df["Target_Tanggal"].apply(
    lambda x: get_dasarian(x) if pd.notna(x) else np.nan
)

df["Target_Dasarian_Tahunan"] = (
    (df["Target_Bulan"] - 1) * 3 + df["Target_Dasarian_Ke_Bulan"]
)

df["Kelompok_Musim_Lokal"] = df["Target_Dasarian_Tahunan"].apply(
    lambda x: local_season_group(int(x)) if pd.notna(x) else np.nan
)

# Hapus baris terakhir karena tidak punya target t+1
df = df.dropna(subset=["Target_Date", "Rainfall_Target"]).copy()

# Split berbasis waktu memakai tanggal target
train_mask = df["Target_Date"] <= pd.Timestamp(TRAIN_END)
val_mask = (df["Target_Date"] > pd.Timestamp(TRAIN_END)) & (df["Target_Date"] <= pd.Timestamp(VAL_END))
test_mask = df["Target_Date"] > pd.Timestamp(VAL_END)

df["Split"] = np.where(train_mask, "train", np.where(val_mask, "validation", "test"))

df_train = df[df["Split"] == "train"].copy()
df_val = df[df["Split"] == "validation"].copy()
df_test = df[df["Split"] == "test"].copy()

print("Jumlah data:")
print(df["Split"].value_counts())


# ============================================================
# 2. HITUNG AMBANG PERSENTIL DARI DATA TRAIN SAJA
# ============================================================

threshold_rows = []

for p in PERCENTILES:
    for group_name, group in df_train.groupby("Kelompok_Musim_Lokal"):
        thr = np.percentile(group["Rainfall_Target"].values, p)

        threshold_rows.append({
            "Percentile": f"P{p}",
            "Kelompok_Musim_Lokal": group_name,
            "Ambang_mm": round(float(thr), 4),
            "Jumlah_Data_Train": len(group)
        })

threshold_table = pd.DataFrame(threshold_rows)

threshold_table.to_csv(
    f"{OUT_TABLES}/tabel_4_12_ambang_persentil_musiman.csv",
    index=False
)


# ============================================================
# 3. BENTUK LABEL UNTUK P95, P97, P99
# ============================================================

for p in PERCENTILES:
    threshold_map = (
        threshold_table[threshold_table["Percentile"] == f"P{p}"]
        .set_index("Kelompok_Musim_Lokal")["Ambang_mm"]
        .to_dict()
    )

    df[f"Threshold_P{p}"] = df["Kelompok_Musim_Lokal"].map(threshold_map)
    df[f"Label_P{p}"] = (
        df["Rainfall_Target"] >= df[f"Threshold_P{p}"]
    ).astype(int)

df.to_csv(f"{OUT_PROCESSED}/dataset_cnn_hujan_lebat.csv", index=False)


# ============================================================
# 4. DISTRIBUSI LABEL
# ============================================================

dist_rows = []

for p in PERCENTILES:
    label_col = f"Label_P{p}"

    for split_name, split_df in df.groupby("Split"):
        n_total = len(split_df)
        n_pos = int(split_df[label_col].sum())
        n_neg = int(n_total - n_pos)

        dist_rows.append({
            "Percentile": f"P{p}",
            "Split": split_name,
            "Jumlah_Data": n_total,
            "Tidak_Hujan_Lebat": n_neg,
            "Hujan_Lebat": n_pos,
            "Persentase_Hujan_Lebat": round(n_pos / n_total * 100, 2)
        })

dist_table = pd.DataFrame(dist_rows)
dist_table.to_csv(
    f"{OUT_TABLES}/tabel_4_13_distribusi_label_hujan_lebat.csv",
    index=False
)

print("\nDistribusi label:")
print(dist_table)


# ============================================================
# 5. SIAPKAN FITUR
# ============================================================

X_all = df[FEATURE_COLS].values.astype(float)

scaler = StandardScaler()
scaler.fit(df_train[FEATURE_COLS].values.astype(float))

X_scaled = scaler.transform(X_all)

# Conv1D input: 5 fitur sebagai deret satu dimensi dengan 1 kanal
X_cnn = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

idx_train = df.index[df["Split"] == "train"].tolist()
idx_val = df.index[df["Split"] == "validation"].tolist()
idx_test = df.index[df["Split"] == "test"].tolist()

X_train = X_cnn[idx_train]
X_val = X_cnn[idx_val]
X_test = X_cnn[idx_test]


# ============================================================
# 6. TRAIN CNN UNTUK P95, P97, P99
# ============================================================

eval_rows = {}
history_store = {}
prediction_store = {}
threshold_scan_store = {}

for p in PERCENTILES:
    print(f"\nTraining CNN untuk label P{p}")

    label_col = f"Label_P{p}"

    y_all = df[label_col].values.astype(int)
    y_train = y_all[idx_train]
    y_val = y_all[idx_val]
    y_test = y_all[idx_test]

    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)

    model = build_cnn_model(input_shape=(len(FEATURE_COLS), 1))

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        min_delta=1e-5
    )

    class_weight = make_class_weight(y_train)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[early_stop],
        class_weight=class_weight
    )

    val_prob = model.predict(X_val, verbose=0).flatten()
    test_prob = model.predict(X_test, verbose=0).flatten()

    best_thr, threshold_scan = choose_threshold_by_validation(y_val, val_prob)

    val_pred = (val_prob >= best_thr).astype(int)
    test_pred = (test_prob >= best_thr).astype(int)

    val_metrics = hit_metrics(y_val, val_pred, val_prob)
    test_metrics = hit_metrics(y_test, test_pred, test_prob)

    row = {
        "Percentile": f"P{p}",
        "Epoch_Terpakai": len(history.history["loss"]),
        "Decision_Threshold": round(best_thr, 2),
        "Val_POD": val_metrics["POD"],
        "Val_FAR": val_metrics["FAR"],
        "Val_CSI": val_metrics["CSI"],
        "Val_AUROC": val_metrics["AUROC"],
        "Test_POD": test_metrics["POD"],
        "Test_FAR": test_metrics["FAR"],
        "Test_CSI": test_metrics["CSI"],
        "Test_AUROC": test_metrics["AUROC"],
        "Test_Accuracy": test_metrics["Accuracy"],
        "Test_F1": test_metrics["F1_Score"],
        "Test_TP": test_metrics["TP"],
        "Test_FP": test_metrics["FP"],
        "Test_FN": test_metrics["FN"],
        "Test_TN": test_metrics["TN"],
    }

    eval_rows[f"P{p}"] = row
    history_store[f"P{p}"] = history.history
    threshold_scan_store[f"P{p}"] = threshold_scan

    pred_df = df[df["Split"] == "test"][[
        "Tanggal_Observasi",
        "Target_Date",
        "Rainfall_Target",
        "Kelompok_Musim_Lokal",
        label_col
    ]].copy()

    pred_df = pred_df.rename(columns={label_col: "Observed_Label"})
    pred_df["Predicted_Probability"] = test_prob
    pred_df["Predicted_Label"] = test_pred
    pred_df["Decision_Threshold"] = best_thr

    prediction_store[f"P{p}"] = pred_df


eval_table = pd.DataFrame(list(eval_rows.values()))
eval_table = eval_table.sort_values(
    ["Val_CSI", "Val_AUROC", "Test_CSI"],
    ascending=[False, False, False]
).reset_index(drop=True)

eval_table.to_csv(
    f"{OUT_TABLES}/tabel_4_14_evaluasi_model_cnn.csv",
    index=False
)

print("\nEvaluasi model:")
print(eval_table)


# ============================================================
# 7. PILIH MODEL TERBAIK BERDASARKAN VALIDASI
# ============================================================

best_percentile = eval_table.loc[0, "Percentile"]
best_row = eval_table.iloc[0].to_dict()

print("\nModel terbaik berdasarkan validasi:")
print(best_row)

with open(f"{OUT_PROCESSED}/selected_cnn_config.json", "w", encoding="utf-8") as f:
    json.dump(best_row, f, indent=4, ensure_ascii=False)

prediction_store[best_percentile].to_csv(
    f"{OUT_PROCESSED}/predictions_cnn_best.csv",
    index=False
)

threshold_scan_store[best_percentile].to_csv(
    f"{OUT_PROCESSED}/threshold_scan_validation_best.csv",
    index=False
)


# ============================================================
# 8. CONFUSION MATRIX MODEL TERBAIK
# ============================================================

best_pred_df = prediction_store[best_percentile]

y_true_best = best_pred_df["Observed_Label"].values.astype(int)
y_pred_best = best_pred_df["Predicted_Label"].values.astype(int)
y_prob_best = best_pred_df["Predicted_Probability"].values.astype(float)

tn, fp, fn, tp = confusion_matrix(y_true_best, y_pred_best, labels=[0, 1]).ravel()

cm_table = pd.DataFrame({
    "Komponen": ["TN", "FP", "FN", "TP"],
    "Jumlah": [tn, fp, fn, tp],
    "Keterangan": [
        "Tidak hujan lebat diprediksi tidak hujan lebat",
        "Tidak hujan lebat diprediksi hujan lebat",
        "Hujan lebat tidak terdeteksi",
        "Hujan lebat berhasil terdeteksi"
    ]
})

cm_table.to_csv(
    f"{OUT_TABLES}/tabel_4_15_confusion_matrix_cnn.csv",
    index=False
)


# ============================================================
# 9. GAMBAR LOSS CNN MODEL TERBAIK
# ============================================================

hist = history_store[best_percentile]

plt.figure(figsize=(9, 5))
plt.plot(hist["loss"], label="Training loss")
plt.plot(hist["val_loss"], label="Validation loss")
plt.title(f"Kurva Loss CNN 1D ({best_percentile})")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_FIGURES}/gambar_4_9_loss_cnn.png")
plt.close()


# ============================================================
# 10. GAMBAR ROC MODEL TERBAIK
# ============================================================

if len(np.unique(y_true_best)) == 2:
    fpr, tpr, roc_thresholds = roc_curve(y_true_best, y_prob_best)
    auroc = roc_auc_score(y_true_best, y_prob_best)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Acuan acak")
    plt.title(f"Kurva ROC CNN 1D pada Data Uji ({best_percentile})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_FIGURES}/gambar_4_10_roc_cnn.png")
    plt.close()


# ============================================================
# 11. GAMBAR CONFUSION MATRIX MODEL TERBAIK
# ============================================================

cm = np.array([[tn, fp], [fn, tp]])

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title(f"Confusion Matrix CNN 1D pada Data Uji ({best_percentile})")
plt.xticks([0, 1], ["Prediksi 0", "Prediksi 1"])
plt.yticks([0, 1], ["Aktual 0", "Aktual 1"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.tight_layout()
plt.savefig(f"{OUT_FIGURES}/gambar_4_11_confusion_matrix_cnn.png")
plt.close()


# ============================================================
# 12. RINGKASAN
# ============================================================

print("\n============================================================")
print("SELESAI CNN 1D HUJAN LEBAT")
print("============================================================")
print(f"Model terbaik: {best_percentile}")
print("\nAmbang persentil musiman:")
print(threshold_table)
print("\nDistribusi label:")
print(dist_table)
print("\nEvaluasi model:")
print(eval_table)
print("\nConfusion matrix model terbaik:")
print(cm_table)
print("\nOutput tersimpan di:")
print(OUT_BASE)