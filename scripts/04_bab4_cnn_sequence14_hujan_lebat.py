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

OUT_BASE = "outputs/cnn_sequence14"
OUT_TABLES = f"{OUT_BASE}/tables"
OUT_FIGURES = f"{OUT_BASE}/figures"
OUT_PROCESSED = f"{OUT_BASE}/processed"

for d in [OUT_TABLES, OUT_FIGURES, OUT_PROCESSED]:
    os.makedirs(d, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

RAIN_SEQUENCE_DAYS = 14
ACCUM_FEATURE_COLS = ["R1", "R3", "R5", "R7", "R10"]
SEASON_FEATURE_COLS = ["sin_dayofyear", "cos_dayofyear"]
AUX_FEATURE_COLS = ACCUM_FEATURE_COLS + SEASON_FEATURE_COLS

HEAVY_RAIN_THRESHOLD = 50.0

TRAIN_END = "2012-12-31"
VAL_END = "2018-12-31"

EPOCHS = 300
PATIENCE = 40
BATCH_SIZE = 32
LEARNING_RATE = 0.001


# ============================================================
# FUNGSI BANTU
# ============================================================

def safe_div(num, den):
    return num / den if den != 0 else 0.0


def hit_metrics(y_true, y_pred_label, y_prob=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_label, labels=[0, 1]).ravel()

    pod = safe_div(tp, tp + fn)
    far = safe_div(fp, tp + fp)
    csi = safe_div(tp, tp + fp + fn)

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
    rows = []

    for thr in np.arange(0.01, 0.96, 0.01):
        y_pred = (y_prob >= thr).astype(int)
        m = hit_metrics(y_true, y_pred, y_prob)

        rows.append({
            "Threshold": round(float(thr), 2),
            "CSI": m["CSI"],
            "POD": m["POD"],
            "FAR": m["FAR"],
            "F1_Score": m["F1_Score"],
            "Precision": m["Precision"],
            "Recall": m["Recall"],
        })

    df_thr = pd.DataFrame(rows)

    df_thr = df_thr.sort_values(
        ["CSI", "F1_Score", "FAR"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    return float(df_thr.loc[0, "Threshold"]), df_thr


def make_class_weight(mode, y_train):
    if mode == "none":
        return None

    if mode == "balanced":
        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        total = n_pos + n_neg

        if n_pos == 0 or n_neg == 0:
            return None

        return {
            0: total / (2 * n_neg),
            1: total / (2 * n_pos)
        }

    if mode == "weight_1_2":
        return {0: 1.0, 1: 2.0}

    if mode == "weight_1_3":
        return {0: 1.0, 1: 3.0}

    if mode == "weight_1_5":
        return {0: 1.0, 1: 5.0}

    raise ValueError(f"Class weight mode tidak dikenali: {mode}")


def build_sequence_aux_cnn(seq_shape, aux_shape):
    """
    Model hybrid:
    - Cabang 1 membaca sequence curah hujan 14 hari terakhir.
    - Cabang 2 membaca fitur tambahan R1/R3/R5/R7/R10 + sin/cos hari dalam tahun.
    """

    seq_input = layers.Input(shape=seq_shape, name="rain_sequence_input")

    x = layers.Conv1D(filters=16, kernel_size=3, activation="relu", padding="same")(seq_input)
    x = layers.Conv1D(filters=16, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(16, activation="relu")(x)

    aux_input = layers.Input(shape=aux_shape, name="aux_feature_input")

    y = layers.Dense(16, activation="relu")(aux_input)
    y = layers.Dense(8, activation="relu")(y)

    z = layers.Concatenate()([x, y])
    z = layers.Dense(16, activation="relu")(z)
    z = layers.Dropout(0.2)(z)
    output = layers.Dense(1, activation="sigmoid")(z)

    model = models.Model(
        inputs=[seq_input, aux_input],
        outputs=output,
        name="cnn_sequence14_aux"
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy")
        ]
    )

    return model


def evaluate_baseline(feature_name, threshold, split_name, split_df):
    y_true = split_df["Label_Lebat_50"].values.astype(int)
    y_pred = (split_df[feature_name].values >= threshold).astype(int)

    m = hit_metrics(y_true, y_pred, y_prob=None)

    return {
        "Model": f"Baseline_{feature_name}_ge_{int(threshold)}",
        "Split": split_name,
        "Decision_Rule": f"{feature_name} >= {threshold} mm",
        "Threshold": threshold,
        **m
    }


# ============================================================
# 1. BACA DAN SUSUN DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)
df["Tanggal_Observasi"] = pd.to_datetime(df["Tanggal_Observasi"])

required_cols = ["Tanggal_Observasi", "Rainfall"] + ACCUM_FEATURE_COLS
missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

df = df.sort_values("Tanggal_Observasi").reset_index(drop=True)

# Target t+1
df["Target_Date"] = df["Tanggal_Observasi"].shift(-1)
df["Rainfall_Target"] = df["Rainfall"].shift(-1)

df = df.dropna(subset=["Target_Date", "Rainfall_Target"]).copy()

df["Label_Lebat_50"] = (df["Rainfall_Target"] >= HEAVY_RAIN_THRESHOLD).astype(int)

df["Kategori_Target"] = np.select(
    [
        df["Rainfall_Target"] > 150,
        (df["Rainfall_Target"] >= 100) & (df["Rainfall_Target"] <= 150),
        (df["Rainfall_Target"] >= 50) & (df["Rainfall_Target"] < 100),
    ],
    [
        "Ekstrem >150 mm",
        "Sangat lebat 100-150 mm",
        "Lebat 50-100 mm",
    ],
    default="Tidak lebat <50 mm"
)

# Fitur musim untuk tanggal target
dayofyear = df["Target_Date"].dt.dayofyear.astype(float)
df["sin_dayofyear"] = np.sin(2 * np.pi * dayofyear / 366.0)
df["cos_dayofyear"] = np.cos(2 * np.pi * dayofyear / 366.0)

# Split berdasarkan tanggal target
train_mask = df["Target_Date"] <= pd.Timestamp(TRAIN_END)
val_mask = (df["Target_Date"] > pd.Timestamp(TRAIN_END)) & (df["Target_Date"] <= pd.Timestamp(VAL_END))
test_mask = df["Target_Date"] > pd.Timestamp(VAL_END)

df["Split"] = np.where(train_mask, "train", np.where(val_mask, "validation", "test"))

print("Jumlah data sebelum pembentukan sequence:")
print(df["Split"].value_counts())


# ============================================================
# 2. BENTUK SEQUENCE 14 HARI
# ============================================================

rows = []
seq_list = []
aux_list = []
y_list = []

rain_values = df["Rainfall"].values.astype(float)

for i in range(RAIN_SEQUENCE_DAYS - 1, len(df)):
    current_row = df.iloc[i]

    # sequence sampai hari t, target adalah t+1
    seq = rain_values[i - RAIN_SEQUENCE_DAYS + 1:i + 1]

    # pastikan tidak melompat tanggal karena data lengkap seharusnya harian
    window_dates = df["Tanggal_Observasi"].iloc[i - RAIN_SEQUENCE_DAYS + 1:i + 1]
    if (window_dates.diff().dropna().dt.days != 1).any():
        continue

    aux = current_row[AUX_FEATURE_COLS].values.astype(float)

    seq_list.append(seq)
    aux_list.append(aux)
    y_list.append(int(current_row["Label_Lebat_50"]))

    rows.append(current_row)

df_model = pd.DataFrame(rows).reset_index(drop=True)

X_seq = np.array(seq_list, dtype=float)
X_aux = np.array(aux_list, dtype=float)
y = np.array(y_list, dtype=int)

# reshape sequence: samples, timesteps, channels
X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))

print("\nData model setelah sequence:")
print(df_model["Split"].value_counts())
print("X_seq:", X_seq.shape)
print("X_aux:", X_aux.shape)
print("y:", y.shape)


# ============================================================
# 3. DISTRIBUSI LABEL
# ============================================================

dist_rows = []

for split_name, split_df in df_model.groupby("Split"):
    n_total = len(split_df)
    n_pos = int(split_df["Label_Lebat_50"].sum())
    n_neg = int(n_total - n_pos)

    n_lebat = int(((split_df["Rainfall_Target"] >= 50) & (split_df["Rainfall_Target"] < 100)).sum())
    n_sangat_lebat = int(((split_df["Rainfall_Target"] >= 100) & (split_df["Rainfall_Target"] <= 150)).sum())
    n_ekstrem = int((split_df["Rainfall_Target"] > 150).sum())

    dist_rows.append({
        "Split": split_name,
        "Jumlah_Data": n_total,
        "Tidak_Hujan_Lebat_<50mm": n_neg,
        "Hujan_Lebat_>=50mm": n_pos,
        "Persentase_Hujan_Lebat": round(n_pos / n_total * 100, 2),
        "Lebat_50_100mm": n_lebat,
        "Sangat_Lebat_100_150mm": n_sangat_lebat,
        "Ekstrem_>150mm": n_ekstrem,
    })

dist_table = pd.DataFrame(dist_rows)
dist_table.to_csv(
    f"{OUT_TABLES}/tabel_4_17_distribusi_label_sequence14.csv",
    index=False
)

df_model.to_csv(
    f"{OUT_PROCESSED}/dataset_cnn_sequence14_metadata.csv",
    index=False
)


# ============================================================
# 4. SPLIT INDEX
# ============================================================

idx_train = df_model.index[df_model["Split"] == "train"].tolist()
idx_val = df_model.index[df_model["Split"] == "validation"].tolist()
idx_test = df_model.index[df_model["Split"] == "test"].tolist()

X_seq_train_raw = X_seq[idx_train]
X_seq_val_raw = X_seq[idx_val]
X_seq_test_raw = X_seq[idx_test]

X_aux_train_raw = X_aux[idx_train]
X_aux_val_raw = X_aux[idx_val]
X_aux_test_raw = X_aux[idx_test]

y_train = y[idx_train]
y_val = y[idx_val]
y_test = y[idx_test]


# ============================================================
# 5. NORMALISASI
# ============================================================

seq_scaler = StandardScaler()
seq_scaler.fit(X_seq_train_raw.reshape(-1, 1))

X_seq_train = seq_scaler.transform(X_seq_train_raw.reshape(-1, 1)).reshape(X_seq_train_raw.shape)
X_seq_val = seq_scaler.transform(X_seq_val_raw.reshape(-1, 1)).reshape(X_seq_val_raw.shape)
X_seq_test = seq_scaler.transform(X_seq_test_raw.reshape(-1, 1)).reshape(X_seq_test_raw.shape)

aux_scaler = StandardScaler()
aux_scaler.fit(X_aux_train_raw)

X_aux_train = aux_scaler.transform(X_aux_train_raw)
X_aux_val = aux_scaler.transform(X_aux_val_raw)
X_aux_test = aux_scaler.transform(X_aux_test_raw)


# ============================================================
# 6. BASELINE
# ============================================================

baseline_rows = []

df_val = df_model[df_model["Split"] == "validation"].copy()
df_test = df_model[df_model["Split"] == "test"].copy()

for feature in ACCUM_FEATURE_COLS:
    for split_name, split_df in [
        ("validation", df_val),
        ("test", df_test),
    ]:
        baseline_rows.append(
            evaluate_baseline(
                feature_name=feature,
                threshold=HEAVY_RAIN_THRESHOLD,
                split_name=split_name,
                split_df=split_df
            )
        )

baseline_table = pd.DataFrame(baseline_rows)
baseline_table.to_csv(
    f"{OUT_TABLES}/tabel_4_18_evaluasi_baseline_sequence14.csv",
    index=False
)


# ============================================================
# 7. TRAIN MODEL UNTUK BEBERAPA CLASS WEIGHT
# ============================================================

class_weight_modes = [
    "none",
    "balanced",
    "weight_1_2",
    "weight_1_3",
    "weight_1_5",
]

cnn_rows = []
history_store = {}
prediction_store = {}
threshold_scan_store = {}

for mode in class_weight_modes:
    print(f"\nTraining CNN sequence14 mode class_weight = {mode}")

    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)

    model = build_sequence_aux_cnn(
        seq_shape=(RAIN_SEQUENCE_DAYS, 1),
        aux_shape=(len(AUX_FEATURE_COLS),)
    )

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        min_delta=1e-5
    )

    class_weight = make_class_weight(mode, y_train)

    history = model.fit(
        [X_seq_train, X_aux_train],
        y_train,
        validation_data=([X_seq_val, X_aux_val], y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[early_stop],
        class_weight=class_weight
    )

    val_prob = model.predict([X_seq_val, X_aux_val], verbose=0).flatten()
    test_prob = model.predict([X_seq_test, X_aux_test], verbose=0).flatten()

    best_thr, threshold_scan = choose_threshold_by_validation(y_val, val_prob)

    val_pred = (val_prob >= best_thr).astype(int)
    test_pred = (test_prob >= best_thr).astype(int)

    val_metrics = hit_metrics(y_val, val_pred, val_prob)
    test_metrics = hit_metrics(y_test, test_pred, test_prob)

    row = {
        "Model": f"CNN_seq14_aux_{mode}",
        "Class_Weight_Mode": mode,
        "Epoch_Terpakai": len(history.history["loss"]),
        "Decision_Threshold": round(best_thr, 2),

        "Val_POD": val_metrics["POD"],
        "Val_FAR": val_metrics["FAR"],
        "Val_CSI": val_metrics["CSI"],
        "Val_AUROC": val_metrics["AUROC"],
        "Val_TP": val_metrics["TP"],
        "Val_FP": val_metrics["FP"],
        "Val_FN": val_metrics["FN"],
        "Val_TN": val_metrics["TN"],

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

    cnn_rows.append(row)
    history_store[mode] = history.history
    threshold_scan_store[mode] = threshold_scan

    pred_df = df_test[[
        "Tanggal_Observasi",
        "Target_Date",
        "Rainfall_Target",
        "Kategori_Target",
        "Label_Lebat_50"
    ] + ACCUM_FEATURE_COLS + SEASON_FEATURE_COLS].copy()

    pred_df = pred_df.rename(columns={"Label_Lebat_50": "Observed_Label"})
    pred_df["Predicted_Probability"] = test_prob
    pred_df["Predicted_Label"] = test_pred
    pred_df["Decision_Threshold"] = best_thr
    pred_df["Class_Weight_Mode"] = mode

    prediction_store[mode] = pred_df


cnn_eval_table = pd.DataFrame(cnn_rows)

cnn_eval_table = cnn_eval_table.sort_values(
    ["Val_CSI", "Val_AUROC", "Val_FAR"],
    ascending=[False, False, True]
).reset_index(drop=True)

cnn_eval_table.to_csv(
    f"{OUT_TABLES}/tabel_4_19_evaluasi_cnn_sequence14_class_weight.csv",
    index=False
)

print("\nEvaluasi CNN sequence14:")
print(cnn_eval_table)


# ============================================================
# 8. PILIH MODEL BERDASARKAN VALIDASI
# ============================================================

best_mode_by_val = cnn_eval_table.loc[0, "Class_Weight_Mode"]
best_row_by_val = cnn_eval_table.iloc[0].to_dict()

# Untuk analisis tambahan, juga simpan model dengan Test_CSI tertinggi.
best_row_by_test = cnn_eval_table.sort_values(
    ["Test_CSI", "Test_AUROC", "Test_FAR"],
    ascending=[False, False, True]
).iloc[0].to_dict()

best_mode_by_test = best_row_by_test["Class_Weight_Mode"]

with open(f"{OUT_PROCESSED}/selected_cnn_sequence14_by_validation.json", "w", encoding="utf-8") as f:
    json.dump(best_row_by_val, f, indent=4, ensure_ascii=False)

with open(f"{OUT_PROCESSED}/selected_cnn_sequence14_by_test_for_analysis.json", "w", encoding="utf-8") as f:
    json.dump(best_row_by_test, f, indent=4, ensure_ascii=False)

# Gambar utama tetap berdasarkan pilihan validasi.
best_mode = best_mode_by_val

prediction_store[best_mode].to_csv(
    f"{OUT_PROCESSED}/predictions_cnn_sequence14_best_by_validation.csv",
    index=False
)

threshold_scan_store[best_mode].to_csv(
    f"{OUT_PROCESSED}/threshold_scan_sequence14_validation_best.csv",
    index=False
)


# ============================================================
# 9. CONFUSION MATRIX MODEL TERPILIH VALIDASI
# ============================================================

best_pred_df = prediction_store[best_mode]

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
    f"{OUT_TABLES}/tabel_4_20_confusion_matrix_sequence14_best.csv",
    index=False
)


# ============================================================
# 10. PERBANDINGAN BASELINE DAN CNN PADA TEST
# ============================================================

baseline_test = baseline_table[baseline_table["Split"] == "test"].copy()

baseline_comp = baseline_test[[
    "Model", "POD", "FAR", "CSI", "AUROC", "TP", "FP", "FN", "TN"
]].copy()

baseline_comp["Tipe"] = "Baseline"

cnn_comp = cnn_eval_table[[
    "Model", "Test_POD", "Test_FAR", "Test_CSI", "Test_AUROC",
    "Test_TP", "Test_FP", "Test_FN", "Test_TN"
]].copy()

cnn_comp = cnn_comp.rename(columns={
    "Test_POD": "POD",
    "Test_FAR": "FAR",
    "Test_CSI": "CSI",
    "Test_AUROC": "AUROC",
    "Test_TP": "TP",
    "Test_FP": "FP",
    "Test_FN": "FN",
    "Test_TN": "TN",
})

cnn_comp["Tipe"] = "CNN_sequence14"

comparison_table = pd.concat([baseline_comp, cnn_comp], ignore_index=True)

comparison_table = comparison_table.sort_values(
    ["CSI", "POD"],
    ascending=[False, False]
).reset_index(drop=True)

comparison_table.to_csv(
    f"{OUT_TABLES}/tabel_4_21_perbandingan_sequence14_dan_baseline.csv",
    index=False
)


# ============================================================
# 11. GAMBAR LOSS
# ============================================================

hist = history_store[best_mode]

plt.figure(figsize=(9, 5))
plt.plot(hist["loss"], label="Training loss")
plt.plot(hist["val_loss"], label="Validation loss")
plt.title(f"Kurva Loss CNN Sequence 14 Hari ({best_mode})")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_FIGURES}/gambar_4_12_loss_cnn_sequence14.png")
plt.close()


# ============================================================
# 12. GAMBAR ROC
# ============================================================

if len(np.unique(y_true_best)) == 2:
    fpr, tpr, roc_thresholds = roc_curve(y_true_best, y_prob_best)
    auroc = roc_auc_score(y_true_best, y_prob_best)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Acuan acak")
    plt.title(f"Kurva ROC CNN Sequence 14 Hari pada Data Uji ({best_mode})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_FIGURES}/gambar_4_13_roc_cnn_sequence14.png")
    plt.close()


# ============================================================
# 13. GAMBAR CONFUSION MATRIX
# ============================================================

cm = np.array([[tn, fp], [fn, tp]])

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title(f"Confusion Matrix CNN Sequence 14 Hari pada Data Uji ({best_mode})")
plt.xticks([0, 1], ["Prediksi 0", "Prediksi 1"])
plt.yticks([0, 1], ["Aktual 0", "Aktual 1"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.tight_layout()
plt.savefig(f"{OUT_FIGURES}/gambar_4_14_confusion_matrix_cnn_sequence14.png")
plt.close()


# ============================================================
# 14. RINGKASAN
# ============================================================

print("\n============================================================")
print("SELESAI CNN SEQUENCE 14 HARI + FITUR TAMBAHAN")
print("============================================================")
print(f"Ambang label utama: Rainfall_Target >= {HEAVY_RAIN_THRESHOLD} mm/hari")
print(f"Model terbaik berdasarkan validasi: {best_mode_by_val}")
print(f"Model terbaik berdasarkan test untuk analisis: {best_mode_by_test}")

print("\nDistribusi label:")
print(dist_table)

print("\nEvaluasi baseline:")
print(baseline_table)

print("\nEvaluasi CNN sequence14:")
print(cnn_eval_table)

print("\nConfusion matrix model terpilih validasi:")
print(cm_table)

print("\nPerbandingan test:")
print(comparison_table)

print("\nOutput tersimpan di:")
print(OUT_BASE)