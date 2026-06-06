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
from matplotlib.patches import Rectangle


warnings.filterwarnings("ignore")


# ============================================================
# KONFIGURASI
# ============================================================

SEED = 42

INPUT_FILE = "outputs/processed/data_fitur_harian_r1_r3_r5_r7_r10.csv"

OUT_BASE = "outputs/cnn_sequence14_feature_engineering"
OUT_TABLES = f"{OUT_BASE}/tables"
OUT_FIGURES = f"{OUT_BASE}/figures"
OUT_PROCESSED = f"{OUT_BASE}/processed"

for d in [OUT_TABLES, OUT_FIGURES, OUT_PROCESSED]:
    os.makedirs(d, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

RAIN_SEQUENCE_DAYS = 14
HEAVY_RAIN_THRESHOLD = 50.0

TRAIN_END = "2012-12-31"
VAL_END = "2018-12-31"

EPOCHS = 300
PATIENCE = 40
BATCH_SIZE = 32
LEARNING_RATE = 0.001

ACCUM_FEATURE_COLS = ["R1", "R3", "R5", "R7", "R10"]
SEASON_FEATURE_COLS = ["sin_dayofyear", "cos_dayofyear"]

CLASS_WEIGHT_SCENARIOS = {
    "none": None,
    "weight_1_2": {0: 1.0, 1: 2.0},
    "weight_1_3": {0: 1.0, 1: 3.0},
}


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
    """
    Threshold dipilih dari validation.
    Prioritas:
    1. CSI tertinggi
    2. F1 tertinggi
    3. FAR terendah
    """
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


def get_dasarian_ke_bulan(day):
    if day <= 10:
        return 1
    elif day <= 20:
        return 2
    return 3


def calc_days_since(series, condition_func, cap_value):
    """
    Menghitung jumlah hari sejak kondisi terakhir terpenuhi.
    Untuk hari sebelum pernah terjadi kondisi, nilai diberi cap_value.
    """
    values = series.values
    out = []
    last_seen = None

    for i, val in enumerate(values):
        if condition_func(val):
            last_seen = i
            out.append(0)
        else:
            if last_seen is None:
                out.append(cap_value)
            else:
                out.append(min(i - last_seen, cap_value))

    return np.array(out)


def build_sequence_aux_cnn(seq_shape, aux_shape):
    seq_input = layers.Input(shape=seq_shape, name="rain_sequence_input")

    x = layers.Conv1D(filters=16, kernel_size=3, activation="relu", padding="same")(seq_input)
    x = layers.Conv1D(filters=16, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(16, activation="relu")(x)

    aux_input = layers.Input(shape=aux_shape, name="aux_feature_input")

    y = layers.Dense(32, activation="relu")(aux_input)
    y = layers.Dropout(0.2)(y)
    y = layers.Dense(16, activation="relu")(y)

    z = layers.Concatenate()([x, y])
    z = layers.Dense(32, activation="relu")(z)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(16, activation="relu")(z)
    output = layers.Dense(1, activation="sigmoid")(z)

    model = models.Model(
        inputs=[seq_input, aux_input],
        outputs=output,
        name="cnn_sequence14_feature_engineering"
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
# 1. BACA DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)
df["Tanggal_Observasi"] = pd.to_datetime(df["Tanggal_Observasi"])

required_cols = ["Tanggal_Observasi", "Rainfall"] + ACCUM_FEATURE_COLS
missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

df = df.sort_values("Tanggal_Observasi").reset_index(drop=True)

df["Tahun"] = df["Tanggal_Observasi"].dt.year
df["Bulan"] = df["Tanggal_Observasi"].dt.month
df["Tanggal"] = df["Tanggal_Observasi"].dt.day
df["Dasarian_Ke_Bulan"] = df["Tanggal"].apply(get_dasarian_ke_bulan)
df["Dasarian_Tahunan"] = (df["Bulan"] - 1) * 3 + df["Dasarian_Ke_Bulan"]

# Target t+1
df["Target_Date"] = df["Tanggal_Observasi"].shift(-1)
df["Rainfall_Target"] = df["Rainfall"].shift(-1)

df = df.dropna(subset=["Target_Date", "Rainfall_Target"]).copy()

df["Target_Year"] = df["Target_Date"].dt.year
df["Target_Month"] = df["Target_Date"].dt.month
df["Target_Day"] = df["Target_Date"].dt.day
df["Target_Dasarian_Ke_Bulan"] = df["Target_Day"].apply(get_dasarian_ke_bulan)
df["Target_Dasarian_Tahunan"] = (
    (df["Target_Month"] - 1) * 3 + df["Target_Dasarian_Ke_Bulan"]
)

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

# Split berbasis Target_Date
train_mask = df["Target_Date"] <= pd.Timestamp(TRAIN_END)
val_mask = (df["Target_Date"] > pd.Timestamp(TRAIN_END)) & (df["Target_Date"] <= pd.Timestamp(VAL_END))
test_mask = df["Target_Date"] > pd.Timestamp(VAL_END)

df["Split"] = np.where(train_mask, "train", np.where(val_mask, "validation", "test"))

print("Jumlah data sebelum sequence:")
print(df["Split"].value_counts())


# ============================================================
# 2. HITUNG ANOMALY DASARIAN BERDASARKAN TRAIN SAJA
# ============================================================

df_train_for_clim = df[df["Split"] == "train"].copy()

dasarian_clim = (
    df_train_for_clim
    .groupby("Dasarian_Tahunan")["Rainfall"]
    .mean()
    .to_dict()
)

df["Clim_Rainfall_Dasarian"] = df["Dasarian_Tahunan"].map(dasarian_clim)
df["Rain_Anomaly"] = df["Rainfall"] - df["Clim_Rainfall_Dasarian"]

# Untuk keamanan jika ada NA, isi 0
df["Rain_Anomaly"] = df["Rain_Anomaly"].fillna(0.0)


# ============================================================
# 3. FITUR MUSIMAN
# ============================================================

target_dayofyear = df["Target_Date"].dt.dayofyear.astype(float)
df["sin_dayofyear"] = np.sin(2 * np.pi * target_dayofyear / 366.0)
df["cos_dayofyear"] = np.cos(2 * np.pi * target_dayofyear / 366.0)


# ============================================================
# 4. FITUR TURUNAN HISTORIS
# ============================================================

# Rolling window memakai informasi sampai hari t.
for w in [3, 5, 7, 14]:
    df[f"max_rain_{w}d"] = df["Rainfall"].rolling(window=w, min_periods=1).max()
    df[f"sum_rain_{w}d"] = df["Rainfall"].rolling(window=w, min_periods=1).sum()
    df[f"mean_rain_{w}d"] = df["Rainfall"].rolling(window=w, min_periods=1).mean()

for w in [3, 7, 14]:
    df[f"rainy_days_{w}d"] = (
        (df["Rainfall"] > 0)
        .rolling(window=w, min_periods=1)
        .sum()
    )
    df[f"dry_days_{w}d"] = (
        (df["Rainfall"] == 0)
        .rolling(window=w, min_periods=1)
        .sum()
    )

for w in [3, 7, 14]:
    df[f"mean_anom_{w}d"] = df["Rain_Anomaly"].rolling(window=w, min_periods=1).mean()
    df[f"max_anom_{w}d"] = df["Rain_Anomaly"].rolling(window=w, min_periods=1).max()

df["days_since_rain"] = calc_days_since(
    df["Rainfall"],
    condition_func=lambda x: x > 0,
    cap_value=30
)

df["days_since_heavy_rain"] = calc_days_since(
    df["Rainfall"],
    condition_func=lambda x: x >= HEAVY_RAIN_THRESHOLD,
    cap_value=365
)

# Trend sederhana: selisih rata-rata 3 hari terakhir dengan 3 hari sebelumnya
df["mean_rain_last3"] = df["Rainfall"].rolling(window=3, min_periods=1).mean()
df["mean_rain_prev3"] = df["Rainfall"].shift(3).rolling(window=3, min_periods=1).mean()
df["trend_rain_3d"] = (df["mean_rain_last3"] - df["mean_rain_prev3"]).fillna(0.0)

df["mean_rain_last7"] = df["Rainfall"].rolling(window=7, min_periods=1).mean()
df["mean_rain_prev7"] = df["Rainfall"].shift(7).rolling(window=7, min_periods=1).mean()
df["trend_rain_7d"] = (df["mean_rain_last7"] - df["mean_rain_prev7"]).fillna(0.0)


ENGINEERED_FEATURE_COLS = [
    # fitur lama
    "R1", "R3", "R5", "R7", "R10",

    # max, sum, mean
    "max_rain_3d", "max_rain_5d", "max_rain_7d", "max_rain_14d",
    "sum_rain_3d", "sum_rain_5d", "sum_rain_7d", "sum_rain_14d",
    "mean_rain_3d", "mean_rain_5d", "mean_rain_7d", "mean_rain_14d",

    # rainy/dry days
    "rainy_days_3d", "rainy_days_7d", "rainy_days_14d",
    "dry_days_3d", "dry_days_7d", "dry_days_14d",

    # days since
    "days_since_rain", "days_since_heavy_rain",

    # anomaly aggregate
    "mean_anom_3d", "mean_anom_7d", "mean_anom_14d",
    "max_anom_3d", "max_anom_7d", "max_anom_14d",

    # trend
    "trend_rain_3d", "trend_rain_7d",

    # seasonal
    "sin_dayofyear", "cos_dayofyear",
]

# Isi NA sisa jika ada
df[ENGINEERED_FEATURE_COLS] = df[ENGINEERED_FEATURE_COLS].fillna(0.0)


# ============================================================
# 5. BENTUK SEQUENCE 14 HARI DENGAN 2 CHANNEL
# ============================================================

rows = []
seq_list = []
aux_list = []
y_list = []

rain_values = df["Rainfall"].values.astype(float)
anom_values = df["Rain_Anomaly"].values.astype(float)

for i in range(RAIN_SEQUENCE_DAYS - 1, len(df)):
    current_row = df.iloc[i]

    # sequence sampai hari t, target t+1
    rain_seq = rain_values[i - RAIN_SEQUENCE_DAYS + 1:i + 1]
    anom_seq = anom_values[i - RAIN_SEQUENCE_DAYS + 1:i + 1]

    window_dates = df["Tanggal_Observasi"].iloc[i - RAIN_SEQUENCE_DAYS + 1:i + 1]
    if (window_dates.diff().dropna().dt.days != 1).any():
        continue

    seq = np.stack([rain_seq, anom_seq], axis=1)  # shape: 14 x 2
    aux = current_row[ENGINEERED_FEATURE_COLS].values.astype(float)

    seq_list.append(seq)
    aux_list.append(aux)
    y_list.append(int(current_row["Label_Lebat_50"]))
    rows.append(current_row)

df_model = pd.DataFrame(rows).reset_index(drop=True)

X_seq = np.array(seq_list, dtype=float)
X_aux = np.array(aux_list, dtype=float)
y = np.array(y_list, dtype=int)

print("\nJumlah data setelah sequence:")
print(df_model["Split"].value_counts())
print("X_seq:", X_seq.shape)
print("X_aux:", X_aux.shape)
print("y:", y.shape)


# ============================================================
# 6. SIMPAN DISTRIBUSI LABEL DAN DAFTAR FITUR
# ============================================================

dist_rows = []

for split_name, split_df in df_model.groupby("Split"):
    n_total = len(split_df)
    n_pos = int(split_df["Label_Lebat_50"].sum())
    n_neg = int(n_total - n_pos)

    dist_rows.append({
        "Split": split_name,
        "Jumlah_Data": n_total,
        "Tidak_Hujan_Lebat_<50mm": n_neg,
        "Hujan_Lebat_>=50mm": n_pos,
        "Persentase_Hujan_Lebat": round(n_pos / n_total * 100, 2),
    })

dist_table = pd.DataFrame(dist_rows)
dist_table.to_csv(
    f"{OUT_TABLES}/tabel_fe_01_distribusi_label.csv",
    index=False
)

feature_table = pd.DataFrame({
    "No": np.arange(1, len(ENGINEERED_FEATURE_COLS) + 1),
    "Fitur_Tambahan": ENGINEERED_FEATURE_COLS,
})
feature_table.to_csv(
    f"{OUT_TABLES}/tabel_fe_02_daftar_fitur.csv",
    index=False
)

df_model.to_csv(
    f"{OUT_PROCESSED}/dataset_cnn_sequence14_feature_engineering_metadata.csv",
    index=False
)


# ============================================================
# 7. SPLIT DAN NORMALISASI
# ============================================================

idx_train = df_model.index[df_model["Split"] == "train"].to_numpy()
idx_val = df_model.index[df_model["Split"] == "validation"].to_numpy()
idx_test = df_model.index[df_model["Split"] == "test"].to_numpy()

# Scaler sequence berdasarkan train saja.
seq_scaler = StandardScaler()
seq_scaler.fit(X_seq[idx_train].reshape(-1, X_seq.shape[-1]))

X_seq_scaled = seq_scaler.transform(
    X_seq.reshape(-1, X_seq.shape[-1])
).reshape(X_seq.shape)

# Scaler aux berdasarkan train saja.
aux_scaler = StandardScaler()
aux_scaler.fit(X_aux[idx_train])

X_aux_scaled = aux_scaler.transform(X_aux)

X_seq_train = X_seq_scaled[idx_train]
X_aux_train = X_aux_scaled[idx_train]
y_train = y[idx_train]

X_seq_val = X_seq_scaled[idx_val]
X_aux_val = X_aux_scaled[idx_val]
y_val = y[idx_val]

X_seq_test = X_seq_scaled[idx_test]
X_aux_test = X_aux_scaled[idx_test]
y_test = y[idx_test]

df_test = df_model[df_model["Split"] == "test"].copy()


# ============================================================
# 8. BASELINE
# ============================================================

baseline_rows = []

for feature in ACCUM_FEATURE_COLS:
    for split_name, split_df in [
        ("validation", df_model[df_model["Split"] == "validation"].copy()),
        ("test", df_model[df_model["Split"] == "test"].copy()),
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
    f"{OUT_TABLES}/tabel_fe_03_evaluasi_baseline.csv",
    index=False
)


# ============================================================
# 9. TRAIN CNN HYBRID DENGAN FITUR TAMBAHAN
# ============================================================

eval_rows = []
history_store = {}
prediction_store = {}
threshold_scan_store = {}

for scenario_name, class_weight in CLASS_WEIGHT_SCENARIOS.items():
    print(f"\nTraining skenario: {scenario_name}")

    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)

    model = build_sequence_aux_cnn(
        seq_shape=(RAIN_SEQUENCE_DAYS, 2),
        aux_shape=(len(ENGINEERED_FEATURE_COLS),)
    )

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        min_delta=1e-5
    )

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
        "Scenario": scenario_name,
        "Class_Weight": "None" if class_weight is None else str(class_weight),
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

    eval_rows.append(row)

    history_store[scenario_name] = history.history
    threshold_scan_store[scenario_name] = threshold_scan

    pred_df = df_test[[
        "Tanggal_Observasi",
        "Target_Date",
        "Rainfall_Target",
        "Kategori_Target",
        "Label_Lebat_50"
    ] + ENGINEERED_FEATURE_COLS].copy()

    pred_df = pred_df.rename(columns={"Label_Lebat_50": "Observed_Label"})
    pred_df["Predicted_Probability"] = test_prob
    pred_df["Predicted_Label"] = test_pred
    pred_df["Decision_Threshold"] = best_thr
    pred_df["Scenario"] = scenario_name

    prediction_store[scenario_name] = pred_df

    threshold_scan.to_csv(
        f"{OUT_PROCESSED}/threshold_scan_validation_{scenario_name}.csv",
        index=False
    )

    pred_df.to_csv(
        f"{OUT_PROCESSED}/predictions_test_{scenario_name}.csv",
        index=False
    )


eval_table = pd.DataFrame(eval_rows)

eval_table = eval_table.sort_values(
    ["Val_CSI", "Val_AUROC", "Val_FAR"],
    ascending=[False, False, True]
).reset_index(drop=True)

eval_table.to_csv(
    f"{OUT_TABLES}/tabel_fe_04_evaluasi_cnn.csv",
    index=False
)

print("\nEvaluasi CNN feature engineering:")
print(eval_table)


# ============================================================
# 10. PERBANDINGAN DENGAN BASELINE DAN HASIL SEBELUMNYA
# ============================================================

baseline_test = baseline_table[baseline_table["Split"] == "test"].copy()

baseline_comp = baseline_test[[
    "Model", "POD", "FAR", "CSI", "AUROC", "TP", "FP", "FN", "TN"
]].copy()
baseline_comp["Tipe"] = "Baseline"

cnn_comp = eval_table[[
    "Scenario", "Test_POD", "Test_FAR", "Test_CSI", "Test_AUROC",
    "Test_TP", "Test_FP", "Test_FN", "Test_TN"
]].copy()

cnn_comp = cnn_comp.rename(columns={
    "Scenario": "Model",
    "Test_POD": "POD",
    "Test_FAR": "FAR",
    "Test_CSI": "CSI",
    "Test_AUROC": "AUROC",
    "Test_TP": "TP",
    "Test_FP": "FP",
    "Test_FN": "FN",
    "Test_TN": "TN",
})
cnn_comp["Tipe"] = "CNN_FeatureEngineering"

# Hasil model sequence14 lama sebagai pembanding manual
previous_model = pd.DataFrame([{
    "Model": "CNN_seq14_aux_none_lama",
    "POD": 0.0857,
    "FAR": 0.9583,
    "CSI": 0.0288,
    "AUROC": 0.6665,
    "TP": 3,
    "FP": 69,
    "FN": 32,
    "TN": 2088,
    "Tipe": "CNN_Previous"
}])

comparison_table = pd.concat([baseline_comp, cnn_comp, previous_model], ignore_index=True)

comparison_table = comparison_table.sort_values(
    ["CSI", "POD"],
    ascending=[False, False]
).reset_index(drop=True)

comparison_table.to_csv(
    f"{OUT_TABLES}/tabel_fe_05_perbandingan_test.csv",
    index=False
)


# ============================================================
# 11. MODEL TERPILIH BERDASARKAN VALIDASI
# ============================================================

best_scenario = eval_table.loc[0, "Scenario"]
best_row = eval_table.iloc[0].to_dict()

with open(f"{OUT_PROCESSED}/selected_feature_engineering_by_validation.json", "w", encoding="utf-8") as f:
    json.dump(best_row, f, indent=4, ensure_ascii=False)

best_pred_df = prediction_store[best_scenario]

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
    f"{OUT_TABLES}/tabel_fe_06_confusion_matrix_best.csv",
    index=False
)


# ============================================================
# 12. GAMBAR LOSS, ROC, DAN CONFUSION MATRIX
# ============================================================

hist = history_store[best_scenario]

plt.figure(figsize=(9, 5))
plt.plot(hist["loss"], label="Training loss")
plt.plot(hist["val_loss"], label="Validation loss")
plt.title(f"Kurva Loss CNN Feature Engineering ({best_scenario})")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_FIGURES}/gambar_fe_01_loss_best.png", dpi=300)
plt.close()

if len(np.unique(y_true_best)) == 2:
    fpr, tpr, _ = roc_curve(y_true_best, y_prob_best)
    auroc = roc_auc_score(y_true_best, y_prob_best)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Acuan acak")
    plt.title(f"Kurva ROC CNN Feature Engineering pada Data Uji ({best_scenario})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_FIGURES}/gambar_fe_02_roc_best.png", dpi=300)
    plt.close()


colors = np.array([
    ["#C8E6C9", "#FFE0B2"],
    ["#FFCDD2", "#81C784"]
])

labels = np.array([
    [f"TN\n{tn}\nTidak hujan lebat\nterprediksi benar",
     f"FP\n{fp}\nFalse alarm"],
    [f"FN\n{fn}\nHujan lebat\ntidak terdeteksi",
     f"TP\n{tp}\nHujan lebat\nterdeteksi"]
])

fig, ax = plt.subplots(figsize=(7, 6))

for i in range(2):
    for j in range(2):
        ax.add_patch(
            Rectangle(
                (j, i),
                1,
                1,
                facecolor=colors[i, j],
                edgecolor="black",
                linewidth=1.5
            )
        )
        ax.text(
            j + 0.5,
            i + 0.5,
            labels[i, j],
            ha="center",
            va="center",
            fontsize=11
        )

ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.invert_yaxis()

ax.set_xticks([0.5, 1.5])
ax.set_xticklabels(["Prediksi 0\nTidak Hujan Lebat", "Prediksi 1\nHujan Lebat"])

ax.set_yticks([0.5, 1.5])
ax.set_yticklabels(["Aktual 0\nTidak Hujan Lebat", "Aktual 1\nHujan Lebat"])

ax.set_title(f"Confusion Matrix CNN Feature Engineering pada Data Uji ({best_scenario})")
ax.set_xlabel("Kelas Prediksi")
ax.set_ylabel("Kelas Aktual")

plt.tight_layout()
plt.savefig(
    f"{OUT_FIGURES}/gambar_fe_03_confusion_matrix_best.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 13. RINGKASAN
# ============================================================

print("\n============================================================")
print("SELESAI CNN SEQUENCE 14 HARI + FEATURE ENGINEERING")
print("============================================================")
print(f"Model terbaik berdasarkan validasi: {best_scenario}")

print("\nDistribusi label:")
print(dist_table)

print("\nDaftar fitur tambahan:")
print(feature_table)

print("\nEvaluasi CNN:")
print(eval_table)

print("\nPerbandingan test:")
print(comparison_table)

print("\nConfusion matrix terbaik:")
print(cm_table)

print("\nOutput tersimpan di:")
print(OUT_BASE)