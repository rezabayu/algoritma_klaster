import os
import json
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

from sklearn.ensemble import RandomForestClassifier
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

warnings.filterwarnings("ignore")


# ============================================================
# KONFIGURASI
# ============================================================

SEED = 42

INPUT_FILE = "outputs/processed/data_fitur_harian_r1_r3_r5_r7_r10.csv"

OUT_BASE = "outputs/feature_engineering_rf_xgboost"
OUT_TABLES = f"{OUT_BASE}/tables"
OUT_FIGURES = f"{OUT_BASE}/figures"
OUT_PROCESSED = f"{OUT_BASE}/processed"

for d in [OUT_TABLES, OUT_FIGURES, OUT_PROCESSED]:
    os.makedirs(d, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)

HEAVY_RAIN_THRESHOLD = 50.0
RAIN_SEQUENCE_DAYS = 14

TRAIN_END = "2012-12-31"
VAL_END = "2018-12-31"

ACCUM_FEATURE_COLS = ["R1", "R3", "R5", "R7", "R10"]

# XGBoost bersifat opsional.
# Jika belum terpasang, jalankan:
# python -m pip install xgboost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


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
            "TP": m["TP"],
            "FP": m["FP"],
            "FN": m["FN"],
            "TN": m["TN"],
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

df["Split"] = np.where(
    df["Target_Date"] <= pd.Timestamp(TRAIN_END),
    "train",
    np.where(
        df["Target_Date"] <= pd.Timestamp(VAL_END),
        "validation",
        "test"
    )
)

print("Jumlah data sebelum penyamaan window 14 hari:")
print(df["Split"].value_counts())


# ============================================================
# 2. ANOMALY DASARIAN BERDASARKAN TRAIN SAJA
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
df["Rain_Anomaly"] = df["Rain_Anomaly"].fillna(0.0)


# ============================================================
# 3. FITUR MUSIMAN
# ============================================================

target_dayofyear = df["Target_Date"].dt.dayofyear.astype(float)
df["sin_dayofyear"] = np.sin(2 * np.pi * target_dayofyear / 366.0)
df["cos_dayofyear"] = np.cos(2 * np.pi * target_dayofyear / 366.0)


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

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

df["mean_rain_last3"] = df["Rainfall"].rolling(window=3, min_periods=1).mean()
df["mean_rain_prev3"] = df["Rainfall"].shift(3).rolling(window=3, min_periods=1).mean()
df["trend_rain_3d"] = (df["mean_rain_last3"] - df["mean_rain_prev3"]).fillna(0.0)

df["mean_rain_last7"] = df["Rainfall"].rolling(window=7, min_periods=1).mean()
df["mean_rain_prev7"] = df["Rainfall"].shift(7).rolling(window=7, min_periods=1).mean()
df["trend_rain_7d"] = (df["mean_rain_last7"] - df["mean_rain_prev7"]).fillna(0.0)


ENGINEERED_FEATURE_COLS = [
    "R1", "R3", "R5", "R7", "R10",

    "max_rain_3d", "max_rain_5d", "max_rain_7d", "max_rain_14d",
    "sum_rain_3d", "sum_rain_5d", "sum_rain_7d", "sum_rain_14d",
    "mean_rain_3d", "mean_rain_5d", "mean_rain_7d", "mean_rain_14d",

    "rainy_days_3d", "rainy_days_7d", "rainy_days_14d",
    "dry_days_3d", "dry_days_7d", "dry_days_14d",

    "days_since_rain", "days_since_heavy_rain",

    "mean_anom_3d", "mean_anom_7d", "mean_anom_14d",
    "max_anom_3d", "max_anom_7d", "max_anom_14d",

    "trend_rain_3d", "trend_rain_7d",

    "sin_dayofyear", "cos_dayofyear",
]

df[ENGINEERED_FEATURE_COLS] = df[ENGINEERED_FEATURE_COLS].fillna(0.0)


# ============================================================
# 5. SAMAKAN WINDOW DENGAN CNN SEQUENCE 14 HARI
# ============================================================

rows = []

for i in range(RAIN_SEQUENCE_DAYS - 1, len(df)):
    window_dates = df["Tanggal_Observasi"].iloc[i - RAIN_SEQUENCE_DAYS + 1:i + 1]

    if (window_dates.diff().dropna().dt.days != 1).any():
        continue

    rows.append(df.iloc[i])

df_model = pd.DataFrame(rows).reset_index(drop=True)

print("\nJumlah data setelah disamakan dengan window 14 hari:")
print(df_model["Split"].value_counts())

df_model.to_csv(
    f"{OUT_PROCESSED}/dataset_feature_engineering_rf_xgboost_metadata.csv",
    index=False
)


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
    f"{OUT_TABLES}/tabel_tree_01_distribusi_label.csv",
    index=False
)

feature_table = pd.DataFrame({
    "No": np.arange(1, len(ENGINEERED_FEATURE_COLS) + 1),
    "Fitur": ENGINEERED_FEATURE_COLS,
})
feature_table.to_csv(
    f"{OUT_TABLES}/tabel_tree_02_daftar_fitur.csv",
    index=False
)


# ============================================================
# 7. SPLIT DATA
# ============================================================

train_df = df_model[df_model["Split"] == "train"].copy()
val_df = df_model[df_model["Split"] == "validation"].copy()
test_df = df_model[df_model["Split"] == "test"].copy()

X_train = train_df[ENGINEERED_FEATURE_COLS].values
y_train = train_df["Label_Lebat_50"].values.astype(int)

X_val = val_df[ENGINEERED_FEATURE_COLS].values
y_val = val_df["Label_Lebat_50"].values.astype(int)

X_test = test_df[ENGINEERED_FEATURE_COLS].values
y_test = test_df["Label_Lebat_50"].values.astype(int)

# Scaling tidak wajib untuk Random Forest / XGBoost.
# Namun disiapkan untuk konsistensi jika dibutuhkan.
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

n_pos_train = int(y_train.sum())
n_neg_train = int(len(y_train) - n_pos_train)
scale_pos_weight_full = n_neg_train / n_pos_train
scale_pos_weight_sqrt = np.sqrt(scale_pos_weight_full)

print("\nDistribusi train:")
print(f"Negatif: {n_neg_train}")
print(f"Positif: {n_pos_train}")
print(f"scale_pos_weight full: {scale_pos_weight_full:.4f}")
print(f"scale_pos_weight sqrt: {scale_pos_weight_sqrt:.4f}")


# ============================================================
# 8. BASELINE
# ============================================================

baseline_rows = []

for feature in ACCUM_FEATURE_COLS:
    for split_name, split_df in [
        ("validation", val_df),
        ("test", test_df),
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
    f"{OUT_TABLES}/tabel_tree_03_evaluasi_baseline.csv",
    index=False
)


# ============================================================
# 9. DEFINISI MODEL
# ============================================================

models_to_run = {}

models_to_run["RF_none_depth5"] = RandomForestClassifier(
    n_estimators=500,
    max_depth=5,
    min_samples_leaf=5,
    random_state=SEED,
    n_jobs=-1,
    class_weight=None
)

models_to_run["RF_balanced_depth5"] = RandomForestClassifier(
    n_estimators=500,
    max_depth=5,
    min_samples_leaf=5,
    random_state=SEED,
    n_jobs=-1,
    class_weight="balanced"
)

models_to_run["RF_balanced_subsample_depth5"] = RandomForestClassifier(
    n_estimators=500,
    max_depth=5,
    min_samples_leaf=5,
    random_state=SEED,
    n_jobs=-1,
    class_weight="balanced_subsample"
)

if XGBOOST_AVAILABLE:
    models_to_run["XGB_none"] = XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1
    )

    models_to_run["XGB_spw_sqrt"] = XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight_sqrt,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1
    )

    models_to_run["XGB_spw_full"] = XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight_full,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1
    )
else:
    print("\nXGBoost belum terpasang. Model XGBoost akan dilewati.")
    print("Install dengan: python -m pip install xgboost")


# ============================================================
# 10. TRAIN DAN EVALUASI
# ============================================================

eval_rows = []
prediction_store = {}
threshold_store = {}
importance_store = {}

for model_name, model in models_to_run.items():
    print(f"\nTraining model: {model_name}")

    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    best_thr, threshold_scan = choose_threshold_by_validation(y_val, val_prob)

    val_pred = (val_prob >= best_thr).astype(int)
    test_pred = (test_prob >= best_thr).astype(int)

    val_metrics = hit_metrics(y_val, val_pred, val_prob)
    test_metrics = hit_metrics(y_test, test_pred, test_prob)

    eval_rows.append({
        "Model": model_name,
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
    })

    pred_df = test_df[[
        "Tanggal_Observasi",
        "Target_Date",
        "Rainfall_Target",
        "Label_Lebat_50"
    ] + ENGINEERED_FEATURE_COLS].copy()

    pred_df = pred_df.rename(columns={"Label_Lebat_50": "Observed_Label"})
    pred_df["Predicted_Probability"] = test_prob
    pred_df["Predicted_Label"] = test_pred
    pred_df["Decision_Threshold"] = best_thr
    pred_df["Model"] = model_name

    prediction_store[model_name] = pred_df
    threshold_store[model_name] = threshold_scan

    threshold_scan.to_csv(
        f"{OUT_PROCESSED}/threshold_scan_validation_{model_name}.csv",
        index=False
    )

    pred_df.to_csv(
        f"{OUT_PROCESSED}/predictions_test_{model_name}.csv",
        index=False
    )

    if hasattr(model, "feature_importances_"):
        imp_df = pd.DataFrame({
            "Feature": ENGINEERED_FEATURE_COLS,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

        imp_df.to_csv(
            f"{OUT_PROCESSED}/feature_importance_{model_name}.csv",
            index=False
        )

        importance_store[model_name] = imp_df


eval_table = pd.DataFrame(eval_rows)

eval_table = eval_table.sort_values(
    ["Val_CSI", "Val_AUROC", "Val_FAR"],
    ascending=[False, False, True]
).reset_index(drop=True)

eval_table.to_csv(
    f"{OUT_TABLES}/tabel_tree_04_evaluasi_model.csv",
    index=False
)

print("\nEvaluasi Random Forest dan XGBoost:")
print(eval_table)


# ============================================================
# 11. PERBANDINGAN TEST DENGAN MODEL SEBELUMNYA
# ============================================================

comparison_rows = []

# Model tree
for _, row in eval_table.iterrows():
    comparison_rows.append({
        "Model": row["Model"],
        "POD": row["Test_POD"],
        "FAR": row["Test_FAR"],
        "CSI": row["Test_CSI"],
        "AUROC": row["Test_AUROC"],
        "TP": row["Test_TP"],
        "FP": row["Test_FP"],
        "FN": row["Test_FN"],
        "TN": row["Test_TN"],
        "Tipe": "FeatureEngineering_Tree"
    })

# Baseline terbaik dari eksperimen sebelumnya
baseline_test = baseline_table[baseline_table["Split"] == "test"].copy()

for _, row in baseline_test.iterrows():
    comparison_rows.append({
        "Model": row["Model"],
        "POD": row["POD"],
        "FAR": row["FAR"],
        "CSI": row["CSI"],
        "AUROC": row["AUROC"],
        "TP": row["TP"],
        "FP": row["FP"],
        "FN": row["FN"],
        "TN": row["TN"],
        "Tipe": "Baseline"
    })

# Hasil laporan utama
comparison_rows.append({
    "Model": "CNN_seq14_aux_none_laporan",
    "POD": 0.0857,
    "FAR": 0.9583,
    "CSI": 0.0288,
    "AUROC": 0.6665,
    "TP": 3,
    "FP": 69,
    "FN": 32,
    "TN": 2088,
    "Tipe": "CNN_Laporan"
})

# Hasil CNN feature engineering sebelumnya
comparison_rows.append({
    "Model": "CNN_feature_engineering_none",
    "POD": 0.5714,
    "FAR": 0.9699,
    "CSI": 0.0294,
    "AUROC": 0.6715,
    "TP": 20,
    "FP": 645,
    "FN": 15,
    "TN": 1512,
    "Tipe": "CNN_FeatureEngineering"
})

comparison_table = pd.DataFrame(comparison_rows)

comparison_table = comparison_table.sort_values(
    ["CSI", "AUROC", "POD"],
    ascending=[False, False, False]
).reset_index(drop=True)

comparison_table.to_csv(
    f"{OUT_TABLES}/tabel_tree_05_perbandingan_test.csv",
    index=False
)


# ============================================================
# 12. PILIH MODEL TERBAIK BERDASARKAN VALIDASI
# ============================================================

best_by_validation = eval_table.iloc[0].to_dict()
best_model_name = best_by_validation["Model"]

with open(f"{OUT_PROCESSED}/selected_tree_model_by_validation.json", "w", encoding="utf-8") as f:
    json.dump(best_by_validation, f, indent=4, ensure_ascii=False)

best_pred_df = prediction_store[best_model_name]

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
        "Hujan lebat berhasil terdeteksi",
    ]
})

cm_table.to_csv(
    f"{OUT_TABLES}/tabel_tree_06_confusion_matrix_best.csv",
    index=False
)


# ============================================================
# 13. GAMBAR ROC, CONFUSION MATRIX, DAN FEATURE IMPORTANCE
# ============================================================

if len(np.unique(y_true_best)) == 2:
    fpr, tpr, _ = roc_curve(y_true_best, y_prob_best)
    auroc = roc_auc_score(y_true_best, y_prob_best)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Acuan acak")
    plt.title(f"Kurva ROC {best_model_name} pada Data Uji")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_FIGURES}/gambar_tree_01_roc_best.png", dpi=300)
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

ax.set_title(f"Confusion Matrix {best_model_name} pada Data Uji")
ax.set_xlabel("Kelas Prediksi")
ax.set_ylabel("Kelas Aktual")

plt.tight_layout()
plt.savefig(
    f"{OUT_FIGURES}/gambar_tree_02_confusion_matrix_best.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


if best_model_name in importance_store:
    imp_df = importance_store[best_model_name].head(15).copy()

    plt.figure(figsize=(9, 6))
    plt.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1])
    plt.title(f"Top 15 Feature Importance {best_model_name}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(f"{OUT_FIGURES}/gambar_tree_03_feature_importance_best.png", dpi=300)
    plt.close()

    imp_df.to_csv(
        f"{OUT_TABLES}/tabel_tree_07_feature_importance_best.csv",
        index=False
    )


# ============================================================
# 14. RINGKASAN
# ============================================================

print("\n============================================================")
print("SELESAI FEATURE ENGINEERING + RANDOM FOREST / XGBOOST")
print("============================================================")

print("\nDistribusi label:")
print(dist_table)

print("\nEvaluasi model:")
print(eval_table)

print("\nPerbandingan test:")
print(comparison_table)

print("\nModel terbaik berdasarkan validasi:")
print(best_by_validation)

print("\nConfusion matrix model terbaik:")
print(cm_table)

print("\nOutput tersimpan di:")
print(OUT_BASE)