import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Nilai confusion matrix dari hasil final CNN sequence 14 hari
tn, fp, fn, tp = 2088, 69, 32, 3

cm = np.array([
    [tn, fp],
    [fn, tp]
])

labels = np.array([
    ["TN\n2088\nTidak hujan lebat\nterprediksi benar",
     "FP\n69\nFalse alarm"],
    ["FN\n32\nHujan lebat\ntidak terdeteksi",
     "TP\n3\nHujan lebat\nterdeteksi"]
])

# Warna manual:
# TN = hijau muda, FP = oranye, FN = merah, TP = hijau lebih kuat
colors = np.array([
    ["#C8E6C9", "#FFE0B2"],
    ["#FFCDD2", "#81C784"]
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

ax.set_title("Confusion Matrix CNN 1D Sequence 14 Hari pada Data Uji")
ax.set_xlabel("Kelas Prediksi")
ax.set_ylabel("Kelas Aktual")

plt.tight_layout()
plt.savefig(
    "outputs/cnn_sequence14/figures/gambar_4_11_confusion_matrix_cnn_sequence14_berwarna.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()