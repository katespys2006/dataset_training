"""
Крок 5. Обробка результатів: збирає results/*.json у таблицю
і будує графіки (точність, час, матриці помилок).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

KEYS = ["logreg", "rf", "mlp"]  # від простої до складної

classes = json.load(open("prepared/classes.json", encoding="utf-8"))
y_test = np.load("prepared/y_test.npy")

rows, preds = [], {}
for k in KEYS:
    r = json.load(open(f"results/{k}.json", encoding="utf-8"))
    rows.append(r)
    preds[k] = np.load(f"results/{k}_pred.npy")
    print(f"\n=== {r['model']} ===")
    print(classification_report(y_test, preds[k], target_names=classes, digits=3))

res = pd.DataFrame(rows).rename(columns={
    "model": "Модель", "accuracy": "Accuracy", "f1_macro": "F1 (macro)",
    "train_time_s": "Час навчання, с", "predict_time_s": "Час передбачення, с"})
res.to_csv("results.csv", index=False)
print("\n", res.to_string(index=False))

short = [m.split(" (")[0] for m in res["Модель"]]

# Графік 1: точність і F1
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(short))
ax.bar(x - 0.2, res["Accuracy"], 0.4, label="Accuracy")
ax.bar(x + 0.2, res["F1 (macro)"], 0.4, label="F1 (macro)")
ax.set_xticks(x, short)
ax.set_ylim(0, 1)
ax.set_title("Точність моделей")
ax.legend()
plt.tight_layout()
plt.savefig("plot_accuracy.png", dpi=150)

# Графік 2: час навчання і передбачення
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(short, res["Час навчання, с"], color="tab:orange")
axes[0].set_title("Час навчання, с")
axes[1].bar(short, res["Час передбачення, с"], color="tab:green")
axes[1].set_title("Час передбачення, с")
plt.tight_layout()
plt.savefig("plot_time.png", dpi=150)

# Графік 3: матриці помилок
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, k, name in zip(axes, KEYS, short):
    ConfusionMatrixDisplay.from_predictions(y_test, preds[k], display_labels=classes,
                                            ax=ax, colorbar=False)
    ax.set_title(name)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)

print("\nГотово: results.csv, plot_accuracy.png, plot_time.png, confusion_matrices.png")
