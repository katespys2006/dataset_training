"""
Модель: Logistic Regression (проста)
Читає підготовлені дані з prepared/, навчає модель, міряє час
і зберігає результати в папку results/.
"""
import json
import os
import time

import numpy as np
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression

SEED = 42
os.makedirs("results", exist_ok=True)

X_train = sparse.load_npz("prepared/X_train.npz")
X_test = sparse.load_npz("prepared/X_test.npz")
y_train = np.load("prepared/y_train.npy")
y_test = np.load("prepared/y_test.npy")

model = LogisticRegression(max_iter=1000, random_state=SEED)

t0 = time.perf_counter()
model.fit(X_train, y_train)
train_time = time.perf_counter() - t0

t0 = time.perf_counter()
pred = model.predict(X_test)
predict_time = time.perf_counter() - t0

result = {
    "model": "Logistic Regression (проста)",
    "accuracy": round(float(accuracy_score(y_test, pred)), 4),
    "f1_macro": round(float(f1_score(y_test, pred, average="macro")), 4),
    "train_time_s": round(train_time, 3),
    "predict_time_s": round(predict_time, 4),
}
np.save("results/logreg_pred.npy", pred)
with open("results/logreg.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(result)
