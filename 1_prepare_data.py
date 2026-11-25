"""
Крок 1. Підготовка даних.
Читає dataset.csv, ділить на навчальну і тестову частини (за колонкою split,
тобто за книжками), перетворює текст у числа (TF-IDF) і зберігає все в папку prepared/.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

os.makedirs("prepared", exist_ok=True)

df = pd.read_csv("dataset.csv")
train = df[df.split == "train"]
test = df[df.split == "test"]
print(f"Навчання: {len(train)} рядків, тест: {len(test)} рядків")
print("Книжки в тесті:", sorted(test.book.unique()))

# TF-IDF: словник вчимо ТІЛЬКИ на train, щоб тест лишався "чесним"
vec = TfidfVectorizer(max_features=5000, stop_words="english", lowercase=True)
X_train = vec.fit_transform(train.text)
X_test = vec.transform(test.text)

# Назви жанрів -> числа 0, 1, 2
le = LabelEncoder()
y_train = le.fit_transform(train.genre)
y_test = le.transform(test.genre)

sparse.save_npz("prepared/X_train.npz", X_train)
sparse.save_npz("prepared/X_test.npz", X_test)
np.save("prepared/y_train.npy", y_train)
np.save("prepared/y_test.npy", y_test)
with open("prepared/classes.json", "w", encoding="utf-8") as f:
    json.dump(list(le.classes_), f)

print("Класи:", list(le.classes_))
print("Розмір матриці ознак:", X_train.shape)
print("Готово: дані збережено в папку prepared/")
