"""
Створення датасету "жанр книжки за уривком тексту".
Запуск: python build_dataset.py   (потрібен інтернет, бібліотеки: requests, pandas)
Результат: dataset.csv
"""
import random
import re
import sys

import pandas as pd
import requests

# 3 жанри, по 5 книжок, у кожному жанрі щонайменше 2 різних автори
# (щоб модель вчила жанр, а не стиль одного автора).
BOOKS = {
    "detective": [
        (1661, "The Adventures of Sherlock Holmes", "Doyle"),
        (2852, "The Hound of the Baskervilles", "Doyle"),
        (863,  "The Mysterious Affair at Styles", "Christie"),
        (204,  "The Innocence of Father Brown", "Chesterton"),
        (155,  "The Moonstone", "Collins"),
    ],
    "scifi": [
        (35,  "The Time Machine", "Wells"),
        (36,  "The War of the Worlds", "Wells"),
        (159, "The Island of Doctor Moreau", "Wells"),
        (164, "Twenty Thousand Leagues Under the Sea", "Verne"),
        (83,  "From the Earth to the Moon", "Verne"),
    ],
    "romance": [
        (1342, "Pride and Prejudice", "Austen"),
        (158,  "Emma", "Austen"),
        (161,  "Sense and Sensibility", "Austen"),
        (1260, "Jane Eyre", "Bronte"),
        (768,  "Wuthering Heights", "Bronte"),
    ],
}

CHUNK_WORDS = 200        # слів в одному рядку
CHUNKS_PER_BOOK = 100    # однакова кількість рядків з кожної книжки
SKIP_WORDS = 2000        # пропускаємо початок і кінець (передмови, змісти)
SEED = 42
TEST_BOOKS_PER_GENRE = 1  # остання книжка кожного жанру йде у тест


def download(book_id):
    for url in (
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
    ):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                r.encoding = "utf-8"
                return r.text
        except requests.RequestException:
            pass
    raise RuntimeError(f"Не вдалося завантажити книжку {book_id}")


def strip_gutenberg(text):
    """Прибирає службовий заголовок і футер Project Gutenberg."""
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", text)
    if start:
        text = text[start.end():]
    if end:
        text = text[: text.find(end.group(0))] if end.group(0) in text else text
    return text


def make_chunks(text, rng):
    words = re.sub(r"\s+", " ", text).split()
    words = words[SKIP_WORDS: len(words) - SKIP_WORDS]
    chunks = [
        " ".join(words[i: i + CHUNK_WORDS])
        for i in range(0, len(words) - CHUNK_WORDS + 1, CHUNK_WORDS)
    ]
    if len(chunks) > CHUNKS_PER_BOOK:
        chunks = rng.sample(chunks, CHUNKS_PER_BOOK)
    return chunks


def build(fetch=download):
    rng = random.Random(SEED)
    rows = []
    for genre, books in BOOKS.items():
        for idx, (book_id, title, author) in enumerate(books):
            split = "test" if idx >= len(books) - TEST_BOOKS_PER_GENRE else "train"
            text = strip_gutenberg(fetch(book_id))
            chunks = make_chunks(text, rng)
            print(f"{genre:10} {title:40} {len(chunks)} шматків ({split})")
            for c in chunks:
                rows.append(
                    {"text": c, "genre": genre, "book": title,
                     "author": author, "split": split}
                )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build()
    df.to_csv("dataset.csv", index=False, encoding="utf-8")
    print("\nГотово: dataset.csv,", len(df), "рядків")
    print(df.groupby(["genre", "split"]).size())
