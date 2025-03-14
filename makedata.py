import os
import pandas as pd
import numpy as np
import json
import random

# 저장할 디렉토리 생성
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)

# 1. 책 데이터 생성 (15권)
num_books = 15
book_ids = list(range(1, num_books + 1))
titles = [f"Book {i}" for i in book_ids]
authors = np.random.choice(["Author A", "Author B", "Author C", "Author D"], size=num_books)
genres = np.random.choice(["Fiction", "Non-Fiction", "Sci-Fi", "Fantasy", "Mystery"], size=num_books)

# 다양한 summary 생성을 위한 요소들
adjectives = ["amazing", "intriguing", "fascinating", "breathtaking", "captivating"]
topics = ["love", "war", "friendship", "betrayal", "adventure", "magic", "technology", "destiny", "mystery"]
themes = ["humanity", "society", "philosophy", "culture", "innovation", "freedom"]
perspectives = ["a unique perspective", "deep insights", "an emotional journey", "unexpected twists", "a compelling narrative"]

summaries = []
for i in range(num_books):
    adj = random.choice(adjectives)
    topic = random.choice(topics)
    theme = random.choice(themes)
    perspective = random.choice(perspectives)
    summary = f"This book is a {adj} tale of {topic}. It delves into {theme} and provides {perspective}."
    summaries.append(summary)

books_df = pd.DataFrame({
    "book_id": book_ids,
    "title": titles,
    "author": authors,
    "genre": genres,
    "summary": summaries
})

# 2. 유저 데이터 생성 (5명)
num_users = 5
user_ids = list(range(1, num_users + 1))
ages = np.random.randint(20, 50, size=num_users)
genders = np.random.choice(["Male", "Female"], size=num_users)
# 각 유저의 관심 장르 (랜덤하게 2개 선택하여 JSON 문자열로 저장)
all_genres = ["Fiction", "Non-Fiction", "Sci-Fi", "Fantasy", "Mystery"]
favorite_genres = [json.dumps(list(np.random.choice(all_genres, size=2, replace=False))) for _ in range(num_users)]

users_df = pd.DataFrame({
    "user_id": user_ids,
    "age": ages,
    "gender": genders,
    "favorite_genres": favorite_genres
})

# 3. 상호작용 데이터 생성: 각 유저가 읽은 책을 랜덤하게 3~7권 선택
interaction_list = []
for user_id in user_ids:
    num_interactions = np.random.randint(3, 8)
    interacted_books = np.random.choice(book_ids, size=num_interactions, replace=False)
    for book_id in interacted_books:
        interaction_list.append({"user_id": user_id, "book_id": book_id})

interactions_df = pd.DataFrame(interaction_list)

# CSV 파일로 저장
books_df.to_csv(os.path.join(data_dir, "books.csv"), index=False)
users_df.to_csv(os.path.join(data_dir, "users.csv"), index=False)
interactions_df.to_csv(os.path.join(data_dir, "interactions.csv"), index=False)

print("샘플 데이터 생성 완료!")
print("books.csv:")
print(books_df.head(), "\n")
print("users.csv:")
print(users_df.head(), "\n")
print("interactions.csv:")
print(interactions_df.head())
