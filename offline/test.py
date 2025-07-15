import csv
import time

# User and movie mappings
user_to_idx = {}
idx_to_user = {}
movie_to_idx = {}
idx_to_movie = {}

# Sparse Lists for user and movie ratings
user_ratings = []
movie_ratings = []

start_time = time.time()
with open('ratings.csv', 'r', newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile)
    header = next(csv_reader)
    for row in csv_reader:
        user_id = row[0]
        movie_id = row[1]
        rating = float(row[2])

        if user_id not in user_to_idx:
            user_idx = len(user_to_idx)
            user_to_idx[user_id] = user_idx
            idx_to_user[user_idx] = user_id
            user_ratings.append([])

        if movie_id not in movie_to_idx:
            movie_idx = len(movie_to_idx)
            movie_to_idx[movie_id] = movie_idx
            idx_to_movie[movie_idx] = movie_id
            movie_ratings.append([]) 

        user_idx = user_to_idx[user_id]
        movie_idx = movie_to_idx[movie_id]
        user_ratings[user_idx].append((movie_idx, rating))

        movie_ratings[movie_idx].append((user_idx, rating))

end_time = time.time()
print("Finished")
print("Time taken:", end_time - start_time, "seconds")
