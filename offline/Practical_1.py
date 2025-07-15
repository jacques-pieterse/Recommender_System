import csv
import numpy as np
import time

# User and movie mappings
user_to_idx = {}
idx_to_user = {}
movie_to_idx = {}
idx_to_movie = {}
user_rating_count = {}
movie_rating_count = {}

start_time = time.time()
with open('ratings.csv', 'r', newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile)
    header = next(csv_reader)
    for row in csv_reader:
        user_id = row[0]
        movie_id = row[1]
        if(user_rating_count.get(user_id) is None):
            user_to_idx[user_id] = len(user_to_idx)
            idx_to_user[len(user_to_idx) - 1] = user_id
            user_rating_count[user_id] = 0
        user_rating_count[user_id] += 1

        if(movie_rating_count.get(movie_id) is None):
            movie_to_idx[movie_id] = len(movie_to_idx)
            idx_to_movie[len(movie_to_idx) - 1] = movie_id
            movie_rating_count[movie_id] = 0
        movie_rating_count[movie_id] += 1

user_ratings = [[[-1, 0]]*user_rating_count[user_id] for user_id in user_rating_count]
movie_ratings = [[[-1, 0]]*movie_rating_count[movie_id] for movie_id in movie_rating_count]

with open('ratings.csv', 'r', newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile)
    header = next(csv_reader)
    for row in csv_reader:
        user_id = row[0]
        movie_id = row[1]
        rating = float(row[2])

        count = 0
        for movie, _ in user_ratings[user_to_idx[user_id]]:
            if movie == -1:
                user_ratings[user_to_idx[user_id]][count] = [movie_to_idx[movie_id], rating]
                break
            count += 1
        
        count = 0
        for user, _ in movie_ratings[movie_to_idx[movie_id]]:
            if user == -1:
                movie_ratings[movie_to_idx[movie_id]][count] = [user_to_idx[user_id], rating]
                break
            count += 1
end_time = time.time()
print("Finished")
print("Time taken:", end_time - start_time, "seconds")
