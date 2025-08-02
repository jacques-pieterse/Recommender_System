import csv
import time
import os
import random
import pickle
import numpy as np

TRAIN_RATIO = 0.8
EXTERNAL_DATA_PATH = "../data/external/"
EXTERNAL_SAVE_DATA_PATH = "../data/processed/"
FILE_NAME = "ratings.csv"
DATA_FOLDER = os.path.join(EXTERNAL_DATA_PATH, "ml-latest-small")
FILE_PATH = os.path.join(DATA_FOLDER, FILE_NAME)
SAVE_PATH = os.path.join(EXTERNAL_SAVE_DATA_PATH, "processed_data.npz")

def load_ratings_data(file_path):
    """Loads raw ratings data from CSV."""
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            header = next(csv_reader)
            return [row for row in csv_reader]
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def split_data(raw_data, train_ratio=TRAIN_RATIO, min_ratings_per_movie=50):
    """Splits ratings into train/test sets and builds mapping structures."""
    user_to_idx, idx_to_user = {}, {}
    movie_to_idx, idx_to_movie = {}, {}

    user_ratings_train, user_ratings_test = [], []
    movie_ratings_train, movie_ratings_test = [], []

    train_count = test_count = 0

    movie_rating_counts = {}
    for row in raw_data:
        movie_id = row[1]
        if movie_id not in movie_rating_counts:
            movie_rating_counts[movie_id] = 0
        movie_rating_counts[movie_id] += 1
    
    valid_movies = {movie_id for movie_id, count in movie_rating_counts.items() if count >= min_ratings_per_movie}

    for row in raw_data:
        user_id, movie_id, rating = row[0], row[1], float(row[2])
        
        if movie_id not in valid_movies:
            continue

        if user_id not in user_to_idx:
            user_idx = len(user_to_idx)
            user_to_idx[user_id] = user_idx
            idx_to_user[user_idx] = user_id
            user_ratings_train.append([])
            user_ratings_test.append([])

        if movie_id not in movie_to_idx:
            movie_idx = len(movie_to_idx)
            movie_to_idx[movie_id] = movie_idx
            idx_to_movie[movie_idx] = movie_id
            movie_ratings_train.append([])
            movie_ratings_test.append([])

        user_idx = user_to_idx[user_id]
        movie_idx = movie_to_idx[movie_id]

        # Only allow test samples if both user and movie were seen in training
        user_seen_before = len(user_ratings_train[user_idx]) > 0
        movie_seen_before = len(movie_ratings_train[movie_idx]) > 0

        if not user_seen_before or not movie_seen_before or random.random() < train_ratio:
            user_ratings_train[user_idx].append((movie_idx, rating))
            movie_ratings_train[movie_idx].append((user_idx, rating))
            train_count += 1
        else:
            user_ratings_test[user_idx].append((movie_idx, rating))
            movie_ratings_test[movie_idx].append((user_idx, rating))
            test_count += 1

    print(f"Train Count: {train_count}")
    print(f"Test Count: {test_count}")

    return (user_ratings_train, user_ratings_test,
            movie_ratings_train, movie_ratings_test,
            user_to_idx, movie_to_idx, idx_to_user, idx_to_movie)

def preprocess_ratings(data):
    """Converts ratings list to flattened COO-like arrays."""
    all_x, all_y, all_r = [], [], []
    ptrs = [0]
    for i, ratings in enumerate(data):
        for j, r in ratings:
            all_x.append(i)
            all_y.append(j)
            all_r.append(r)
        ptrs.append(len(all_r))
    return (
        np.array(all_x, dtype=np.int32),
        np.array(all_y, dtype=np.int32),
        np.array(all_r, dtype=np.float64),
        np.array(ptrs, dtype=np.int32)
    )

def save_processed_data(path, **arrays):
    """Saves all processed arrays to a .npz file."""
    np.savez_compressed(path, **arrays)
    print(f"Processed data saved to: {path}")

def process_and_save_dataset(dataset_name):
    """Helper to process and save a specific dataset by folder name."""
    folder = os.path.join(EXTERNAL_DATA_PATH, dataset_name)
    print("Processing:", folder)
    file_path = os.path.join(folder, "ratings.csv")

    dataset_name = dataset_name.rstrip('/')
    save_name = f"processed_{dataset_name.split('-')[-1]}.npz"
    save_base = os.path.join(EXTERNAL_SAVE_DATA_PATH, save_name)

    # Load ratings
    raw_data = load_ratings_data(file_path)

    (user_ratings_train, user_ratings_test,
     movie_ratings_train, movie_ratings_test,
     user_to_idx, movie_to_idx, idx_to_user, idx_to_movie) = split_data(raw_data)

    # Preprocess
    user_train_x, user_train_y, user_train_r, user_train_ptr = preprocess_ratings(user_ratings_train)
    user_test_x, user_test_y, user_test_r, user_test_ptr = preprocess_ratings(user_ratings_test)
    movie_train_x, movie_train_y, movie_train_r, movie_train_ptr = preprocess_ratings(movie_ratings_train)
    movie_test_x, movie_test_y, movie_test_r, movie_test_ptr = preprocess_ratings(movie_ratings_test)

    # Save arrays
    save_processed_data(
        save_base,
        user_train_x=user_train_x,
        user_train_y=user_train_y,
        user_train_r=user_train_r,
        user_train_ptr=user_train_ptr,
        user_test_x=user_test_x,
        user_test_y=user_test_y,
        user_test_r=user_test_r,
        user_test_ptr=user_test_ptr,
        movie_train_x=movie_train_x,
        movie_train_y=movie_train_y,
        movie_train_r=movie_train_r,
        movie_train_ptr=movie_train_ptr,
        movie_test_x=movie_test_x,
        movie_test_y=movie_test_y,
        movie_test_r=movie_test_r,
        movie_test_ptr=movie_test_ptr
    )

    # Save mappings
    mappings = {
        'user_to_idx': user_to_idx,
        'movie_to_idx': movie_to_idx,
        'idx_to_user': idx_to_user,
        'idx_to_movie': idx_to_movie
    }
    with open(save_base.replace(".npz", "_mappings.pkl"), 'wb') as f:
        pickle.dump(mappings, f)

    print(f"{dataset_name} processing complete.\n")

def load_processed_data(path=SAVE_PATH):
    """Loads both the preprocessed arrays and mapping dictionaries."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data file not found at: {path}")
    
    mapping_path = path.replace('.npz', '_mappings.pkl')
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Mapping file not found at: {mapping_path}")

    data_npz = np.load(path)
    with open(mapping_path, 'rb') as f:
        mappings = pickle.load(f)

    arrays = {
        'user_train_x': data_npz['user_train_x'],
        'user_train_y': data_npz['user_train_y'],
        'user_train_r': data_npz['user_train_r'],
        'user_train_ptr': data_npz['user_train_ptr'],
        'user_test_x': data_npz['user_test_x'],
        'user_test_y': data_npz['user_test_y'],
        'user_test_r': data_npz['user_test_r'],
        'user_test_ptr': data_npz['user_test_ptr'],
        'movie_train_x': data_npz['movie_train_x'],
        'movie_train_y': data_npz['movie_train_y'],
        'movie_train_r': data_npz['movie_train_r'],
        'movie_train_ptr': data_npz['movie_train_ptr'],
        'movie_test_x': data_npz['movie_test_x'],
        'movie_test_y': data_npz['movie_test_y'],
        'movie_test_r': data_npz['movie_test_r'],
        'movie_test_ptr': data_npz['movie_test_ptr'],
    }

    return arrays, mappings

def process_and_save_both_datasets():
    """Processes and saves both the small and large datasets."""
    start_time = time.time()
    print("Processing small dataset...")
    process_and_save_dataset("ml-latest-small/")

    print("Processing large dataset...")
    process_and_save_dataset("ml-32m/")

    print("All datasets processed and saved.")
    print("Total time:", round(time.time() - start_time, 2), "seconds")

def filter_by_min_ratings(x, y, r, min_ratings, axis='y'):
    """
    Filters flattened data by minimum number of ratings per item.

    Parameters:
        x (np.ndarray): Array of user or movie indices (e.g., users if axis='y').
        y (np.ndarray): Array of movie or user indices (e.g., movies if axis='y').
        r (np.ndarray): Array of ratings.
        min_ratings (int): Minimum number of ratings required to keep an item.
        axis (str): 'y' to filter by movie, 'x' to filter by user.

    Returns:
        filtered_x, filtered_y, filtered_r, valid_ids (np.ndarray): Filtered arrays and valid item indices.
    """
    if axis not in ['x', 'y']:
        raise ValueError("axis must be either 'x' or 'y'")

    item_array = y if axis == 'y' else x
    item_counts = np.bincount(item_array)
    valid_ids = np.where(item_counts >= min_ratings)[0]
    valid_set = set(valid_ids)

    mask = np.isin(item_array, valid_ids)
    return x[mask], y[mask], r[mask], valid_ids

# if __name__ == "__main__":
#     # Example usage
#     process_and_save_both_datasets()
#     # To load processed data:
#     # data, mappings = load_processed_data()
#     # print(data['user_train_x'], mappings['user_to_idx'])