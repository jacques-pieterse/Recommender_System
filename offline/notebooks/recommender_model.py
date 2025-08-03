import numpy as np
import time
from numba import njit, prange

@njit(parallel=True, cache=True)
def update_user_embeddings(num_users, user_ptrs, all_m, all_r,
                           user_embeddings, item_embeddings,
                           user_biases, item_biases,
                           lam, gamma, tau_identity):
    """
    Update user embeddings and bias in parallel.

    Parameters:
    - num_users (int): Number of users.
    - user_ptrs (np.ndarray[int]): Pointer array which points to start and end of each users' ratings.
    - all_m (np.ndarray[int]): Array of movie indices rated by users.
    - all_r (np.ndarray[float]): Array of ratings given by users.
    - user_embeddings (np.ndarray[float]): 2D array (shape: (num_users, embedding_dim)) of current user embeddings.
    - item_embeddings (np.ndarray[float]): 2D array (shape: (num_items, embedding_dim)) of current item embeddings.
    - user_biases (np.ndarray[float]): 1D array (shape: (num_users)) of current user biases.
    - item_biases (np.ndarray[float]): 1D array (shape: (num_items)) of current item biases.
    - lam (float): Regularization parameter for reward difference.
    - gamma (float): Regularization parameter for biases.
    - tau_identity (np.ndarray[float]): Precomputed tau * identity matrix where tau is regularization parameter for embeddings.
    """
    # Loop through each user in parallel and update the user's embedding and bias.
    # Possible as user embeddings and item embeddings are independent of each other.
    for u in prange(num_users):
        start = user_ptrs[u]
        end = user_ptrs[u+1]
        n = end - start
        if n == 0:
            continue
            
        # Grab the movies and ratings for this user at once
        # This avoids repeated indexing and is more efficient
        movies = all_m[start:end]
        ratings = all_r[start:end]
        item_vecs = item_embeddings[movies]
        item_bias = item_biases[movies]
    
        # Perform vectorized prediction
        preds = item_vecs @ user_embeddings[u] + item_bias
        
        # Calculate residuals and update user bias
        residuals = ratings - preds
        user_bias = lam * np.sum(residuals) / (lam * n + gamma)
        user_biases[u] = user_bias
    
        # Update user embedding
        adjusted = ratings - user_bias - item_bias
        A = lam * (item_vecs.T @ item_vecs) + tau_identity
        b = lam * (item_vecs.T @ adjusted)
        user_embeddings[u] = np.linalg.solve(A, b)

@njit(parallel=True, cache=True)
def update_item_embeddings(num_items, item_ptrs, all_u, all_r,
                           user_embeddings, item_embeddings,
                           user_biases, item_biases,
                           lam, gamma, tau_identity):
    """
    Update item embeddings and bias in parallel.

    Parameters:
    - num_items (int): Number of items.
    - item_ptrs (np.ndarray[int]): Pointer array which points to start and end of each item's ratings.
    - all_u (np.ndarray[int]): Array of user indices who rated items.
    - all_r (np.ndarray[float]): Array of ratings given by users to items.
    - user_embeddings (np.ndarray[float]): 2D array (shape: (num_users, embedding_dim)) of current user embeddings.
    - item_embeddings (np.ndarray[float]): 2D array (shape: (num_items, embedding_dim)) of current item embeddings.
    - user_biases (np.ndarray[float]): 1D array (shape: (num_users)) of current user biases.
    - item_biases (np.ndarray[float]): 1D array (shape: (num_items)) of current item biases.
    - lam (float): Regularization parameter for reward difference.
    - gamma (float): Regularization parameter for biases.
    - tau_identity (np.ndarray[float]): Precomputed tau * identity matrix where tau is regularization parameter for embeddings.
    """
    # Loop through each movie in parallel and update the movie's embedding and bias.
    # Possible as user embeddings and item embeddings are independent of each other.
    for m in prange(num_items):
        start = item_ptrs[m]
        end = item_ptrs[m+1]
        n = end - start
        if n == 0:
            continue
            
        # Grab the users and ratings for this movie at once
        # This avoids repeated indexing and is more efficient
        users = all_u[start:end]
        ratings = all_r[start:end]
        user_vecs = user_embeddings[users]
        user_bias = user_biases[users]
        
        # Perform vectorized prediction
        preds = user_vecs @ item_embeddings[m] + user_bias
        
        # Calculate residuals and update item bias
        residuals = ratings - preds
        item_bias = lam * np.sum(residuals) / (lam * n + gamma)
        item_biases[m] = item_bias
        
        # Update item embedding
        adjusted = ratings - user_bias - item_bias
        A = lam * (user_vecs.T @ user_vecs) + tau_identity
        b = lam * (user_vecs.T @ adjusted)
        item_embeddings[m] = np.linalg.solve(A, b)

@njit(parallel=True, cache=True)
def calculate_rmse(ptrs, all_item_indices, all_ratings, 
                       user_embeddings, item_embeddings, user_biases, item_biases):
    """
    Calculates the Root Mean Squared Error in parallel.

    Parameters:
    - ptrs (np.ndarray[int]): Pointer array which points to start and end of each user's ratings.
    - all_item_indices (np.ndarray[int]): Array of item indices.
    - all_ratings (np.ndarray[float]): Array of ratings.
    - user_embeddings (np.ndarray[float]): 2D array (shape: (num_users, embedding_dim)) of user embeddings.
    - item_embeddings (np.ndarray[float]): 2D array (shape: (num_items, embedding_dim)) of item embeddings.
    - user_biases (np.ndarray[float]): 1D array (shape: (num_users)) of user biases.
    - item_biases (np.ndarray[float]): 1D array (shape: (num_items)) of item biases.

    Returns:
    - float: The RMSE value.
    """
    total_squared_error = 0.0
    
    # Loop over all users in parallel and calculate the RMSE
    for u in prange(len(ptrs) - 1):
        start, end = ptrs[u], ptrs[u+1]
        if start == end:
            continue
            
        item_idxs = all_item_indices[start:end]
        actuals = all_ratings[start:end]
        
        # Vectorized prediction for all items rated by this user
        preds = (user_embeddings[u] @ item_embeddings[item_idxs].T) + user_biases[u] + item_biases[item_idxs]
        total_squared_error += np.sum((actuals - preds) ** 2)

    return np.sqrt(total_squared_error / len(all_ratings))

def training(num_users, num_items,
            all_m_train, all_r_train_u, user_ptrs_train,
            all_u_rev_train, all_r_train_m, item_ptrs_train,
            all_m_test, all_r_test, user_ptrs_test,
            num_epochs=20, lam=0.1, gamma=0.1, tau=0.1,
            embeddings_dim=20, scale=0.1):
    """
    Optimised ALS training using preprocessed data.

    Parameters:
    - num_users (int): Number of users.
    - num_items (int): Number of items.
    - all_m_train (np.ndarray[int]): Array of movie indices rated by users in training
    - all_r_train_u (np.ndarray[float]): Array of ratings given by users in training.
    - user_ptrs_train (np.ndarray[int]): Pointer array for training user ratings.
    - all_u_rev_train (np.ndarray[int]): Array of user indices who rated items in training.
    - all_r_train_m (np.ndarray[float]): Array of ratings given by users to items in training.
    - item_ptrs_train (np.ndarray[int]): Pointer array for training item ratings.
    - all_m_test (np.ndarray[int]): Array of movie indices rated by users in testing.
    - all_r_test (np.ndarray[float]): Array of ratings given by users in testing.
    - user_ptrs_test (np.ndarray[int]): Pointer array for testing user ratings.
    - num_epochs (int): Number of epochs to train.
    - lam (float): Regularization parameter for reward difference.
    - gamma (float): Regularization parameter for biases.
    - tau (float): Regularization parameter for embeddings.
    - embeddings_dim (int): Dimensionality of the embeddings.
    - scale (float): Scale for initial random embeddings.

    Returns:
    - training_loss (list): List of training loss values for each epoch.
    - training_RMSE (list): List of training RMSE values for each epoch.
    - testing_RMSE (list): List of testing RMSE values for each epoch.
    - embeddings (list): List containing user and item embeddings.
    - biases (list): List containing user and item biases.
    """
    print("Starting training")
    start_time = time.time()

    user_embeddings = np.random.normal(0, scale, (num_users, embeddings_dim)).astype(np.float64)
    item_embeddings = np.random.normal(0, scale, (num_items, embeddings_dim)).astype(np.float64)
    user_biases = np.zeros(num_users, dtype=np.float64)
    item_biases = np.zeros(num_items, dtype=np.float64)
    tau_identity = np.ascontiguousarray(tau * np.eye(embeddings_dim, dtype=np.float64))

    training_RMSE, testing_RMSE = [], []
    training_loss = []

    for epoch in range(num_epochs):
        epoch_start = time.time()

        update_user_embeddings(num_users, user_ptrs_train, all_m_train, all_r_train_u,
                               user_embeddings, item_embeddings,
                               user_biases, item_biases,
                               lam, gamma, tau_identity)

        update_item_embeddings(num_items, item_ptrs_train, all_u_rev_train, all_r_train_m,
                               user_embeddings, item_embeddings,
                               user_biases, item_biases,
                               lam, gamma, tau_identity)

        train_rmse = calculate_rmse(user_ptrs_train, all_m_train, all_r_train_u,
                                        user_embeddings, item_embeddings,
                                        user_biases, item_biases)

        test_rmse = calculate_rmse(user_ptrs_test, all_m_test, all_r_test,
                                       user_embeddings, item_embeddings,
                                       user_biases, item_biases)

        squared_user_bias = np.sum(user_biases**2)
        squared_item_bias = np.sum(item_biases**2)

        embedding_user_norm = np.sum(user_embeddings * user_embeddings)
        embedding_item_norm = np.sum(item_embeddings * item_embeddings)

        train_loss = (
            lam * train_rmse**2 * len(all_r_train_u)
            + gamma * (squared_user_bias + squared_item_bias)
            + tau * (embedding_user_norm + embedding_item_norm)
        )

        training_RMSE.append(train_rmse)
        testing_RMSE.append(test_rmse)
        training_loss.append(train_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f} | Time: {time.time() - epoch_start:.2f}s")

    print(f"Finished training. Total time: {time.time() - start_time:.2f}s")

    return training_loss, training_RMSE, testing_RMSE, [user_embeddings, item_embeddings], [user_biases, item_biases]