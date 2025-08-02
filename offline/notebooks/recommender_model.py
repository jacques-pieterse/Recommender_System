import numpy as np
import time
from numba import njit, prange

# --- OPTIMIZED Numba-Accelerated Updates (Parallel) ---
@njit(parallel=True, cache=True)
def update_user_embeddings(num_users, user_ptrs, all_m, all_r,
                           user_embeddings, item_embeddings,
                           user_biases, item_biases,
                           lam, gamma, tau_identity):
    """
    Update user embeddings in parallel using the pre-calculated item embeddings.
    """
    dim = user_embeddings.shape[1]
    # This loop now runs in parallel across all available CPU cores
    for u in prange(num_users):
        start = user_ptrs[u]
        end = user_ptrs[u+1]
        n = end - start
        if n == 0:
            continue
            
        # Use fancy indexing to grab all necessary data at once
        movies = all_m[start:end]
        ratings = all_r[start:end]
        item_vecs = item_embeddings[movies]
        item_bias = item_biases[movies]
    
        # Perform vectorized prediction
        preds = item_vecs @ user_embeddings[u] + item_bias
        
        residuals = ratings - preds
        user_bias = lam * np.sum(residuals) / (lam * n + gamma)
        user_biases[u] = user_bias
    
        # Solve for the new embedding
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
    Update item embeddings in parallel using the pre-calculated user embeddings.
    """
    dim = item_embeddings.shape[1]
    # This loop also runs in parallel
    for m in prange(num_items):
        start = item_ptrs[m]
        end = item_ptrs[m+1]
        n = end - start
        if n == 0:
            continue
            
        # Use fancy indexing
        users = all_u[start:end]
        ratings = all_r[start:end]
        user_vecs = user_embeddings[users]
        user_bias = user_biases[users]
        
        # Vectorized prediction
        preds = user_vecs @ item_embeddings[m] + user_bias
        
        residuals = ratings - preds
        item_bias = lam * np.sum(residuals) / (lam * n + gamma)
        item_biases[m] = item_bias
        
        # Solve for the new embedding
        adjusted = ratings - user_bias - item_bias
        A = lam * (user_vecs.T @ user_vecs) + tau_identity
        b = lam * (user_vecs.T @ adjusted)
        item_embeddings[m] = np.linalg.solve(A, b)

# --- OPTIMIZED JIT-compiled evaluation function (Parallel) ---
@njit(parallel=True, cache=True)
def calculate_rmse_jit(ptrs, all_user_indices, all_item_indices, all_ratings, 
                       user_embeddings, item_embeddings, user_biases, item_biases):
    """
    Calculates the Root Mean Squared Error in parallel.
    This function is generic and processes the data from a user-centric view.
    """
    total_squared_error = 0.0
    
    # Loop over all users in parallel
    for u in prange(len(ptrs) - 1):
        start, end = ptrs[u], ptrs[u+1]
        if start == end:
            continue
            
        item_idxs = all_item_indices[start:end]
        actuals = all_ratings[start:end]
        
        # Vectorized prediction for all items rated by this user
        preds = (user_embeddings[u] @ item_embeddings[item_idxs].T) + user_biases[u] + item_biases[item_idxs]
        
        # Numba automatically handles thread-safe summation for reduction operations
        total_squared_error += np.sum((actuals - preds) ** 2)

    return np.sqrt(total_squared_error / len(all_ratings))

# --- OPTIMIZED Main Training Function ---
def training_als_from_preprocessed(
    mu, 
    num_users, num_items,
    all_u_train, all_m_train, all_r_train_u, user_ptrs_train,
    all_m_rev_train, all_u_rev_train, all_r_train_m, item_ptrs_train,
    all_u_test, all_m_test, all_r_test, user_ptrs_test,
    num_epochs=20, lam=0.1, gamma=0.1, tau=0.1,
    embeddings_dim=20, scale=0.1
):
    """
    Optimized ALS training using preprocessed data.
    """
    print("Starting ALS optimization (preprocessed)...")
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

        train_rmse = calculate_rmse_jit(user_ptrs_train, all_u_train, all_m_train, all_r_train_u,
                                        user_embeddings, item_embeddings,
                                        user_biases, item_biases)

        test_rmse = calculate_rmse_jit(user_ptrs_test, all_u_test, all_m_test, all_r_test,
                                       user_embeddings, item_embeddings,
                                       user_biases, item_biases)

        squared_user_bias = np.sum(user_biases**2)
        squared_item_bias = np.sum(item_biases**2)

        # Frobenius norms of embedding matrices
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