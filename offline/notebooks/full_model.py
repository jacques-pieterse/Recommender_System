import numpy as np
from numba import njit, prange
import time

class RecommenderSystem:
    def __init__(self, user_to_idx, item_to_idx, ratings_mean):
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.num_users = len(user_to_idx)
        self.num_items = len(item_to_idx)
        self.user_embeddings = None
        self.item_embeddings = None
        self.user_biases = None
        self.item_biases = None
        self.ratings_mean = ratings_mean
        self.scale = 0.1  # Example scale for random initialization
        self.embeddings_dim = 10  # Example dimension for embeddings
        self.lam = 0.1  # Regularization parameter
        self.gamma = 0.1  # Bias regularization parameter
        self.tau = 0.1  # Regularization for embeddings

    def training(self, num_epochs, 
                user_ptrs_train, item_ptrs_train, user_ptrs_test,
                all_u_train, all_m_train, all_r_train_u,
                all_u_test, all_m_test, all_r_test, 
                all_u_rev_train, all_r_train_m):
        print("Starting ALS optimization (preprocessed)...")
        start_time = time.time()

        self.user_embeddings = np.random.normal(0, self.scale, (self.num_users, self.embeddings_dim)).astype(np.float64)
        self.item_embeddings = np.random.normal(0, self.scale, (self.num_items, self.embeddings_dim)).astype(np.float64)
        self.user_biases = np.zeros(self.num_users, dtype=np.float64)
        self.item_biases = np.zeros(self.num_items, dtype=np.float64)
        tau_identity = np.ascontiguousarray(self.tau * np.eye(self.embeddings_dim, dtype=np.float64))

        training_RMSE, testing_RMSE = [], []
        training_loss = []

        for epoch in range(num_epochs):
            epoch_start = time.time()

            update_user(user_ptrs_train, all_m_train, all_r_train_u,
                        tau_identity)

            update_item(item_ptrs_train, all_u_rev_train, all_r_train_m,
                        tau_identity)

            train_rmse = calculate_rmse(user_ptrs_train, all_u_train, all_m_train, all_r_train_u)

            test_rmse = calculate_rmse(user_ptrs_test, all_u_test, all_m_test, all_r_test)

            squared_user_bias = np.sum(self.user_biases**2)
            squared_item_bias = np.sum(self.item_biases**2)

            # Frobenius norms of embedding matrices
            embedding_user_norm = np.sum(self.user_embeddings * self.user_embeddings)
            embedding_item_norm = np.sum(self.item_embeddings * self.item_embeddings)

            train_loss = (
                self.lam * train_rmse**2 * len(all_r_train_u)
                + self.gamma * (squared_user_bias + squared_item_bias)
                + self.tau * (embedding_user_norm + embedding_item_norm)
            )

            training_RMSE.append(train_rmse)
            testing_RMSE.append(test_rmse)
            training_loss.append(train_loss)

            print(f"Epoch {epoch+1}/{num_epochs} | Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f} | Time: {time.time() - epoch_start:.2f}s")

        print(f"Finished training. Total time: {time.time() - start_time:.2f}s")

        return training_loss, training_RMSE, testing_RMSE

    def train_dummy_user(self, num_epochs, dummy_user_embedding, dummy_user_bias, rating):
        movie_rated = rating[0]
        rating_value = rating[1]
        movie_embedding = self.item_embeddings[movie_rated]
        movie_bias = self.item_biases[movie_rated]
        tau_identity = self.tau * np.eye(self.embeddings_dim)
        for epoch in range(num_epochs):
            actual_rating = rating_value
            pred = (dummy_user_embedding @ self.item_embeddings) + dummy_user_bias + movie_bias
            residual = actual_rating - pred

            dummy_user_bias = self.lam * residual / ((self.lam * 1) + self.gamma)

            adjusted_residual = actual_rating - dummy_user_bias - movie_bias
            user_inverse_term = self.lam * np.outer(movie_embedding, movie_embedding) + tau_identity
            user_term = self.lam * (movie_embedding * adjusted_residual)

            dummy_user_embedding = np.linalg.solve(user_inverse_term, user_term)

            pred = (dummy_user_embedding @ movie_embedding) + dummy_user_bias + movie_bias
            train_error = actual_rating - pred
            train_loss = (self.lam * train_error ** 2) \
                + (self.gamma * (np.sum(dummy_user_bias ** 2) + np.sum(self.item_biases[movie_rated] ** 2))) \
                + (self.tau * (np.sum(dummy_user_embedding ** 2) + np.sum(movie_embedding ** 2)))


            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")
        return dummy_user_embedding, dummy_user_bias
    
    def get_hyperparameters(self):
        """
        Returns the hyperparameters used in the model.
        """
        return {
            'embeddings_dim': self.embeddings_dim,
            'lam': self.lam,
            'gamma': self.gamma,
            'tau': self.tau
        }

    def get_model(self):
        """
        Returns the trained model parameters.
        """
        return {
            'user_embeddings': self.user_embeddings,
            'item_embeddings': self.item_embeddings,
            'user_biases': self.user_biases,
            'item_biases': self.item_biases
        }

    def get_polarizing(self, top_n=10, descending=True):
        movie_norms = np.linalg.norm(self.item_embeddings, axis=1)
        if descending:
            top_indices = np.argsort(-movie_norms)[:top_n]
        else:
            top_indices = np.argsort(movie_norms)[:top_n]
        top_values = movie_norms[top_indices]
        top_pairs = list(zip(top_indices, top_values))

        return top_pairs


@njit(parallel=True, cache=True)
def update_user(user_ptrs, all_m, all_r, user_embeddings, item_embeddings,
                user_biases, item_biases, lam, gamma, tau_identity):
    """
    Update user embeddings in parallel using the pre-calculated item embeddings.
    """
    # This loop now runs in parallel across all available CPU cores
    for u in prange(len(user_ptrs) - 1):
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
def update_item(item_ptrs, all_u, all_r, item_embeddings, user_embeddings,
                user_biases, item_biases, lam, gamma, tau_identity):
    """
    Update item embeddings in parallel using the pre-calculated user embeddings.
    """
    # This loop also runs in parallel
    for m in prange(len(item_ptrs) - 1):
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

@njit(parallel=True, cache=True)
def calculate_rmse(ptrs, all_user_indices, all_item_indices, all_ratings):
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
        preds = (self.user_embeddings[u] @ self.item_embeddings[item_idxs].T) + self.user_biases[u] + self.item_biases[item_idxs]

        # Numba automatically handles thread-safe summation for reduction operations
        total_squared_error += np.sum((actuals - preds) ** 2)

    return np.sqrt(total_squared_error / len(all_ratings))