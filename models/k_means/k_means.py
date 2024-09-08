import numpy as np

class KMeans():
    def __init__(self, k=3, maxiters=100, plot_iters=False, tol=1e-4):
        self.k = k
        self.maxiters = maxiters
        self.plot_iters = plot_iters
        self.tol = tol
        self.centroids = None
        self.labels = None

    def fit(self, x):
        # Input validation
        if x.size == 0:
            raise ValueError("Input data is empty. Cannot perform KMeans clustering.")
        
        self.x = x
        self.n_samples, self.n_features = self.x.shape

        if self.n_samples < self.k:
            raise ValueError(f"Number of samples ({self.n_samples}) must be greater than number of clusters ({self.k}).")

        # Initialization Step
        random_idxs = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = self.x[random_idxs]

        for _ in range(self.maxiters):
            self.labels = self.get_clusters(self.centroids, self.x)

            centroids_old = self.centroids.copy()
            self.centroids = self.get_centroids()

            if np.sum((centroids_old - self.centroids)**2) < self.tol:
                break

        return self

    def predict(self, x):
        return self.get_clusters(self.centroids, x)

    def get_cost(self):
        distances = np.linalg.norm(self.x - self.centroids[self.labels], axis=1)
        return distances

    def get_clusters(self, centroids, x):
        distances = np.linalg.norm(x[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def get_centroids(self):
        centroids = np.zeros((self.k, self.n_features))
        for k in range(self.k):
            centroids[k] = np.mean(self.x[self.labels == k], axis=0)
        return centroids