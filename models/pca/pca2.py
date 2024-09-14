import numpy as np

class PCA:
    def __init__(self, n_comps):
        self.n_comps = n_comps
        self.components = None
        self.mean = None
        self.explained_variance_ratio = None

    def fit(self, X):
        
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        cov = np.cov(X, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        idx = np.argpartition(eigenvalues, -self.n_comps)[-self.n_comps:]
        
        idx = idx[np.argsort(eigenvalues[idx])[::-1]]
        
        self.components = eigenvectors[:, idx]

        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio = eigenvalues[idx] / total_var

        return self

    def transform(self, X):
        X = X - self.mean

        return np.dot(X, self.components)

    def checkPCA(self, X):
        self.fit(X)

        X_transformed = self.transform(X)

        if X_transformed.shape[1] != self.n_comps:
            return False

        original_var = np.var(X, axis=0).sum()
        transformed_var = np.var(X_transformed, axis=0).sum()

        if not np.isclose(original_var, transformed_var, rtol=0.05):
            return False

        return True