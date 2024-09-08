import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, n_components=5, max_iters=100, tol=1e-4, reg_covar=1e-6):
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol
        self.reg_covar = reg_covar
        self.weights = None
        self.means = None
        self.covariances = None

    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.n_samples, self.n_features = X.shape

        # Initialize parameters
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = X[np.random.choice(self.n_samples, self.n_components, replace=False)]
        self.covariances = np.array([np.eye(self.n_features) for _ in range(self.n_components)])

        log_likelihood = -np.inf
        for _ in range(self.max_iters):
            # E-step
            responsibilities = self._e_step(X)

            # M-step
            self._m_step(X, responsibilities)

            # Compute log-likelihood
            new_log_likelihood = self._compute_log_likelihood(X)

            # Check for convergence
            if np.abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        return self

    def _e_step(self, X):
        return self._estimate_responsibilities(X)

    def _m_step(self, X, responsibilities):
        total_responsibility = responsibilities.sum(axis=0)
        self.weights = total_responsibility / self.n_samples
        self.means = np.dot(responsibilities.T, X) / total_responsibility[:, np.newaxis]
        
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covariances[k] = np.dot(responsibilities[:, k] * diff.T, diff) / total_responsibility[k]
            # Add regularization to ensure positive definiteness
            self.covariances[k] += np.eye(self.n_features) * self.reg_covar

    def _estimate_responsibilities(self, X):
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return np.exp(log_resp)

    def _estimate_weighted_log_prob(self, X):
        return self._estimate_log_prob(X) + np.log(self.weights)

    def _estimate_log_prob(self, X):
        return np.array([multivariate_normal.logpdf(X, mean=mean, cov=cov) 
                         for mean, cov in zip(self.means, self.covariances)]).T

    def _compute_log_likelihood(self, X):
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        return logsumexp(weighted_log_prob, axis=1).sum()

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        responsibilities = self._estimate_responsibilities(X)
        return np.argmax(responsibilities, axis=1)

    def score(self, X):
        return self._compute_log_likelihood(X)

def logsumexp(a, axis=None, keepdims=False):
    a_max = np.max(a, axis=axis, keepdims=True)
    tmp = np.exp(a - a_max)
    s = np.sum(tmp, axis=axis, keepdims=keepdims)
    out = np.log(s)
    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    out += a_max
    return out