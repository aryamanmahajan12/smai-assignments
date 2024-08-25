import numpy  as np
from collections import Counter


def euclidean_distance(x1, x2, feature_indices=None):
    if feature_indices is not None:
        x1 = x1[feature_indices]
        x2 = x2[feature_indices]
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1,x2,feature_indices=None):
    if feature_indices is not None:
        x1 = x1[feature_indices]
        x2 = x2[feature_indices]
    distance = np.sum(np.abs(x1 - x2))
    return distance

def cosine_distance(x1, x2, feature_indices=None):
    if feature_indices is not None:
        x1 = x1[feature_indices]
        x2 = x2[feature_indices]
    
    dot_product = np.dot(x1, x2)
    magnitude_x1 = np.linalg.norm(x1)
    magnitude_x2 = np.linalg.norm(x2)
    cosine_similarity = dot_product / (magnitude_x1 * magnitude_x2)
    distance = 1 - cosine_similarity    
    return distance

def manhattan_distance(x1, x2,feature_indices=None):
    return np.sum(np.abs(x1[6:20] - x2[6:20]))

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1[6:20] - x2[6:20]) ** 2))

def cosine_distance(x1, x2):
    dot_product = np.dot(x1[6:20], x2[6:20])
    magnitude_x1 = np.linalg.norm(x1[6:20])
    magnitude_x2 = np.linalg.norm(x2[6:20])
    cosine_similarity = dot_product / (magnitude_x1 * magnitude_x2)
    distance = 1 - cosine_similarity    
    return distance


class KNN:

    def __init__(self, k=3, feature_indices=None):
        self.k = k
        self.feature_indices = feature_indices

    def fit(self, X):
        self.X_train = X

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):

        #self.feature_indices = list(range(5, 20))
        
        distances = [cosine_distance(x, x_train) for x_train in self.X_train]   

        k_indices = np.argsort(distances)[:self.k]
        nearest_genres = [self.X_train[i, 20] for i in k_indices]  
        most_common = Counter(nearest_genres).most_common(1)
        return most_common[0][0]
