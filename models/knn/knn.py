<<<<<<< HEAD
import numpy as np
import pandas as pd
from collections import Counter

def calculate_accuracy(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("The lengths of y_true and y_pred must be the same")
    
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    accuracy = correct / total
    
    return accuracy

def convert_columns_to_numeric(df):
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def min_max_normalize(X):
    X = X.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)
    col_min = np.min(X, axis=0)
    col_max = np.max(X, axis=0)
    denominator = col_max - col_min
    denominator[denominator == 0] = 1 
    X_normalized = (X - col_min) / denominator
    return X_normalized

def manhattan_distance(X1, X2):
    return np.abs(X1[:, np.newaxis] - X2).sum(axis=2)

def euclidean_distance(X1, X2):
    return np.sqrt(np.sum(np.square(X1[:, np.newaxis] - X2), axis=2))

def cosine_distance(X1, X2):
    X1_normalized = X1 / np.linalg.norm(X1, axis=1, keepdims=True)
    X2_normalized = X2 / np.linalg.norm(X2, axis=1, keepdims=True)
    cosine_similarity = np.dot(X1_normalized, X2_normalized.T)
    return 1 - cosine_similarity

def most_common_genre(nearest_genres_row):
    genre_list = nearest_genres_row.tolist()
    most_common = Counter(genre_list).most_common(1)
    return most_common[0][0]


class KNN:

    def __init__(self, k, distance_metric='cosine'):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def predict(self, X):
        if self.distance_metric == 'cosine':
            distances = cosine_distance(X, self.X_train)
        elif self.distance_metric == 'manhattan':
            distances = manhattan_distance(X, self.X_train)
        elif self.distance_metric == 'euclidean':
            distances = euclidean_distance(X, self.X_train)
        else:
            raise ValueError("Unsupported distance metric")
        
        k_indices = np.argpartition(distances, self.k, axis=1)[:, :self.k]
        nearest_genres = self.Y_train[k_indices]
        
        predictions = np.apply_along_axis(most_common_genre, 1, nearest_genres)
        
        return predictions
=======
import numpy as np
import pandas as pd
from collections import Counter

def calculate_accuracy(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("The lengths of y_true and y_pred must be the same")
    
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    accuracy = correct / total
    
    return accuracy

def convert_columns_to_numeric(df):
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def min_max_normalize(X):
    X = X.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)
    col_min = np.min(X, axis=0)
    col_max = np.max(X, axis=0)
    denominator = col_max - col_min
    denominator[denominator == 0] = 1 
    X_normalized = (X - col_min) / denominator
    return X_normalized

def manhattan_distance(X1, X2):
    return np.abs(X1[:, np.newaxis] - X2).sum(axis=2)

def euclidean_distance(X1, X2):
    return np.sqrt(np.sum(np.square(X1[:, np.newaxis] - X2), axis=2))

def cosine_distance(X1, X2):
    X1_normalized = X1 / np.linalg.norm(X1, axis=1, keepdims=True)
    X2_normalized = X2 / np.linalg.norm(X2, axis=1, keepdims=True)
    cosine_similarity = np.dot(X1_normalized, X2_normalized.T)
    return 1 - cosine_similarity

def most_common_genre(nearest_genres_row):
    genre_list = nearest_genres_row.tolist()
    most_common = Counter(genre_list).most_common(1)
    return most_common[0][0]


class KNN:

    def __init__(self, k, distance_metric='cosine'):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def predict(self, X):
        if self.distance_metric == 'cosine':
            distances = cosine_distance(X, self.X_train)
        elif self.distance_metric == 'manhattan':
            distances = manhattan_distance(X, self.X_train)
        elif self.distance_metric == 'euclidean':
            distances = euclidean_distance(X, self.X_train)
        else:
            raise ValueError("Unsupported distance metric")
        
        k_indices = np.argpartition(distances, self.k, axis=1)[:, :self.k]
        nearest_genres = self.Y_train[k_indices]
        
        predictions = np.apply_along_axis(most_common_genre, 1, nearest_genres)
        
        return predictions
>>>>>>> fe38184abf01ec865519a15b9085a37923b231b0
