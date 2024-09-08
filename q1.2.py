import numpy as np
import pandas as pd
from collections import Counter
import time
from knn import KNN
import matplotlib.pyplot as plt

def calculate_accuracy(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("The lengths of y_true and y_pred must be the same")
    return np.mean(y_true == y_pred)

def calculate_precision_recall_f1(y_true, y_pred):
    labels = np.unique(y_true)
    label_to_index = {label: i for i, label in enumerate(labels)}
    
    y_true_encoded = np.array([label_to_index[label] for label in y_true])
    y_pred_encoded = np.array([label_to_index.get(label, -1) for label in y_pred])
    
    unknown_count = np.sum(y_pred_encoded == -1)
    
    mask = y_pred_encoded != -1
    y_true_encoded = y_true_encoded[mask]
    y_pred_encoded = y_pred_encoded[mask]
    
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for i in range(len(labels)):
        cm[i] = np.bincount(y_pred_encoded[y_true_encoded == i], minlength=len(labels))
    
    tp = np.diag(cm)
    
    fp = np.sum(cm, axis=0) - tp
    
    fn = np.sum(cm, axis=1) - tp
    
    precision = np.zeros_like(tp, dtype=float)
    mask = (tp + fp) != 0
    precision[mask] = tp[mask] / (tp[mask] + fp[mask])
    
    recall = np.zeros_like(tp, dtype=float)
    mask = (tp + fn) != 0
    recall[mask] = tp[mask] / (tp[mask] + fn[mask])
    
    f1 = np.zeros_like(tp, dtype=float)
    mask = (precision + recall) != 0
    f1[mask] = 2 * (precision[mask] * recall[mask]) / (precision[mask] + recall[mask])
    
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    return macro_precision, macro_recall, macro_f1

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

start = time.time()
s = time.time()

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('val.csv')

genre_column = train_data.columns[-1]

numeric_columns = train_data.columns[:-1]
train_data[numeric_columns] = train_data[numeric_columns].apply(pd.to_numeric, errors='coerce')
test_data[numeric_columns] = test_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

columns_to_drop = [0, 10, 12]
train_data_dropped = train_data.drop(train_data.columns[columns_to_drop], axis=1)
test_data_dropped = test_data.drop(test_data.columns[columns_to_drop], axis=1)

X_train = train_data_dropped.iloc[:, :-1].values
y_train = train_data_dropped[genre_column].values

X_test = test_data_dropped.iloc[:, :-1].values
y_test = test_data_dropped[genre_column].values

X_train_normalized = min_max_normalize(X_train)
X_test_normalized = min_max_normalize(X_test)

X_test_100_normalized = X_test_normalized[:100]
y_test_100 = y_test[:100]

e = time.time()
el = e-s
print(f"Elapsed time for preprocessing data: {el:.4f} seconds")

k = 5
distance_metric = 'cosine'

knn = KNN(k=k, distance_metric=distance_metric)
knn.fit(X_train_normalized, y_train)

y_pred_100 = knn.predict(X_test_100_normalized)

accuracy = calculate_accuracy(y_test_100, y_pred_100)
precision, recall, f1 = calculate_precision_recall_f1(y_test_100, y_pred_100)

print(f"Accuracy on validation = {accuracy:.4f}")
print(f"Precision on validation = {precision:.4f}")
print(f"Recall on validation = {recall:.4f}")
print(f"F1 score on validation = {f1:.4f}")

end = time.time()
elapsed = end - start 
print(f"Elapsed time: {elapsed:.4f} seconds")

def plot_k_vs_accuracy(X_train, y_train, X_test, y_test, k_range, distance_metric):
    accuracies = []
    for k in k_range:
        knn = KNN(k=k, distance_metric=distance_metric)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test_100_normalized)
        accuracy = calculate_accuracy(y_test_100, y_pred)
        accuracies.append(accuracy)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, accuracies, marker='o')
    plt.title(f'k vs Accuracy (Distance Metric: {distance_metric})')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

k_range = range(1, 21)
distance_metric = 'euclidean'

plot_k_vs_accuracy(X_train_normalized, y_train, X_test_100_normalized, y_test_100, k_range, distance_metric)