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
