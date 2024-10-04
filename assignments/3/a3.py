import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from models.MLP.MLP import MLPClassifier

"""----------------------------------------------DATASET ANALYSIS AND PREPROCESSING--------------------------------------------------------"""

wine = pd.read_csv(r"C:\Users\aryma\Desktop\SMAI\smai\smai-m24-assignments-aryamanmahajan123\data\external\WineQT.csv")

# Get the number of records and features
num_records, num_features = wine.shape
print(f"Number of records  : {num_records}")
print(f"Number of features : {num_features}")

# Display statistics for each feature
for i in range(1, num_features): 
    print(f"\nAttribute  =  {wine.columns[i]}")
    print(f"Mean    : {wine.iloc[:, i].mean():.2f}")  
    print(f"Max     : {wine.iloc[:, i].max():.2f}")   
    print(f"Min     : {wine.iloc[:, i].min():.2f}") 
    print(f"Std Dev : {wine.iloc[:, i].std():.2f}")

# Count occurrences of each quality label
label_counts = wine['quality'].value_counts()
print("\nLabel Counts:")
print(label_counts)

# Create output directory for plots
output_dir = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(output_dir, exist_ok=True)

# Plot distribution of labels in dataset
plt.figure(figsize=(10, 6))
label_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Labels in Dataset')
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.savefig(os.path.join(output_dir, 'label_distribution.png'))
plt.tight_layout()
plt.show()

wine.dropna(inplace=True)

X = wine.drop('quality', axis=1) 
y = wine['quality'].values          

print(f"y len : {len(y.shape)}")


class_counts = pd.Series(y).value_counts()
print("\nClass Distribution:")
print(class_counts)

if (class_counts < 2).any():
    print("Warning: One or more classes have fewer than 2 instances.")

normalizer = MinMaxScaler()
X_normalized = normalizer.fit_transform(X)
standardizer = StandardScaler()
X_standardized = standardizer.fit_transform(X_normalized)

X_train, X_temp, y_train, y_temp = train_test_split(X_standardized, y, test_size=0.2, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"\nTraining set shape: {X_train.shape}, {y_train.shape}")
print(f"Validation set shape: {X_val.shape}, {y_val.shape}")
print(f"Test set shape: {X_test.shape}, {y_test.shape}")

print("\nTraining Labels Distribution:")
print(pd.Series(y_train).value_counts(normalize=True))

print("\nValidation Labels Distribution:")
print(pd.Series(y_val).value_counts(normalize=True))

print("\nTest Labels Distribution:")
print(pd.Series(y_test).value_counts(normalize=True))



"""-----------------------------------------------------Testing The Model----------------------------------------------------------------"""



mlp = MLPClassifier(hidden_layers=[64, 64], learning_rate=0.1, activation='relu', 
                 optimizer='sgd', batch_size=32, epochs=1000)

mlp.fit(X_train,y_train,X_val,y_val)
y_val_pred = mlp.predict(X_val)
print(y_val_pred)
print(y_val)
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_val, y_val_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.tight_layout()
plt.show()