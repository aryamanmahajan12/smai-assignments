import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))



"""----------------------------------------------DATASET ANALYSIS AND PREPROCESSING--------------------------------------------------------"""

wine = pd.read_csv(r"C:\Users\aryma\Desktop\SMAI\smai\smai-m24-assignments-aryamanmahajan123\data\external\WineQT.csv")

x,y = wine.shape

print(f"Number of records  : {x}")
print(f"Number of features : {y}")

for i in range(1, y):  
    print(f"Attribute  =  {wine.columns[i]}")
    print(f"Mean    : {np.mean(wine.iloc[:, i])}")  
    print(f"Max     : {np.max(wine.iloc[:, i])}")   
    print(f"Min     : {np.min(wine.iloc[:,i])}") 
    print(f"Std Dev : {np.std(wine.iloc[:,i])}")

label_counts = wine['quality'].value_counts()
output_dir = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(output_dir, exist_ok=True)

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

imputer = SimpleImputer(strategy='mean')
wine_imputed = pd.DataFrame(imputer.fit_transform(wine), columns=wine.columns)

min_max_scaler = MinMaxScaler()
normalized_data = min_max_scaler.fit_transform(wine_imputed)

normalized_df = pd.DataFrame(normalized_data, columns=wine_imputed.columns)

standard_scaler = StandardScaler()
standardized_data = standard_scaler.fit_transform(wine_imputed)

standardized_df = pd.DataFrame(standardized_data, columns=wine_imputed.columns)

print("Normalized Data:")
print(normalized_df.head())

print("\nStandardized Data:")
print(standardized_df.head())