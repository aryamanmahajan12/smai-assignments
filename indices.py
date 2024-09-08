import pandas as pd

data = pd.read_csv('val.csv')

columns = data.columns

for index, column_name in enumerate(columns):
    print(f"Index: {index}, Column Name: {column_name}")