import pandas as pd
import numpy as np

data = pd.read_csv('spotify.csv')

data = data.sample(frac=1, random_state=42).reset_index(drop=True)

train_size = int(0.8 * len(data))
val_size = int(0.1 * len(data))

train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

print(f"Training Dataset size: {len(train_data)}")
print(f"Validation Dataset size: {len(val_data)}")
print(f"Test Dataset size: {len(test_data)}")


print(train_data.head())
print(val_data.head())
print(test_data.head())


train_data.to_csv('train.csv', index=False)
val_data.to_csv('val.csv', index=False)
test_data.to_csv('test.csv', index=False)
