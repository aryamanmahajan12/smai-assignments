import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('reg_train.csv')

x = data['x']
y = data['y']

plt.figure(figsize=(10, 6))
plt.scatter(x, y, marker='o', color='b', label='x vs y')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Plot of X vs Y')
plt.legend()
plt.grid(True)

plt.show()