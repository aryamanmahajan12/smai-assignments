import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')


plt.hist(data['energy'], bins=15, color='blue', edgecolor='black')
plt.title('Number of songs in each energy category')
plt.xlabel('Energy')
plt.ylabel('Frequency')
plt.show()


data['track_genre'].value_counts().plot(kind='bar', color='green')
plt.title('Counts of Categorical Feature')
plt.xlabel('Categories')
plt.ylabel('Counts')
plt.show()


x_feature = data['energy']
y_feature = data['speechiness']

third_feature = data['danceability']

plt.scatter(x_feature, y_feature, c=third_feature, cmap='viridis', marker='o', s=100)
plt.colorbar(label='explicit')
plt.title('Scatter Plot with Third Variable danceability as Color')
plt.xlabel('energy')
plt.ylabel('speechiness')
plt.show()
