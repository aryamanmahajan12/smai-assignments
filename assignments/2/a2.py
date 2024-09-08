import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.k_means.k_means import KMeans
from models.gmm.gmm import GMM

x = pd.read_feather(r"C:\Users\aryma\Desktop\SMAI\smai\smai-m24-assignments-aryamanmahajan123\data\external\word-embeddings.feather")

print("Initial data shape:", x.shape)
print("Initial data types:")
print(x.dtypes)

embeddings = np.stack(x['vit'].values)

print("Embeddings shape:", embeddings.shape)


costs = []
k_values = range(1, 15)  

for i in k_values:
    kmeans = KMeans(k=i, maxiters=500, plot_iters=False, tol=1e-9)
    kmeans.fit(embeddings)
    cost = np.mean(kmeans.get_cost())
    costs.append(cost)

plt.plot(k_values, costs, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Cost (WCSS)')
plt.title('Cost vs Number of Clusters')
plt.show()


kmeans = KMeans(k=8, maxiters=500, plot_iters=False, tol=1e-9)
kmeans.fit(embeddings)

print("Final cost:", np.mean(kmeans.get_cost()))
x['Cluster'] = kmeans.labels

cluster_dict = defaultdict(list)

for word, cluster in zip(x['words'], x['Cluster']):
    cluster_dict[cluster].append(word)

for cluster, words in cluster_dict.items():
    print(f"\nCluster {cluster}: {len(words)} words")
    print(", ".join(words))

x.to_csv('clustered_word_embeddings.csv', index=True)
print("\nResults saved to 'clustered_word_embeddings.csv'")

with open('cluster_words.txt', 'w') as f:
    for cluster, words in cluster_dict.items():
        f.write(f"Cluster {cluster}: {len(words)} words\n")
        f.write(", ".join(words) + "\n\n")
print("Cluster words saved to 'cluster_words.txt'")


print("\nPerforming GMM Analysis:")
likelihoods = []
n_components_range = range(1, 15)

for n_components in n_components_range:
    gmm = GMM(n_components=n_components, max_iters=100, tol=1e-4, reg_covar=1e-6)
    gmm.fit(embeddings)
    likelihood = gmm.score(embeddings)
    likelihoods.append(likelihood)
    print(f"Components: {n_components}, Log-likelihood: {likelihood}")

# Plot log-likelihood vs number of components
plt.figure(figsize=(10, 6))
plt.plot(n_components_range, likelihoods, marker='o')
plt.xlabel('Number of components')
plt.ylabel('Log-likelihood')
plt.title('GMM: Log-likelihood vs Number of Components')
plt.show()

# Choose the best number of components (you might want to use a different criterion)
best_n_components = n_components_range[np.argmax(likelihoods)]
print(f"\nBest number of components for GMM: {best_n_components}")

# Fit the GMM with the best number of components
gmm = GMM(n_components=best_n_components, max_iters=100, tol=1e-4, reg_covar=1e-6)
gmm.fit(embeddings)

# Get cluster assignments
gmm_cluster_assignments = gmm.predict(embeddings)

# Add GMM cluster assignments to the dataframe
x['GMM_Cluster'] = gmm_cluster_assignments

# Create a dictionary of GMM clusters and their words
gmm_cluster_dict = defaultdict(list)
for word, cluster in zip(x['words'], x['GMM_Cluster']):
    gmm_cluster_dict[cluster].append(word)

# Print GMM cluster information
print("\nGMM Clustering Results:")
for cluster, words in gmm_cluster_dict.items():
    print(f"\nGMM Cluster {cluster}: {len(words)} words")
    print(", ".join(words[:10]))  # Print first 10 words in each cluster

# Save GMM results
x.to_csv('gmm_clustered_word_embeddings.csv', index=True)
print("\nGMM results saved to 'gmm_clustered_word_embeddings.csv'")

with open('gmm_cluster_words.txt', 'w') as f:
    for cluster, words in gmm_cluster_dict.items():
        f.write(f"GMM Cluster {cluster}: {len(words)} words\n")
        f.write(", ".join(words) + "\n\n")
print("GMM cluster words saved to 'gmm_cluster_words.txt'")