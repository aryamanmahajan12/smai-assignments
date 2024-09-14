import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import mplcursors
import sys
import os
from collections import defaultdict
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.k_means.k_means import KMeans
from models.gmm.gmm import GMM
from models.pca.pca2 import PCA

x = pd.read_feather(r"C:\Users\aryma\Desktop\SMAI\smai\smai-m24-assignments-aryamanmahajan123\data\external\word-embeddings.feather")

print("Initial data shape:", x.shape)
print("Initial data types:")
print(x.dtypes)

embeddings = np.stack(x['vit'].values)

# print("Embeddings shape:", embeddings.shape)


# costs = []
# k_values = range(1, 15)  

# for i in k_values:
#     kmeans = KMeans(k=i, maxiters=500, plot_iters=False, tol=1e-9)
#     kmeans.fit(embeddings)
#     cost = np.mean(kmeans.get_cost())
#     costs.append(cost)

# plt.plot(k_values, costs, marker='o')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Cost (WCSS)')
# plt.title('Cost vs Number of Clusters')
# plt.show()


# kmeans = KMeans(k=8, maxiters=500, plot_iters=False, tol=1e-9)
# kmeans.fit(embeddings)

# print("Final cost:", np.mean(kmeans.get_cost()))
# x['Cluster'] = kmeans.labels

# cluster_dict = defaultdict(list)

# for word, cluster in zip(x['words'], x['Cluster']):
#     cluster_dict[cluster].append(word)

# for cluster, words in cluster_dict.items():
#     print(f"\nCluster {cluster}: {len(words)} words")
#     print(", ".join(words))

# x.to_csv('clustered_word_embeddings.csv', index=True)
# print("\nResults saved to 'clustered_word_embeddings.csv'")

# with open('cluster_words.txt', 'w') as f:
#     for cluster, words in cluster_dict.items():
#         f.write(f"Cluster {cluster}: {len(words)} words\n")
#         f.write(", ".join(words) + "\n\n")
# print("Cluster words saved to 'cluster_words.txt'")


# print("\nPerforming GMM Analysis:")
# likelihoods = []
# n_components_range = range(1, 9)

# for n_components in n_components_range:
#     gmm = GMM(n_components=n_components, max_iters=100, tol=1e-4, reg_covar=1e-6)
#     gmm.fit(embeddings)
#     likelihood = gmm.score(embeddings)
#     likelihoods.append(likelihood)
#     print(f"Components: {n_components}, Log-likelihood: {likelihood}")

# # Plot log-likelihood vs number of components
# plt.figure(figsize=(10, 6))
# plt.plot(n_components_range, likelihoods, marker='o')
# plt.xlabel('Number of components')
# plt.ylabel('Log-likelihood')
# plt.title('GMM: Log-likelihood vs Number of Components')
# plt.show()

# # Choose the best number of components (you might want to use a different criterion)
# best_n_components = n_components_range[np.argmax(likelihoods)]
# print(f"\nBest number of components for GMM: {best_n_components}")

# # Fit the GMM with the best number of components
# gmm = GMM(n_components=best_n_components, max_iters=100, tol=1e-4, reg_covar=1e-6)
# gmm.fit(embeddings)

# # Get cluster assignments
# gmm_cluster_assignments = gmm.predict(embeddings)

# # Add GMM cluster assignments to the dataframe
# x['GMM_Cluster'] = gmm_cluster_assignments

# # Create a dictionary of GMM clusters and their words
# gmm_cluster_dict = defaultdict(list)
# for word, cluster in zip(x['words'], x['GMM_Cluster']):
#     gmm_cluster_dict[cluster].append(word)

# # Print GMM cluster information
# print("\nGMM Clustering Results:")
# for cluster, words in gmm_cluster_dict.items():
#     print(f"\nGMM Cluster {cluster}: {len(words)} words")
#     print(", ".join(words[:10]))  # Print first 10 words in each cluster

# # Save GMM results
# x.to_csv('gmm_clustered_word_embeddings.csv', index=True)
# print("\nGMM results saved to 'gmm_clustered_word_embeddings.csv'")

# with open('gmm_cluster_words.txt', 'w') as f:
#     for cluster, words in gmm_cluster_dict.items():
#         f.write(f"GMM Cluster {cluster}: {len(words)} words\n")
#         f.write(", ".join(words) + "\n\n")
# print("GMM cluster words saved to 'gmm_cluster_words.txt'")


# # print("\nPerforming sklearn GMM Analysis on unnormalized data:")
# # bic_scores = []
# # aic_scores = []
# # n_components_range = range(2, 10)

# # for n_components in n_components_range:
# #     gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=10)
# #     gmm.fit(embeddings)
# #     bic_scores.append(gmm.bic(embeddings))
# #     aic_scores.append(gmm.aic(embeddings))
# #     print(f"Components: {n_components}, BIC: {bic_scores[-1]}, AIC: {aic_scores[-1]}")

# # # Plot BIC and AIC scores
# # plt.figure(figsize=(12, 5))
# # plt.subplot(1, 2, 1)
# # plt.plot(n_components_range, bic_scores, marker='o')
# # plt.xlabel('Number of components')
# # plt.ylabel('BIC')
# # plt.title('GMM: BIC Score vs Number of Components')

# # plt.subplot(1, 2, 2)
# # plt.plot(n_components_range, aic_scores, marker='o')
# # plt.xlabel('Number of components')
# # plt.ylabel('AIC')
# # plt.title('GMM: AIC Score vs Number of Components')
# # plt.tight_layout()
# # plt.show()

# # # Choose the best number of components based on lowest BIC score
# # best_n_components_bic = n_components_range[np.argmin(bic_scores)]
# # print(f"\nBest number of components for sklearn GMM (based on BIC): {best_n_components_bic}")

# # # Choose the best number of components based on lowest AIC score
# # best_n_components_aic = n_components_range[np.argmin(aic_scores)]
# # print(f"Best number of components for sklearn GMM (based on AIC): {best_n_components_aic}")

# # # Fit the GMM with the best number of components (using BIC)
# # best_gmm = GaussianMixture(n_components=best_n_components_bic, random_state=42, n_init=10)
# # sklearn_gmm_cluster_assignments = best_gmm.fit_predict(embeddings)

# # # Add sklearn GMM cluster assignments to the dataframe
# # x['sklearn_GMM_Cluster'] = sklearn_gmm_cluster_assignments

# # # Print sklearn GMM cluster information
# # print("\nsklearn GMM Clustering Results:")
# # for cluster in range(best_n_components_bic):
# #     cluster_words = x[x['sklearn_GMM_Cluster'] == cluster]['words'].tolist()
# #     print(f"\nsklearn GMM Cluster {cluster}: {len(cluster_words)} words")
# #     print(", ".join(cluster_words[:10]))  # Print first 10 words in each cluster

# # # Save results
# # x.to_csv('gmm_clustering_results_unnormalized.csv', index=True)
# # print("\nGMM clustering results saved to 'gmm_clustering_results_unnormalized.csv'")

# scaler = StandardScaler()
# embeddings_normalized = scaler.fit_transform(embeddings)

# print("\nPerforming sklearn GMM Analysis:")
# bic_scores = []
# aic_scores = []
# n_components_range = range(1, 15)

# for n_components in n_components_range:
#     gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=10)
#     gmm.fit(embeddings_normalized)
#     bic_scores.append(gmm.bic(embeddings_normalized))
#     aic_scores.append(gmm.aic(embeddings_normalized))
#     print(f"Components: {n_components}, BIC: {bic_scores[-1]}, AIC: {aic_scores[-1]}")

# # Plot BIC and AIC scores
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(n_components_range, bic_scores, marker='o')
# plt.xlabel('Number of components')
# plt.ylabel('BIC')
# plt.title('GMM: BIC Score vs Number of Components')

# plt.subplot(1, 2, 2)
# plt.plot(n_components_range, aic_scores, marker='o')
# plt.xlabel('Number of components')
# plt.ylabel('AIC')
# plt.title('GMM: AIC Score vs Number of Components')
# plt.tight_layout()
# plt.show()

# # Choose the best number of components based on lowest BIC score
# best_n_components_bic = n_components_range[np.argmin(bic_scores)]
# print(f"\nBest number of components for sklearn GMM (based on BIC): {best_n_components_bic}")

# # Choose the best number of components based on lowest AIC score
# best_n_components_aic = n_components_range[np.argmin(aic_scores)]
# print(f"Best number of components for sklearn GMM (based on AIC): {best_n_components_aic}")

# # Fit the GMM with the best number of components (using BIC)
# best_gmm = GaussianMixture(n_components=best_n_components_bic, random_state=42, n_init=10)
# sklearn_gmm_cluster_assignments = best_gmm.fit_predict(embeddings_normalized)

# # Add sklearn GMM cluster assignments to the dataframe
# x['sklearn_GMM_Cluster'] = sklearn_gmm_cluster_assignments

# # Print sklearn GMM cluster information
# print("\nsklearn GMM Clustering Results:")
# for cluster in range(best_n_components_bic):
#     cluster_words = x[x['sklearn_GMM_Cluster'] == cluster]['words'].tolist()
#     print(f"\nsklearn GMM Cluster {cluster}: {len(cluster_words)} words")
#     print(", ".join(cluster_words[:10]))  # Print first 10 words in each cluster

# # Save results
# x.to_csv('gmm_clustering_results.csv', index=True)
# print("\nGMM clustering results saved to 'gmm_clustering_results.csv'")



# print("Running aic and bic analysis for custom implementation")

# likelihood_values = []
# aic_values = []
# bic_values = []

# n_comp_range = range(1,15)
# n_samples, n_features = embeddings.shape

# for n_comp in n_comp_range:
#     gmm = GMM(n_components=n_comp,max_iters=100,tol=1e-4,reg_covar=1e-6)
#     gmm.fit(embeddings)
#     likelihood = gmm.score(embeddings)
#     likelihood_values.append(likelihood)

#     n_params = n_comp*(n_features + n_features * (n_features + 1) / 2) + n_components - 1
#     aic = -2*likelihood + 2*n_params
#     bic = -2*likelihood + np.log(n_samples)*n_params

#     aic_values.append(aic)
#     bic_values.append(bic)

#     print(f"Components: {n_comp}, Log-likelihood: {likelihood}, AIC: {aic}, BIC: {bic}")


# plt.figure(figsize=(15,5))

# plt.subplot(1,3,1)
# plt.plot(n_comp_range,likelihood_values)
# plt.xlabel('k')
# plt.ylabel('Likelihoods')
# plt.title('Log likelihood vs k')


# plt.subplot(1,3,2)
# plt.plot(n_comp_range,aic_values)
# plt.xlabel('k')
# plt.ylabel('AIC')
# plt.title('AIC vs k')


# plt.subplot(1,3,3)
# plt.plot(n_comp_range,bic_values)
# plt.xlabel('k')
# plt.ylabel('Likelihoods')
# plt.title('BIC vs k')

# plt.tight_layout()
# plt.show()


# best_k_likelihood = n_comp_range[np.argmax(likelihood_values)]
# best_k_aic = n_comp_range[np.argmin(aic_values)]
# best_k_bic = n_comp_range[np.argmin(bic_values)]

# print(f"\n Best k as per likelihood:{best_k_likelihood}")
# print(f"Best k ar per AIC : {best_k_aic}")
# print(f"Best k as per BIC : {best_k_bic}")


# best_k = best_k_bic

# gmm = GMM(n_components=best_k,max_iters=100,tol=1e-4, reg_covar=1e-6)
# gmm.fit(embeddings)

# # Get cluster assignments
# gmm_cluster_assignments = gmm.predict(embeddings)

# # Add GMM cluster assignments to the dataframe
# x['GMM_Cluster'] = gmm_cluster_assignments

# # Create a dictionary of GMM clusters and their words
# gmm_cluster_dict = defaultdict(list)
# for word, cluster in zip(x['words'], x['GMM_Cluster']):
#     gmm_cluster_dict[cluster].append(word)

# # Print GMM cluster information
# print(f"\nGMM Clustering Results (using {best_n_components} components):")
# for cluster, words in gmm_cluster_dict.items():
#     print(f"\nGMM Cluster {cluster}: {len(words)} words")
#     print(", ".join(words[:10]))  # Print first 10 words in each cluster



print("\nPerforming PCA Analysis:")

# Initialize PCA with different numbers of components
n_components_list = [3]  # You can adjust these values
pca_results = {}

for n_components in n_components_list:
    # Create a PCA instance
    pca = PCA(n_comps=n_components)
    
    # Fit PCA to the embeddings
    pca.fit(embeddings)
    
    # Transform the embeddings
    reduced_embeddings = pca.transform(embeddings)
    
    # Store the results
    pca_results[n_components] = {
        'pca': pca,
        'reduced_embeddings': reduced_embeddings
    }
    
    print(f"PCA with {n_components} components:")
    print(f"  Original shape: {embeddings.shape}")
    print(f"  Reduced shape: {reduced_embeddings.shape}")
    print(f"  Explained variance ratio sum: {np.sum(pca.explained_variance_ratio):.4f}")
    print(f"  PCA check result: {pca.checkPCA(embeddings)}")
    print()


# Print updated embeddings shape
print("\nUpdated embeddings shape:", reduced_embeddings.shape)

# Save words and their corresponding reduced dimensions to a file
output_file = 'reduced_word_embeddings.txt'

with open(output_file, 'w', encoding='utf-8') as f:
    for word, embedding in zip(x['words'], reduced_embeddings):
        # Convert the embedding vector (3 numbers) to a comma-separated string
        embedding_str = ','.join(map(str, embedding))
        # Write the word followed by the 3 numbers to the file
        f.write(f"{word} --- {embedding_str}\n")

print(f"Reduced word embeddings saved to {output_file}")


# x_vals = reduced_embeddings[:, 0]  # X values (1st dimension)
# y_vals = reduced_embeddings[:, 1]  # Y values (2nd dimension)
# words = x['words']  # List of words

# # Create the scatter plot
# fig, ax = plt.subplots()
# scatter = ax.scatter(x_vals, y_vals)

# # Label the axes and title
# ax.set_xlabel('Dimension 1')
# ax.set_ylabel('Dimension 2')
# ax.set_title('2D Word Embeddings')

# # Enable hover functionality with mplcursors
# cursor = mplcursors.cursor(scatter, hover=True)

# # Display the word when hovering over a point
# @cursor.connect("add")
# def on_add(sel):
#     idx = sel.target.index
#     sel.annotation.set(text=words[idx])

# # Show the plot
# plt.show()

x_vals = reduced_embeddings[:, 0]  
y_vals = reduced_embeddings[:, 1]  
z_vals = reduced_embeddings[:, 2]  
words = x['words']  

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(x_vals, y_vals, z_vals)

ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
ax.set_title('3D Word Embeddings')

cursor = mplcursors.cursor(scatter, hover=True)

@cursor.connect("add")
def on_add(sel):
    idx = sel.index
    sel.annotation.set(text=words[idx])

plt.show()