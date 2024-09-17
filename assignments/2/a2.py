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
from scipy.cluster import hierarchy as hc
from scipy.spatial.distance import pdist

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.k_means.k_means import KMeans
from models.gmm.gmm import GMM
from models.pca.pca2 import PCA
from models.knn.knn import KNN,calculate_accuracy,min_max_normalize

"""---------------------------------------------------------------------LOAD THE DATASET-----------------------------------------------------------------------------------"""

x = pd.read_feather(r"C:\Users\aryma\Desktop\SMAI\smai\smai-m24-assignments-aryamanmahajan123\data\external\word-embeddings.feather")

print("Initial data shape:", x.shape)
print("Initial data types:")
print(x.dtypes)

embeddings = np.stack(x['vit'].values)

print("Embeddings shape:", embeddings.shape)


"""------------------------------------------------------------------ELBOW METHOD FOR K-Means-----------------------------------------------------------------------------"""

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

kmeans1=8



"""--------------------------------------------------------------Perform K-Means Using Kmeans1-------------------------------------------------------------------------"""


kmeans = KMeans(k=kmeans1, maxiters=500, plot_iters=False, tol=1e-9)
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


"""-------------------------------------------------------------------Run GMM Using Implemented Class------------------------------------------------------------------"""

print("\nPerforming GMM Analysis:")
likelihoods = []
n_components_range = range(1, 9)

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



"""------------------------------------------------------------------RUN GMM USING IN BUILT CLASS---------------------------------------------------------------------"""

print("\nPerforming sklearn GMM Analysis on unnormalized data:")
bic_scores = []
aic_scores = []
n_components_range = range(2, 10)

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=10)
    gmm.fit(embeddings)
    bic_scores.append(gmm.bic(embeddings))
    aic_scores.append(gmm.aic(embeddings))
    print(f"Components: {n_components}, BIC: {bic_scores[-1]}, AIC: {aic_scores[-1]}")

# Plot BIC and AIC scores
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(n_components_range, bic_scores, marker='o')
plt.xlabel('Number of components')
plt.ylabel('BIC')
plt.title('GMM: BIC Score vs Number of Components')

plt.subplot(1, 2, 2)
plt.plot(n_components_range, aic_scores, marker='o')
plt.xlabel('Number of components')
plt.ylabel('AIC')
plt.title('GMM: AIC Score vs Number of Components')
plt.tight_layout()
plt.show()

# Choose the best number of components based on lowest BIC score
best_n_components_bic = n_components_range[np.argmin(bic_scores)]
print(f"\nBest number of components for sklearn GMM (based on BIC): {best_n_components_bic}")

# Choose the best number of components based on lowest AIC score
best_n_components_aic = n_components_range[np.argmin(aic_scores)]
print(f"Best number of components for sklearn GMM (based on AIC): {best_n_components_aic}")

# Fit the GMM with the best number of components (using BIC)
best_gmm = GaussianMixture(n_components=best_n_components_bic, random_state=42, n_init=10)
sklearn_gmm_cluster_assignments = best_gmm.fit_predict(embeddings)

# Add sklearn GMM cluster assignments to the dataframe
x['sklearn_GMM_Cluster'] = sklearn_gmm_cluster_assignments

# Print sklearn GMM cluster information
print("\nsklearn GMM Clustering Results:")
for cluster in range(best_n_components_bic):
    cluster_words = x[x['sklearn_GMM_Cluster'] == cluster]['words'].tolist()
    print(f"\nsklearn GMM Cluster {cluster}: {len(cluster_words)} words")
    print(", ".join(cluster_words[:10]))  # Print first 10 words in each cluster

# Save results
x.to_csv('gmm_clustering_results_unnormalized.csv', index=True)
print("\nGMM clustering results saved to 'gmm_clustering_results_unnormalized.csv'")



"""------------------------------------------------------------------Working with Normalized Data-----------------------------------------------------------------------------"""


scaler = StandardScaler()
embeddings_normalized = scaler.fit_transform(embeddings)

print("\nPerforming sklearn GMM Analysis:")
bic_scores = []
aic_scores = []
n_components_range = range(1, 15)

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=10)
    gmm.fit(embeddings_normalized)
    bic_scores.append(gmm.bic(embeddings_normalized))
    aic_scores.append(gmm.aic(embeddings_normalized))
    print(f"Components: {n_components}, BIC: {bic_scores[-1]}, AIC: {aic_scores[-1]}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(n_components_range, bic_scores, marker='o')
plt.xlabel('Number of components')
plt.ylabel('BIC')
plt.title('GMM: BIC Score vs Number of Components')

plt.subplot(1, 2, 2)
plt.plot(n_components_range, aic_scores, marker='o')
plt.xlabel('Number of components')
plt.ylabel('AIC')
plt.title('GMM: AIC Score vs Number of Components')
plt.tight_layout()
plt.show()

best_n_components_bic = n_components_range[np.argmin(bic_scores)]
print(f"\nBest number of components for sklearn GMM (based on BIC): {best_n_components_bic}")

best_n_components_aic = n_components_range[np.argmin(aic_scores)]
print(f"Best number of components for sklearn GMM (based on AIC): {best_n_components_aic}")

best_gmm = GaussianMixture(n_components=best_n_components_bic, random_state=42, n_init=10)
sklearn_gmm_cluster_assignments = best_gmm.fit_predict(embeddings_normalized)

x['sklearn_GMM_Cluster'] = sklearn_gmm_cluster_assignments

print("\nsklearn GMM Clustering Results:")
for cluster in range(best_n_components_bic):
    cluster_words = x[x['sklearn_GMM_Cluster'] == cluster]['words'].tolist()
    print(f"\nsklearn GMM Cluster {cluster}: {len(cluster_words)} words")
    print(", ".join(cluster_words[:10])) 

x.to_csv('gmm_clustering_results.csv', index=True)
print("\nGMM clustering results saved to 'gmm_clustering_results.csv'")


"""--------------------------------------------------------RUN AIC-BIC Analysis for Implemented Class-----------------------------------------------------------------"""

print("\nRunning aic and bic analysis for custom implementation")

likelihood_values = []
aic_values = []
bic_values = []

n_comp_range = range(1,15)
n_samples, n_features = embeddings.shape

for n_comp in n_comp_range:
    gmm = GMM(n_components=n_comp,max_iters=100,tol=1e-4,reg_covar=1e-6)
    gmm.fit(embeddings)
    likelihood = gmm.score(embeddings)
    likelihood_values.append(likelihood)

    n_params = n_comp*(n_features + n_features * (n_features + 1) / 2) + n_components - 1
    aic = -2*likelihood + 2*n_params
    bic = -2*likelihood + np.log(n_samples)*n_params

    aic_values.append(aic)
    bic_values.append(bic)

    print(f"Components: {n_comp}, Log-likelihood: {likelihood}, AIC: {aic}, BIC: {bic}")


plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.plot(n_comp_range,likelihood_values)
plt.xlabel('k')
plt.ylabel('Likelihoods')
plt.title('Log likelihood vs k')


plt.subplot(1,3,2)
plt.plot(n_comp_range,aic_values)
plt.xlabel('k')
plt.ylabel('AIC')
plt.title('AIC vs k')


plt.subplot(1,3,3)
plt.plot(n_comp_range,bic_values)
plt.xlabel('k')
plt.ylabel('Likelihoods')
plt.title('BIC vs k')

plt.tight_layout()
plt.show()


best_k_likelihood = n_comp_range[np.argmax(likelihood_values)]
best_k_aic = n_comp_range[np.argmin(aic_values)]
best_k_bic = n_comp_range[np.argmin(bic_values)]

print(f"\n Best k as per likelihood:{best_k_likelihood}")
print(f"Best k ar per AIC : {best_k_aic}")
print(f"Best k as per BIC : {best_k_bic}")


kgmm1 = best_k_bic

gmm = GMM(n_components=kgmm1,max_iters=100,tol=1e-4, reg_covar=1e-6)
gmm.fit(embeddings)

gmm_cluster_assignments = gmm.predict(embeddings)

x['GMM_Cluster'] = gmm_cluster_assignments

gmm_cluster_dict = defaultdict(list)
for word, cluster in zip(x['words'], x['GMM_Cluster']):
    gmm_cluster_dict[cluster].append(word)

print(f"\nGMM Clustering Results (using {best_n_components} components):")
for cluster, words in gmm_cluster_dict.items():
    print(f"\nGMM Cluster {cluster}: {len(words)} words")
    print(", ".join(words[:10])) 


"""--------------------------------------------------------------------RUN PCA TO VISUALIZE DATA ON 2D and 3D----------------------------------------------------------"""

print("\nPerforming PCA Analysis:")

n_components_list = [3] 
pca_results = {}

for n_components in n_components_list:
    pca = PCA(n_comps=n_components)
    pca.fit(embeddings)
    
    
    reduced_embeddings = pca.transform(embeddings)
    
    pca_results[n_components] = {
        'pca': pca,
        'reduced_embeddings': reduced_embeddings
    }
    
    print(f"  PCA with {n_components} components:")
    print(f"  Original shape: {embeddings.shape}")
    print(f"  Reduced shape: {reduced_embeddings.shape}")
    print(f"  Explained variance ratio sum: {np.sum(pca.explained_variance_ratio):.4f}")
    print(f"  PCA check result: {pca.checkPCA(embeddings)}")
    print()


print("\nUpdated embeddings shape:", reduced_embeddings.shape)

output_file = 'reduced_word_embeddings.txt'

with open(output_file, 'w', encoding='utf-8') as f:
    for word, embedding in zip(x['words'], reduced_embeddings):
        embedding_str = ','.join(map(str, embedding))
        f.write(f"{word} --- {embedding_str}\n")

print(f"Reduced word embeddings saved to {output_file}")


x_vals = reduced_embeddings[:, 0] 
y_vals = reduced_embeddings[:, 1]  
words = x['words']  

fig, ax = plt.subplots()
scatter = ax.scatter(x_vals, y_vals)

ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_title('2D Word Embeddings')

cursor = mplcursors.cursor(scatter, hover=True)

@cursor.connect("add")
def on_add(sel):
    idx = sel.index
    sel.annotation.set(text=words[idx])

plt.show()

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


"""----------------------------------------------------K-MEANS BASED ON K=K2 OBTAINED BY 2D VISUALIZATION-----------------------------------"""

"""------------------------------------------------------------------------K=5--------------------------------------------------------------"""

k2 = 5

kmeans = KMeans(k=k2, maxiters=500, plot_iters=False, tol=1e-9)
kmeans.fit(embeddings)

print("Final cost:", np.mean(kmeans.get_cost()))
x['Cluster'] = kmeans.labels

cluster_dict = defaultdict(list)

for word, cluster in zip(x['words'], x['Cluster']):
    cluster_dict[cluster].append(word)

for cluster, words in cluster_dict.items():
    print(f"\nCluster {cluster}: {len(words)} words")
    print(", ".join(words))



"""-------------------------------------------------------------PCA + KMeans Clustering-----------------------------------------------------"""
"""------------------------------------------------------------------SCREE PLOT-------------------------------------------------------------"""

n_components = embeddings.shape[1]  
pca = PCA(n_comps=n_components)
pca.fit(embeddings)

cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio)

plt.figure(figsize=(15, 6))
plt.plot(range(1, n_components + 1), cumulative_variance_ratio, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Scree Plot')

plt.axhline(y=0.95, color='r', linestyle='--')

n_components_95 = next(i for i, ratio in enumerate(cumulative_variance_ratio) if ratio >= 0.95) + 1
plt.annotate(f'95% variance explained\nby {n_components_95} components', 
             xy=(n_components_95, 0.95), xytext=(n_components_95+10, 0.85),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.grid(True)

axins = plt.axes([0.5, 0.2, 0.4, 0.4])
axins.plot(range(1, 201), cumulative_variance_ratio[:200], 'ro-')
axins.set_title('First 200 Components')
axins.grid(True)

plt.show()

print(f"Number of components needed to explain 95% of the variance: {n_components_95}")

print("\nExplained variance ratio for first 10 components:")
print(pca.explained_variance_ratio[:10])
print("\nExplained variance ratio for last 10 components:")
print(pca.explained_variance_ratio[-10:])


"""------------------------------------------------REDUCING DATASET AS PER THE ESTIMATED DIMENSIONS----------------------------------------"""


pca_r = PCA(n_comps = n_components_95)
pca_r.fit(embeddings)
reduced_dataset = pca_r.transform(embeddings)
output_file = r'C:\Users\aryma\Desktop\SMAI\smai\smai-m24-assignments-aryamanmahajan123\data\interim\reduced_word_embeddings_dataset.txt'

with open(output_file, 'w', encoding='utf-8') as f:
    for word, embedding in zip(x['words'], reduced_dataset):
        embedding_str = ','.join(map(str, embedding))
        f.write(f"{word} --- {embedding_str}\n")

print(f"Reduced word embeddings saved to {output_file}")


"""--------------------------------------------------Running elbow method on Reduced Dataset------------------------------------------"""


print("Running K Means elbow method on reduced dataset")
print(f"Shape of reduced dataset : {reduced_dataset.shape}")
costs_r = []
k_values_r = range(1, 15)  

for i in k_values_r:
    kmeans = KMeans(k=i, maxiters=500, plot_iters=False, tol=1e-9)
    kmeans.fit(reduced_dataset)
    cost = np.mean(kmeans.get_cost())
    costs_r.append(cost)

plt.plot(k_values_r, costs_r, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Cost (WCSS)')
plt.title('Cost vs Number of Clusters')
plt.show()


kmeans3 = 8



"""--------------------------------------------------Perform K Means on Reduced Dataset with k = kmeans3----------------------------------------"""



print("Performing K Means clustering on reduced dataset with k = kmeans3")
kmeans_r = KMeans(k=kmeans3,maxiters=100,plot_iters=False,tol=1e-4)
kmeans_r.fit(reduced_dataset)

print("Final cost:", np.mean(kmeans_r.get_cost()))
x['Cluster'] = kmeans_r.labels

cluster_dict = defaultdict(list)

for word, cluster in zip(x['words'], x['Cluster']):
    cluster_dict[cluster].append(word)

for cluster, words in cluster_dict.items():
    print(f"\nCluster {cluster}: {len(words)} words")
    print(", ".join(words))

output_dir = r'C:\Users\aryma\Desktop\SMAI\smai\smai-m24-assignments-aryamanmahajan123\data\interim'

csv_file_path = f'{output_dir}\\clustered_word_embeddings_reduced.csv'
x.to_csv(csv_file_path, index=True)
print(f"\nResults saved to '{csv_file_path}'")

text_file_path = f'{output_dir}\\cluster_words_reduced.txt'
with open(text_file_path, 'w') as f:
    for cluster, words in cluster_dict.items():
        f.write(f"Cluster {cluster}: {len(words)} words\n")
        f.write(", ".join(words) + "\n\n")
print(f"Cluster words saved to '{text_file_path}'")



"""-------------------------------------------Perform GMM on original dataset with num_components=k2------------------------------------"""


gmm = GMM(n_components=k2,max_iters=100,tol=1e-4,reg_covar=1e-6)
gmm.fit(embeddings)

gmm_cluster_assignments = gmm.predict(embeddings)

x['GMM_Cluster'] = gmm_cluster_assignments

gmm_cluster_dict = defaultdict(list)
for word, cluster in zip(x['words'], x['GMM_Cluster']):
    gmm_cluster_dict[cluster].append(word)

print("\nGMM Clustering Results:")
for cluster, words in gmm_cluster_dict.items():
    print(f"\nGMM Cluster {cluster}: {len(words)} words")
    print(", ".join(words[:10]))  

csv_file_path = f'{output_dir}\\clustered_word_embeddings_reduced_gmm.csv'

x.to_csv(csv_file_path, index=True)
print("\nGMM results saved to 'gmm_clustered_word_embeddings.csv'")

text_file_path = f'{output_dir}\\cluster_words_gmm.txt'

with open(text_file_path, 'w') as f:
    for cluster, words in gmm_cluster_dict.items():
        f.write(f"GMM Cluster {cluster}: {len(words)} words\n")
        f.write(", ".join(words) + "\n\n")
print(f"GMM cluster words saved to '{text_file_path}' ")



"""---------------------------------------------------------------------------------PCA + GMMs------------------------------------------------------------------------"""
"""------------------------------------------------------------------Optimal num of clusters for reduced dataset------------------------------------------------------"""


print("Running aic and bic analysis for reduced dataset")

likelihood_values = []
aic_values = []
bic_values = []

n_comp_range = range(1,15)
n_samples, n_features = reduced_dataset.shape

for n_comp in n_comp_range:
    gmm = GMM(n_components=n_comp,max_iters=100,tol=1e-4,reg_covar=1e-6)
    gmm.fit(reduced_dataset)
    likelihood = gmm.score(reduced_dataset)
    likelihood_values.append(likelihood)

    n_params = n_comp*(n_features + n_features * (n_features + 1) / 2) + n_components - 1
    aic = -2*likelihood + 2*n_params
    bic = -2*likelihood + np.log(n_samples)*n_params

    aic_values.append(aic)
    bic_values.append(bic)

    print(f"Components: {n_comp}, Log-likelihood: {likelihood}, AIC: {aic}, BIC: {bic}")


plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.plot(n_comp_range,likelihood_values)
plt.xlabel('k')
plt.ylabel('Likelihoods')
plt.title('Log likelihood vs k')

plt.subplot(1,3,2)
plt.plot(n_comp_range,aic_values)
plt.xlabel('k')
plt.ylabel('AIC')
plt.title('AIC vs k')

plt.subplot(1,3,3)
plt.plot(n_comp_range,bic_values)
plt.xlabel('k')
plt.ylabel('BIC')
plt.title('BIC vs k')

plt.tight_layout()
plt.show()


best_k_likelihood = n_comp_range[np.argmax(likelihood_values)]
best_k_aic = n_comp_range[np.argmin(aic_values)]
best_k_bic = n_comp_range[np.argmin(bic_values)]

print(f"\n Best k as per likelihood:{best_k_likelihood}")
print(f"Best k ar per AIC : {best_k_aic}")
print(f"Best k as per BIC : {best_k_bic}")


kgmm3 = best_k_bic



"""-------------------------------------------------------------Run GMM on reduced dataset with num components = kgmm3-------------------------------------------------"""

gmm = GMM(n_components=kgmm3,max_iters=100,tol=1e-4, reg_covar=1e-6)
gmm.fit(embeddings)

gmm_cluster_assignments = gmm.predict(embeddings)

x['GMM_Cluster'] = gmm_cluster_assignments

gmm_cluster_dict = defaultdict(list)
for word, cluster in zip(x['words'], x['GMM_Cluster']):
    gmm_cluster_dict[cluster].append(word)

print(f"\nGMM Clustering Results (using {kgmm3} components):")
for cluster, words in gmm_cluster_dict.items():
    print(f"\nGMM Cluster {cluster}: {len(words)} words")
    print(", ".join(words[:10])) 

text_file_path = f'{output_dir}\\cluster_words_gmm_reduced.txt'

with open(text_file_path, 'w') as f:
    for cluster, words in gmm_cluster_dict.items():
        f.write(f"GMM Cluster {cluster}: {len(words)} words\n")
        f.write(", ".join(words) + "\n\n")
print(f"GMM cluster words saved to '{text_file_path}' ")


"""----------------------------------------------------Comparisons of the three schemes----------------------------------------------------"""

"""-------------------------------------------------------------------o---------------------------------------------------------------------"""









"""-------------------------------------------------------------------------HIERARCHICHCAL CLUSTERING------------------------------------------------------------------"""

linkage_methods = ['single', 'complete', 'average', 'ward']
distance_metrics = ['euclidean', 'cityblock']  # cityblock is Manhattan distance

def plot_dendrogram(Z, title):
    plt.figure(figsize=(10, 7))
    hc.dendrogram(Z)
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()

for metric in distance_metrics:
    for method in linkage_methods:
        try:
            distMatrix = pdist(embeddings, metric=metric)
            
            Z = hc.linkage(distMatrix, method=method)
            
            plot_dendrogram(Z, f'Dendrogram ({metric} distance, {method} linkage)')
            
            print(f"Linkage matrix shape for {metric} distance and {method} linkage: {Z.shape}")
            print("First few rows of the linkage matrix:")
            print(Z[:5])
            print("\n")
        except Exception as e:
            print(f"Error occurred with {metric} distance and {method} linkage: {str(e)}")
            print("\n")

kbest1 = 5  
kbest2 = 10 

def create_and_visualize_clusters(Z, k, title, words):
    clusters = hc.fcluster(Z, t=k, criterion='maxclust')
    
    plt.figure(figsize=(10, 7))
    hc.dendrogram(Z)
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.axhline(y=hc.dendrogram(Z)['dcoord'][0][1], color='r', linestyle='--')
    plt.show()
    
    print(f"Number of words: {len(words)}")
    print(f"Number of clusters: {len(clusters)}")
    
    min_length = min(len(words), len(clusters))
    words = words[:min_length]
    clusters = clusters[:min_length]
    
    word_cluster_map = pd.DataFrame({'Word': words, 'Cluster': clusters})
    word_cluster_map = word_cluster_map.sort_values('Cluster')
    
    print(f"Word-to-Cluster mapping for k={k}:")
    for cluster in range(1, k+1):
        cluster_words = word_cluster_map[word_cluster_map['Cluster'] == cluster]['Word'].tolist()
        print(f"Cluster {cluster}: {', '.join(cluster_words)}")
    print("\n")

    return word_cluster_map

for metric in distance_metrics:
    distMatrix = pdist(embeddings, metric=metric)
    
    method = 'ward' if metric == 'euclidean' else 'average'
    Z = hc.linkage(distMatrix, method=method)
    
    print(f"Shape of embeddings: {embeddings.shape}")
    print(f"Number of words: {len(words)}")
    print(f"Shape of linkage matrix Z: {Z.shape}")
    
    try:
        word_cluster_map1 = create_and_visualize_clusters(Z, kbest1, f'Dendrogram with {kbest1} clusters ({metric} distance)', words)
        word_cluster_map2 = create_and_visualize_clusters(Z, kbest2, f'Dendrogram with {kbest2} clusters ({metric} distance)', words)
        
        word_cluster_map1.to_csv(f'word_cluster_map_{metric}_{kbest1}.csv', index=False)
        word_cluster_map2.to_csv(f'word_cluster_map_{metric}_{kbest2}.csv', index=False)
        print(f"Word-to-cluster mappings saved to CSV files for {metric} distance.")
    except Exception as e:
        print(f"Error occurred with {metric} distance: {str(e)}")
    print("\n")




"""------------------------------------------------------------------------------PCA + KNN----------------------------------------------------------------------------"""
import time

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


def encode_categorical(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            unique_values = df[column].unique()
            value_to_int = {value: i for i, value in enumerate(unique_values)}
            df[column] = df[column].map(value_to_int)
    return df

def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    return (X - mean) / std, mean, std

# Load the data
sp_file_path = r'C:\Users\aryma\Desktop\SMAI\smai\smai-m24-assignments-aryamanmahajan123\data\external\spotify.csv'
sp = pd.read_csv(sp_file_path)

# Encode categorical variables
sp_encoded = encode_categorical(sp)

# Set random seed and shuffle the data
np.random.seed(42)
sp_encoded = sp_encoded.sample(frac=1).reset_index(drop=True)

# Split the data
train_size = int(0.8 * len(sp_encoded))
val_size = int(0.1 * len(sp_encoded))
train_sp = sp_encoded[:train_size]
val_sp = sp_encoded[train_size:train_size + val_size]
test_sp = sp_encoded[train_size + val_size:]

print(f"\nShape of Spotify training data: {train_sp.shape}")

# Identify numeric columns (assuming the last column is the target variable)
numeric_columns = train_sp.select_dtypes(include=[np.number]).columns[:-1]

# Scale the features
X_train_scaled, train_mean, train_std = standardize(train_sp[numeric_columns].values)

# Perform PCA
n_components = len(numeric_columns)
pca = PCA(n_comps=n_components)  # Adjust this line based on your PCA class initialization
pca.fit(X_train_scaled)

# Get explained variance ratio (adjust this based on your PCA class methods)
explained_variance_ratio = pca.explained_variance_ratio

# Calculate cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Plot the scree plot
plt.figure(figsize=(15, 6))
plt.plot(range(1, n_components + 1), cumulative_variance_ratio, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Scree Plot')

n_components_90 = next(i for i, ratio in enumerate(cumulative_variance_ratio) if ratio >= 0.90) + 1
plt.annotate(f'90% variance explained\nby {n_components_90} components', 
             xy=(n_components_90, 0.90), xytext=(n_components_90+1, 0.80),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.grid(True)
plt.show()

print(f"Number of components needed to explain 90% of the variance: {n_components_90}")


pca_90 = PCA(n_comps=n_components_90) 
pca_90.fit(X_train_scaled)
X_train_pca = pca_90.transform(X_train_scaled)  

y_train = train_sp.iloc[:, -1].values

X_train_normalized = min_max_normalize(X_train_pca)

X_val = val_sp[numeric_columns].values
X_val_scaled = (X_val - train_mean) / train_std
X_val_pca = pca_90.transform(X_val_scaled)
X_val_normalized = min_max_normalize(X_val_pca)
y_val = val_sp.iloc[:, -1].values
Xtbu = X_train_normalized[:100]
ytbu = y_train[:100]
xvalu = X_val_normalized[:100]
yvalu = y_val[:100]

knn = KNN(k=5, distance_metric='cosine')
knn.fit(Xtbu, ytbu)
s = time.time()
y_pred = knn.predict(xvalu)
e = time.time()
accuracy = calculate_accuracy(yvalu, y_pred)
print(f"Accuracy: {accuracy:.4f}")
precision, recall, f1 = calculate_precision_recall_f1(yvalu, y_pred)

print(f"Accuracy on validation = {accuracy:.4f}")
print(f"Precision on validation = {precision:.4f}")
print(f"Recall on validation = {recall:.4f}")
print(f"F1 score on validation = {f1:.4f}")