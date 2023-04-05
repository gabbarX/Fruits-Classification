from sklearn.cluster import KMeans
import numpy as np

# generate some example data
X = np.random.rand(100, 2)

# create a KMeans object and fit the data
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

# print the cluster labels for each data point
print(kmeans.labels_)

# print the coordinates of the cluster centroids
print(kmeans.cluster_centers_)
