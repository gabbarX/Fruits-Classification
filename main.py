import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")

# print(np.asarray(df))
# print(df.columns["category"].unique())

# del df[df.columns[0]]
X = df.values[:, :-1]
y = df.values[:, -1]

y = np.asarray(y)
print(np.unique(y))
# print(y.)


clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = clf.fit_predict(X)

# plot the results

plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="viridis")
plt.show()
# print(df.isnull().sum())
# kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
# print(kmeans.labels_.shape)
# print(kmeans.cluster_centers_)
# print(df.head())
# print(df.describe)
# print(X)
