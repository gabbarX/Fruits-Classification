import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

df = pd.read_csv("train.csv")

print(df.head)
# del df[df.columns[0]]
X = df.values[:, :-1]
y = df.values[:, -1]

# print(df.isnull().sum())
# kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
# print(kmeans.labels_.shape)
# print(kmeans.cluster_centers_)
# print(df.head())
# print(df.describe)
# print(X)
