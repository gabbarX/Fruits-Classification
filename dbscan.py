import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")

# print(np.asarray(df))
# print(df.columns["category"].unique())
# print(df["category"])
# del df[df.columns[0]]


X = df.drop(["category", "ID"], axis=1)
y = df["category"]

X = X.values
y = y.values


# Create KMeans instance
kmeans = KMeans(n_clusters=4, random_state=42)

# Fit the model and predict clusters
clusters = kmeans.fit_predict(X)

# Convert clusters to dataframe
df_clusters = pd.DataFrame(clusters, columns=["cluster_label"])


# Print first few rows of data
print(clusters)
