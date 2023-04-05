import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA


df = pd.read_csv("train.csv")

# print(np.asarray(df))
# print(df.columns["category"].unique())
# print(df["category"])
# del df[df.columns[0]]

labelEncoder = preprocessing.LabelEncoder()
df["category"] = labelEncoder.fit_transform(df["category"])

X = df.drop(["category", "ID"], axis=1)
y = df["category"]

X = X.values
y = y.values

# print(X)
# Applied PCA to the data
# pca = PCA(n_components=X.shape[0])
# X = pca.fit_transform(X, y)
# print(X)
# print(y)


df_test = pd.read_csv("test.csv")
df_test = df_test.drop(["ID"], axis=1)
X_test = df_test.values
# print(X_test)
# Applying knn
knn = KNeighborsClassifier(1)
knn.fit(X, y)


# pca = PCA(n_components=X_test.shape[0])
# X_test = pca.fit_transform(X_test)

y_pred = knn.predict(X_test)


hehe = labelEncoder.inverse_transform(y_pred)
final_df = pd.DataFrame(hehe)
# print(hehe)
final_df.to_csv("file1.csv")
