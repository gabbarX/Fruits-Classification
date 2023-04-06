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
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, StandardScaler


df = pd.read_csv("train.csv")
lda = LinearDiscriminantAnalysis()


scalar = StandardScaler()


labelEncoder = preprocessing.LabelEncoder()
df["category"] = labelEncoder.fit_transform(df["category"])

X = df.drop(["category", "ID"], axis=1)
y = df["category"]

X = X.values

X = scalar.fit_transform(X, y)
y = y.values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lof = LocalOutlierFactor(n_neighbors = 2)
y_pred = lof.fit_predict(X)
x_train = X[y_pred != -1]
y_train = y[y_pred != -1]

lr = LogisticRegression(max_iter=10000)

lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

# print(len(y_pred))
# print(len(y_test))

accuracy = accuracy_score(y_test, y_pred)
print("AccuracyLOFLR:", accuracy)







# print(y_pred)

# labelEncoder = preprocessing.LabelEncoder()
# df["category"] = labelEncoder.fit_transform(df["category"])

# X = df.drop(["category", "ID"], axis=1)
# y = df["category"]

# X = X.values
# y = y.values

# # print(X)
# # Applied PCA to the data
# pca = PCA(n_components=X.shape[0])
# X = pca.fit_transform(X, y)
# # print(X)
# # print(y)

# x_train, x_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# print(x_train)

# knn = KNeighborsClassifier(5)
# knn.fit(x_train, y_train)
# y_pred = knn.predict(x_test)

# print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# # clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
# # y_pred = clf.fit_predict(X)

# # plot the results

# # plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="viridis")
# # plt.show()
# # print(df.isnull().sum())
# # kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
# # print(kmeans.labels_.shape)
# # print(kmeans.cluster_centers_)
# # print(df.head())
# # print(df.describe)
# # print(X)
