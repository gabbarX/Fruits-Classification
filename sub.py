import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
lda = LinearDiscriminantAnalysis()

labelEncoder = preprocessing.LabelEncoder()
df["category"] = labelEncoder.fit_transform(df["category"])

X = df.drop(["category", "ID"], axis=1)
y = df["category"]

X = X.values
y = y.values

df_test = pd.read_csv("test.csv")
df_test = df_test.drop(["ID"], axis=1)
X_test = df_test.values

lda.fit(X, y)
X = lda.transform(X)
X_test = lda.transform(X_test)


# Applying knn
knn = KNeighborsClassifier(1)
knn.fit(X, y)
y_pred = knn.predict(X_test)


print(y_pred)
hehe = labelEncoder.inverse_transform(y_pred)
final_df = pd.DataFrame(hehe)
# print(hehe)
final_df.to_csv("file1.csv")
