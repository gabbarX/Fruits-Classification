import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


df = pd.read_csv("train.csv")
lda = LinearDiscriminantAnalysis()


labelEncoder = preprocessing.LabelEncoder()
df["category"] = labelEncoder.fit_transform(df["category"])

X = df.drop(["category", "ID"], axis=1)
y = df["category"]

X = X.values
y = y.values


# lda.fit(X, y)
# X = lda.transform(X)
X = lda.fit_transform(X, y)

# print(X)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Applying knn
knn = KNeighborsClassifier(5)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# print(y_pred)
