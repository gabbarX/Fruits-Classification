import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
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


# lda.fit(X, y)
# X = lda.transform(X)
# X = lda.fit_transform(X, y)

# print(X)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# # Applying knn
# knn = KNeighborsClassifier(5)
# knn.fit(x_train, y_train)
# y_pred = knn.predict(x_test)
# print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# rf = RandomForestClassifier(n_estimators=1000, max_depth=30)
# rf.fit(x_train, y_train)

lr = LogisticRegression(max_iter=10000)
lr.fit(x_train, y_train)

# # Step 6: Evaluate the model
# y_pred = rf.predict(x_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)


y_pred = lr.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("AccuracyLR:", accuracy)

# print(y_pred)
