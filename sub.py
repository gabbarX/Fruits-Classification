import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA


df = pd.read_csv("train.csv")
scalar = StandardScaler()
labelEncoder = preprocessing.LabelEncoder()
lof = LocalOutlierFactor(n_neighbors = 2)

df["category"] = labelEncoder.fit_transform(df["category"])

X = df.drop(["category", "ID"], axis=1)
y = df["category"]

X = X.values
y = y.values

y_pred = lof.fit_predict(X)
x_train = X[y_pred != -1]
y_train = y[y_pred != -1]



df_test = pd.read_csv("test.csv")
df_test = df_test.drop(["ID"], axis=1)
X_test = df_test.values

# lda.fit(X, y)
# X = lda.transform(X)
# X_test = lda.transform(X_test)


# Applying knn
# knn = KNeighborsClassifier(1)
# knn.fit(X, y)
# y_pred = knn.predict(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

lr = LogisticRegression(max_iter=1000)
lr.fit(x_train, y_train)

# Step 6: Evaluate the model
y_pred = lr.predict(X_test)


print(y_pred)
hehe = labelEncoder.inverse_transform(y_pred)
final_df = pd.DataFrame(hehe)
# print(hehe)
final_df.to_csv("rf.csv")
