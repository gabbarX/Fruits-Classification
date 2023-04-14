import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA


df = pd.read_csv("train.csv")
labelEncoder = preprocessing.LabelEncoder()
lof = LocalOutlierFactor(n_neighbors=20)

df["category"] = labelEncoder.fit_transform(df["category"])

X = df.drop(["category", "ID"], axis=1)
y = df["category"]

X = X.values
y = y.values

# Add KNN clustering labels as additional features
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X, y)
cluster_labels = knn.predict(X)
X = np.hstack((X, cluster_labels.reshape(-1, 1)))

# X = scalar.fit_transform(X, y)
y_pred = lof.fit_predict(X)
x_train = X[y_pred != -1]
y_train = y[y_pred != -1]


df_test = pd.read_csv("test.csv")
df_test = df_test.drop(["ID"], axis=1)

# Add KNN clustering labels as additional features to test data
cluster_labels_test = knn.predict(df_test)
X_test = np.hstack((df_test.values, cluster_labels_test.reshape(-1, 1)))

# X_test = scalar.transform(X_test)

# Define the hyperparameters to tune
param_grid = {"C": [0.1, 1, 10], "penalty": ["l1", "l2"], "max_iter": [1000, 5000]}

# Perform grid search to find the best hyperparameters
lr = LogisticRegression()
grid_search = GridSearchCV(lr, param_grid, cv=5)
grid_search.fit(x_train, y_train)

# Fit the logistic regression model with the best hyperparameters found
best_lr = grid_search.best_estimator_
best_lr.fit(x_train, y_train)

y_pred = best_lr.predict(X_test)

hehe = labelEncoder.inverse_transform(y_pred)
final_df = pd.DataFrame(hehe)
final_df.to_csv("cluster.csv")
