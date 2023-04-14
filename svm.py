import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
from sklearn import svm


df = pd.read_csv("train.csv")
labelEncoder = preprocessing.LabelEncoder()
lof = LocalOutlierFactor(n_neighbors=20)

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


# Using grid search to find optimal hyperparameters for SVM
parameters = {
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "C": [1, 10, 100, 0.1],
    "gamma": [0.01, 0.1, 1],
}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(x_train, y_train)
best_params = clf.best_params_

# Train SVM model using best hyperparameters
svm_model = svm.SVC(
    kernel=best_params["kernel"], C=best_params["C"], gamma=best_params["gamma"]
)
svm_model.fit(x_train, y_train)

# Predict on test data using the trained SVM model
y_pred = svm_model.predict(X_test)

hehe = labelEncoder.inverse_transform(y_pred)
final_df = pd.DataFrame(hehe)
final_df.to_csv("svm2.csv")
