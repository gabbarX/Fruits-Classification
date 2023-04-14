import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("train.csv")
scalar = StandardScaler()
labelEncoder = LabelEncoder()
lof = LocalOutlierFactor(n_neighbors=20)

df["category"] = labelEncoder.fit_transform(df["category"])

X = df.drop(["category", "ID"], axis=1)
y = df["category"]

X = X.values
y = y.values

X = scalar.fit_transform(X, y)
y_pred = lof.fit_predict(X)
x_train = X[y_pred != -1]
y_train = y[y_pred != -1]

df_test = pd.read_csv("test.csv")
df_test = df_test.drop(["ID"], axis=1)
X_test = df_test.values

# Define a range of hyperparameters to search over
param_grid = {"C": [0.01, 0.1, 1, 10, 100], "penalty": ["l1", "l2"]}

# Create a logistic regression model
lr = LogisticRegression(max_iter=5000)

# Create a grid search object with cross-validation
grid_search = GridSearchCV(lr, param_grid, cv=5)

# Fit the grid search object to the data
grid_search.fit(x_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

# Create a logistic regression model with the best hyperparameters
lr = LogisticRegression(**best_params, max_iter=5000)

# Fit the model on the training data
lr.fit(x_train, y_train)

# Make predictions on the test data
y_pred = lr.predict(X_test)

hehe = labelEncoder.inverse_transform(y_pred)
final_df = pd.DataFrame(hehe)
final_df.to_csv("rf3.csv")
