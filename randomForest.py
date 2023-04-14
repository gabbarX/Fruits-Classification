# Import the necessary libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import LocalOutlierFactor

# Load the data (in this example, we'll use the iris dataset)
df = pd.read_csv("train.csv")
# lda = LinearDiscriminantAnalysis()

from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

labelEncoder = preprocessing.LabelEncoder()
df["category"] = labelEncoder.fit_transform(df["category"])

X = df.drop(["category", "ID"], axis=1)
y = df["category"]

X = X.values

# X = scalar.fit_transform(X, y)
y = y.values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the parameter grid to search over
param_grid = {
    "n_neighbors": [2, 5, 10, 20],
    "contamination": [0.01, 0.05, 0.1],
    "metric": ["euclidean", "manhattan", "mahalanobis"],
}

# Instantiate the random forest classifier and the grid search object
lof = LocalOutlierFactor()
grid_search = GridSearchCV(lof, param_grid, cv=5, n_jobs=-1)

# Fit the grid search object to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)
