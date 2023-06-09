import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

df = pd.read_csv("train.csv")
labelEncoder = LabelEncoder()
lof = LocalOutlierFactor(n_neighbors=120)

df["category"] = labelEncoder.fit_transform(df["category"])

X = df.drop(["category", "ID"], axis=1)
y = df["category"]

X = X.values
y = y.values

# Applying Knn to have cluster labels as external features
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X, y)
cluster_labels = knn.predict(X)
X = np.hstack((X, cluster_labels.reshape(-1, 1)))

# Applying PCA on the data to reduce dimension of the data
n_components = 450
pca = PCA(n_components=n_components)
pca.fit(X)
X = pca.transform(X)

# Applying LDA on the data to increase between class seperation
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
X = lda.transform(X)


# Applying LOF on the data in order to remove the outliers
y_pred = lof.fit_predict(X)
x_train = X[y_pred != -1]
y_train = y[y_pred != -1]


# reading the test data
df_test = pd.read_csv("test.csv")
df_test = df_test.drop(["ID"], axis=1)
X_test = df_test.values

# using cluster labels as external features
cluster_labels = knn.predict(X_test)
X_test = np.hstack((X_test, cluster_labels.reshape(-1, 1)))


# applying PCA and LDA on the test data
X_test = pca.transform(X_test)
X_test = lda.transform(X_test)


# Create a logistic regression model
lr = LogisticRegression()


lr = LogisticRegression(max_iter=10000)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(lr, X, y, cv=cv)

# print the mean and standard deviation of the scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))


# Fit the model on the training data
lr.fit(x_train, y_train)

# Make predictions on the test data
y_pred = lr.predict(X_test)

hehe = labelEncoder.inverse_transform(y_pred)
final_df = pd.DataFrame(hehe)
final_df.to_csv("ankitsub3.csv")
