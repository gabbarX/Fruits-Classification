import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
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
from sklearn.model_selection import KFold, cross_val_score

df = pd.read_csv("train.csv")
lof = LocalOutlierFactor(n_neighbors=120)

labelEncoder = preprocessing.LabelEncoder()
df["category"] = labelEncoder.fit_transform(df["category"])


y = df["category"]
df = df.drop(["category", "ID"], axis=1)

print(df.head)
X = df.values
y = y.values


# LOF
y_pred = lof.fit_predict(X)
X = X[y_pred != -1]
y = y[y_pred != -1]


n_components = 520
pca = PCA(n_components=n_components)
pca.fit(X)
X = pca.transform(X)

lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
X = lda.transform(X)

# kmeancluster = 35
# knn = KNeighborsClassifier(n_neighbors=30)
# knn.fit(X, y)
# cluster_labels = knn.predict(X)
# X = np.hstack((X, cluster_labels.reshape(-1, 1)))


# kmeans = KMeans(n_clusters=kmeancluster, random_state=42)
# kmeans.fit(X)
# cluster_labels = kmeans.predict(X)
# X = np.hstack((X, cluster_labels.reshape(-1, 1)))


# # Remove columns with low variance
# selector = VarianceThreshold()
# selector.fit_transform(df)
# df = df[df.columns[selector.get_support(indices=True)]]


# print(df.head)


x_train = X
y_train = y


# print(keep_indices)
df_test = pd.read_csv("test.csv")
df_test = df_test.drop(["ID"], axis=1)
# df_test = df_test[keep_indices]


print(df_test.head)


X_test = df_test.values
X_test = pca.transform(X_test)
X_test = lda.transform(X_test)


# kmeans = KMeans(n_clusters=kmeancluster, random_state=42)
# kmeans.fit(X_test)
# cluster_labels = kmeans.predict(X_test)
# X_test = np.hstack((X_test, cluster_labels.reshape(-1, 1)))

logisticRiter = 1500

log_reg = LogisticRegression(max_iter=logisticRiter)

# define the number of folds
k = 5

# define the cross-validation method
cv = KFold(n_splits=k, shuffle=True, random_state=42)

# perform cross-validation
scores = cross_val_score(log_reg, X, y, cv=cv)

# print the mean and standard deviation of the scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))


lr = LogisticRegression(max_iter=logisticRiter)
lr.fit(x_train, y_train)

# Step 6: Evaluate the model
y_pred = lr.predict(X_test)


print(y_pred)
hehe = labelEncoder.inverse_transform(y_pred)
final_df = pd.DataFrame(hehe)

# print(hehe)
final_df.to_csv("dhakasub.csv")
