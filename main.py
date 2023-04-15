import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("train.csv")

# rf_clf = RandomForestClassifier(n_estimators=2700, max_depth=109, random_state=42)

labelEncoder = preprocessing.LabelEncoder()
df["category"] = labelEncoder.fit_transform(df["category"])

X = df.drop(["category", "ID"], axis=1)
y = df["category"]

X = X.values

# X = scalar.fit_transform(X, y)
y = y.values


n_components = 450
pca = PCA(n_components=n_components)
X = pca.fit_transform(X)

lda = LinearDiscriminantAnalysis()
X = lda.fit_transform(X, y)

# knn = KNeighborsClassifier(n_neighbors=30)
# knn.fit(X, y)
# cluster_labels = knn.predict(X)
# X = np.hstack((X, cluster_labels.reshape(-1, 1)))

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# iso = IsolationForest(
#     n_estimators=100, max_samples="auto", contamination="auto", random_state=42
# )
# y_pred = iso.fit_predict(X)
# x_train = X[y_pred != -1]
# y_train = y[y_pred != -1]


lr = LogisticRegression(max_iter=10000)
# rf_clf.fit(x_train, y_train)

lr.fit(x_train, y_train)
# y_pred = rf_clf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("AccuracyLOFRF:", accuracy)
