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


df = pd.read_csv("train.csv")
lof = LocalOutlierFactor(n_neighbors=2)

labelEncoder = preprocessing.LabelEncoder()
df["category"] = labelEncoder.fit_transform(df["category"])


y = df["category"]
df = df.drop(["category", "ID"], axis=1)


zero_percents = (df == 0).sum(axis=0) / len(df)
# Define threshold for percentage of zeros
threshold = 0.92

# Get indices of columns to keep
keep_indices = zero_percents[zero_percents < threshold].index

# Remove columns with high percentage of zeros
df = df[keep_indices]
print(df.head)
X = df.values
y = y.values
