import pandas as pd
from sklearn import preprocessing
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Load the Iris dataset
# iris = load_iris()

df = pd.read_csv("train.csv")

# print(np.asarray(df))
# print(df.columns["category"].unique())
# print(df["category"])
# del df[df.columns[0]]

labelEncoder = preprocessing.LabelEncoder()
df["category"] = labelEncoder.fit_transform(df["category"])

X = df.drop(["category", "ID"], axis=1)
y = df["category"]

X = X.values
y = y.values

scaler = StandardScaler()

X = scaler.fit_transform(X)

# Define the LogisticRegression classifier
clf = LogisticRegression(max_iter=1000)

# Define the number of folds
num_folds = 5

# Define the k-fold cross-validation object
kfold = KFold(n_splits=num_folds)

# Perform the k-fold cross-validation
scores = cross_val_score(clf, X, y, cv=kfold)

# Print the accuracy scores for each fold
for i, score in enumerate(scores):
    print("Fold {}: {}".format(i + 1, score))

# Calculate the mean and standard deviation of the scores
mean_score = scores.mean()

# Print the mean and standard deviation
print("Mean Accuracy: {:.2f}%".format(mean_score * 100))

df_test = pd.read_csv("test.csv")
df_test = df_test.drop(["ID"], axis=1)
X_test = df_test.values

clf.fit(X, y)
pred = clf.predict(X_test)


# print(f"Accuracy: {accuracy_score(y, pred)}")
print(pred)
