import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


def read_data(train_path, test_path):
    # read train and test data from local path
    # return the form need for models
    train = pd.read_csv(train_path)
    X_train = train.drop(["category", "ID"], axis=1)
    y_train = train["category"]
    X_test = pd.read_csv(test_path)
    X_test = X_test.drop(["ID"], axis=1)
    # print(X_test)
    return X_train, y_train, X_test


def gscv(X_train, y_train):
    # using Grid Search Cross Validation to tune hyper-parameters of kNN model on training data
    # return the tuned best model
    parameters = {
        "n_estimators": [10, 50, 100, 1000, 1500, 2000],
        "max_depth": [None, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        "min_samples_split": [2, 5],
    }
    model_rfc = RandomForestClassifier()
    CV = KFold(n_splits=10, shuffle=True, random_state=0)

    gscv = GridSearchCV(model_rfc, parameters, cv=CV)
    gscv.fit(X_train, y_train)
    return gscv.best_estimator_


def KFoldCV(X_train, y_train, K=10, random_seed=0):
    cv = KFold(n_splits=K, shuffle=True, random_state=random_seed)
    #   acc is mean accuracy of 10 fold cross validation
    acc = 0
    rfcBest = gscv(X_train, y_train)

    for train_index, val_index in cv.split(X_train):
        x_train, x_val = X_train.iloc[train_index], X_train.iloc[val_index]
        Y_train, Y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        rfcBest.fit(x_train, Y_train)
        acc += rfcBest.score(x_val, Y_val)
    acc /= K
    return acc


# def kNN(X_train, y_train, X_test):
#     # tune k Nearest Neighbors model parameters with gscv
#     # return prediction on test data
#     rfcBest = gscv(X_train, y_train)
#     rfcBest.fit(X_train, y_train)
#     prediction = rfcBest.predict(X_test)
#     return prediction


def randomForest(X_train, y_train, X_test):
    # tune k Nearest Neighbors model parameters with gscv
    # return prediction on test data
    rfcBest = gscv(X_train, y_train)
    rfcBest.fit(X_train, y_train)
    prediction = rfcBest.predict(X_test)
    return prediction


if __name__ == "__main__":
    X_train, y_train, X_test = read_data("train.csv", "test.csv")
    # acc = KFoldCV(X_train, y_train, K=10, random_seed=0)
    # print("10 fold cross validation mean accuracy on training data: ", acc)
    prediction = randomForest(X_train, y_train, X_test)
    print("Prediction on test data:")
    print(prediction)
    final_df = pd.DataFrame(prediction)
    final_df.to_csv("knn.csv")
