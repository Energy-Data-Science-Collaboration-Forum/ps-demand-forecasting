import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit


def train(target, features):

    overlapping_dates = target.index.intersection(features.index)
    y = target.loc[overlapping_dates].copy()
    X = features.loc[overlapping_dates].copy()

    predictions = tss_cross_val_predict(X, y)
    model = train_full_model(X, y)

    return model, predictions


def tss_cross_val_predict(X, y, min_train=7):

    test_predictions = []
    # weekly window
    nsplits = abs(round((X.index.min() - X.index.max()).days / 7))
    tscv = TimeSeriesSplit(n_splits=nsplits)
    model = LinearRegression()

    for train_index, test_index in tscv.split(X):

        if len(train_index) < min_train:
            continue

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train = y.iloc[train_index]

        model.fit(X_train, y_train)

        test_predictions.append(model.predict(X_test))

    test_predictions = np.array(test_predictions).flatten()
    # samples from the first training iteration don't have any test predictions so should be discarded
    num_samples_train_first_iteration = X.shape[0] - test_predictions.shape[0]
    test_predictions = pd.Series(test_predictions, index=X.index[num_samples_train_first_iteration:])

    return test_predictions


def train_full_model(X, y):
    model = LinearRegression()
    model.fit(X, y)

    return model
