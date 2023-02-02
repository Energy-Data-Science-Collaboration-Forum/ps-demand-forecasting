import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


def train_glm_63(target, features):
    """Train a Linear Regression model based on CWV

    Args:
        target (pandas DataFrame): A DataFrame with a column named LDZ for the gas demand actuals
        features (pandas DataFrame): A DataFrame with a column named CWV for the CWV forecast

    Returns:
        pandas Series: A Series with the predictions from the linear model, named GLM_CWV
    """
    logger.info("Training linear model with TED forecast, Wind forecast and Actual within-day so far feature")
    X = features[
        ["TED_DA_FORECAST", "WIND_FORECAST", "ACTUAL_D_SOFAR_ALL_BUT_WIND_AND_GT"]
    ].dropna()

    X, y = check_overlapping_dates(target, X)

    model = LinearRegression()
    predictions = tss_cross_val_predict(X, y, model)
    predictions.name = "GLM_63"

    model = LinearRegression()
    model.fit(X, y)

    return model, predictions


def tss_cross_val_predict(X, y, model, min_train=7):
    """Apply a form of Time Series cross validation with the given data and for the given model
    We expand the data by a week in each fold and retrain the model to generate predictions for the next week.

    Args:
        X (pandas DataFrame): A DataFrame with features
        y (pandas DataFrame): A DataFrame with the target
        model (a sklearn Model): A model object with a fit and predict function
        min_train (int, optional): Number of historical values necessary to start the training cadence. Defaults to 7.

    Returns:
        pandas Series: A Series with the predictions from all the folds
    """
    test_predictions = []
    # weekly window
    nsplits = abs(round((X.index.min() - X.index.max()).days / 7))
    tscv = TimeSeriesSplit(n_splits=nsplits)

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
    test_predictions = pd.Series(
        test_predictions, index=X.index[num_samples_train_first_iteration:]
    )

    return test_predictions


def check_overlapping_dates(dataset_one, dataset_two):
    """Determine the overlapping dates from the given datasets and filter them both by it

    Args:
        dataset_one (pandas DataFrame): A DataFrame with dates on the index
        dataset_two (pandas DataFrame): A DataFrame with dates on the index

    Returns:
        tuple: A tuple of DataFrame, in reverse order from the input (just to confuse you)
    """
    overlapping_dates = dataset_one.index.intersection(dataset_two.index)
    d1 = dataset_one.loc[overlapping_dates].copy()
    d2 = dataset_two.loc[overlapping_dates].copy()

    return d2, d1
