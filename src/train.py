import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from prophet import Prophet

from src.utils import fix_missing_values

logger = logging.getLogger(__name__)

PROPHET_FEATURES = [
    "DEMAND_42_NEXT_DAY",
    "DEMAND_15_NEXT_DAY",
    "WIND_FORECAST_3_NEXT_DAY",
    "WIND_FORECAST_29_NEXT_DAY",
    "DEMAND_19_NEXT_DAY",
    "DEMAND_9_NEXT_DAY",
    "WIND_FORECAST_9_NEXT_DAY",
    "DEMAND_45_NEXT_DAY",
    "WIND_FORECAST_21_NEXT_DAY",
    "POWER_STATION_48_PREVIOUS_DAY",
    "PCA4",
    "WIND_FORECAST_37_NEXT_DAY",
    "TED_DA_FORECAST",
    "DEMAND_13_NEXT_DAY",
    "DEMAND_17_NEXT_DAY",
    "DEMAND_12_NEXT_DAY",
    "PCA5",
    "DEMAND_10_NEXT_DAY",
    "DEMAND_44_NEXT_DAY",
    "WIND_FORECAST_23_NEXT_DAY",
    "PCA10",
    "WIND_FORECAST_11_NEXT_DAY",
    "DEMAND_43_NEXT_DAY",
    "DEMAND_16_NEXT_DAY",
    "DEMAND_46_NEXT_DAY",
    "WIND_FORECAST_47_NEXT_DAY",
    "WIND_FORECAST_35_NEXT_DAY",
    "DEMAND_43_CURRENT_DAY",
    "WIND_FORECAST_25_NEXT_DAY",
    "WIND_FORECAST_31_NEXT_DAY",
    "DEMAND_32_NEXT_DAY",
    "WIND_FORECAST_33_NEXT_DAY",
]

PROPHET_BASE_FEATS = ["WIND", "DEMAND", "TED_DA_FORECAST", "INTERCONNECTOR", "POWERSTATION", "REST"]

def train_gam(target, features, min_train=30):
    logger.info("Training prophet model with lots of features")
        
    # select features on which to base PCA
    X = select_features(features)
    preprocessor = get_preprocessor()

    # fit on training data
    proph_data = pd.concat([X, target], axis=1).rename({"PS": "y"}, axis=1)
    proph_data = fix_missing_values(proph_data)
    proph_data.index.name = "ds"
    cols = X.columns.tolist() + [f"PCA{i}" for i in range(1, 14)]

    test_predictions = []
    nsplits = abs(round((proph_data.index.min() - proph_data.index.max()).days))
    tscv = TimeSeriesSplit(n_splits=nsplits)

    for train_index, test_index in tscv.split(proph_data):

        if len(train_index) < min_train:
            continue
    
        X_train = proph_data.drop(columns="y").iloc[train_index]
        y_train = proph_data[["y"]].iloc[train_index]
        X_test = proph_data.drop(columns="y").iloc[test_index]

        X_train = pd.DataFrame(preprocessor.fit_transform(X_train), columns=cols, index=X_train.index)
        X_test = pd.DataFrame(preprocessor.transform(X_test), columns=cols, index=X_test.index)

        prophmodel = get_prophet_model()
        model = prophmodel.fit(pd.concat([X_train, y_train], axis=1).reset_index())
        
        test_predictions.append(model.predict(X_test.reset_index()))
    
    test_predictions = pd.concat(test_predictions)[["ds", "yhat"]]
    result = test_predictions.rename(columns={"ds":"GAS_DAY", "yhat":"PS_GAM"}).set_index("GAS_DAY")

    prophmodel = get_prophet_model()
    X_train = pd.DataFrame(preprocessor.fit_transform(proph_data.drop(columns="y")), columns=cols)
    X_train["y"] = proph_data["y"].values
    X_train["ds"] = proph_data.index.values
    model = prophmodel.fit(X_train)

    return model, result


def get_prophet_model():
    result = Prophet(daily_seasonality=False, yearly_seasonality=False)
    result.add_country_holidays(country_name="UK")

    for regressor in PROPHET_FEATURES:
        result.add_regressor(regressor)

    return result


def get_preprocessor():
    pipe = make_pipeline(
        StandardScaler(),
        FeatureUnion(
            transformer_list=[
                ('pca', PCA(n_components=13)),
                ('identity', 'passthrough')
            ]
        ),
        StandardScaler()
    )
    return pipe


def select_features(input_data):
    base_features = ["WIND_FORECAST", "INTERCONNECTORS", "REST", "POWER_STATION", "WIND", "DEMAND", "DEMAND"]
    suffix = ["NEXT_DAY", "PREVIOUS_DAY", "PREVIOUS_DAY", "PREVIOUS_DAY","PREVIOUS_DAY", "NEXT_DAY", "CURRENT_DAY"]
    string_mask = [f"{a}_\d*_{b}" for a, b in zip(base_features, suffix)] # i.e. WIND_FORECAST_\d*_NEXT_DAY
    
    mask = input_data.columns.str.contains("|".join(string_mask))
    selected_columns = input_data.columns[mask].tolist()
    result = input_data[selected_columns + ["TED_DA_FORECAST"]].copy()
    return result


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
