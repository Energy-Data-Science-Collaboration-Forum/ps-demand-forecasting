import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RANSACRegressor, HuberRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer, mean_absolute_error
from prophet import Prophet
from tqdm import tqdm

from src.utils import fix_missing_values

logger = logging.getLogger(__name__)

# 31 Prophet Features
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
    logger.info("Training prophet model with 31 features")
        
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
    result = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
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
        ["TED_DA_FORECAST", 
         "WIND_FORECAST", 
         "ACTUAL_D_SOFAR_ALL_BUT_WIND_AND_GT"
        ]
    ].dropna()

    X, y = check_overlapping_dates(target, X)

    model = LinearRegression()
    predictions = tss_cross_val_predict(X, y, model)
    predictions.name = "GLM_63 NGT Model with 3 Features - Baseline Model"
    model.fit(X, y)

    return model, predictions


def train_glm_63_f(target, features):
    """Train a Linear Regression model based on 34 Features

    Args:
        target (pandas DataFrame): A DataFrame with a column named LDZ for the gas demand actuals
        features (pandas DataFrame): A DataFrame with a column named CWV for the CWV forecast

    Returns:
        tuple: A tuple containing the best Linear Regression model and its predictions
    """

    logger.info("Training linear model with 34 Features")


    # Define the feature matrix
    X = features[
        [
            "TED_DA_FORECAST",
            "ACTUAL_D_SOFAR_ALL_BUT_WIND_AND_GT",
            "POWER_STATION_1_PREVIOUS_DAY",
            "POWER_STATION_16_PREVIOUS_DAY",          
            "POWER_STATION_33_PREVIOUS_DAY",
            "POWER_STATION_38_PREVIOUS_DAY",
            "POWER_STATION_43_PREVIOUS_DAY",
            "POWER_STATION_44_PREVIOUS_DAY",  
            "POWER_STATION_47_PREVIOUS_DAY",  
            "POWER_STATION_48_PREVIOUS_DAY",                    
            "DEMAND_9_NEXT_DAY",
            "DEMAND_10_NEXT_DAY",
            "DEMAND_19_NEXT_DAY",
            "DEMAND_38_NEXT_DAY",
            "DEMAND_43_NEXT_DAY",  
            "DEMAND_45_NEXT_DAY",          
            "DEMAND_46_NEXT_DAY",            
            "DEMAND_48_NEXT_DAY",
            "WIND_FORECAST",
            "WIND_11_PREVIOUS_DAY",
            "REST_1_PREVIOUS_DAY",            
            "REST_7_PREVIOUS_DAY",
            "REST_18_PREVIOUS_DAY",
            "REST_37_PREVIOUS_DAY",
            "INTERCONNECTORS_11_PREVIOUS_DAY",
            "INTERCONNECTORS_25_PREVIOUS_DAY",
            "INTERCONNECTORS_46_PREVIOUS_DAY",             
            "INTERCONNECTORS_47_PREVIOUS_DAY",           
            "INTERCONNECTORS_48_PREVIOUS_DAY",
            "DEMAND_11_CURRENT_DAY",
            "DEMAND_19_CURRENT_DAY",
            "DEMAND_38_CURRENT_DAY",            
            "DEMAND_47_CURRENT_DAY",
            "DEMAND_48_CURRENT_DAY"            
        ]
    ].dropna()

    # Check for overlapping dates between X and target
    X, y = check_overlapping_dates(target, X)

    # Create a LinearRegression model
    model = LinearRegression()

    # Make predictions using cross-validation
    predictions = tss_cross_val_predict(X, y, model)

    # Set the name of the predictions
    predictions.name = "GLM_63 Model with 34 Features"

    # Fit the model to the data
    model.fit(X, y)

    return model, predictions


def train_huber(target, features):
    """Train an improved Huber regression model based on CWV.

    This function trains a Huber regression model on the given data using various improvement techniques such as hyperparameter tuning, feature selection, and regularization. The model is trained using the following hyperparameters:

    * epsilon: The threshold at which the Huber loss function switches from quadratic to linear.
    * alpha: The regularization strength.
    * max_iter: The maximum number of iterations to run the solver.
    * tol: The tolerance for the stopping criteria.

    The predictions of the model are also returned.

    Args:
        target (pandas DataFrame): A DataFrame with a column named LDZ for the gas demand actuals
        features (pandas DataFrame): A DataFrame with a column named CWV for the CWV forecast

    Returns:
        tuple: A tuple containing the best HuberRegressor model and its predictions
    """

    # Define the feature matrix
    X = features[
        [
            "TED_DA_FORECAST",
            "ACTUAL_D_SOFAR_ALL_BUT_WIND_AND_GT",
            "POWER_STATION_1_PREVIOUS_DAY",
            "POWER_STATION_16_PREVIOUS_DAY",          
            "POWER_STATION_33_PREVIOUS_DAY",
            "POWER_STATION_38_PREVIOUS_DAY",
            "POWER_STATION_43_PREVIOUS_DAY",
            "POWER_STATION_44_PREVIOUS_DAY",  
            "POWER_STATION_47_PREVIOUS_DAY",  
            "POWER_STATION_48_PREVIOUS_DAY",                    
            "DEMAND_9_NEXT_DAY",
            "DEMAND_10_NEXT_DAY",
            "DEMAND_19_NEXT_DAY",
            "DEMAND_38_NEXT_DAY",
            "DEMAND_43_NEXT_DAY",  
            "DEMAND_45_NEXT_DAY",          
            "DEMAND_46_NEXT_DAY",            
            "DEMAND_48_NEXT_DAY",
            "WIND_FORECAST",
            "WIND_11_PREVIOUS_DAY",
            "REST_1_PREVIOUS_DAY",            
            "REST_7_PREVIOUS_DAY",
            "REST_18_PREVIOUS_DAY",
            "REST_37_PREVIOUS_DAY",
            "INTERCONNECTORS_11_PREVIOUS_DAY",
            "INTERCONNECTORS_25_PREVIOUS_DAY",
            "INTERCONNECTORS_46_PREVIOUS_DAY",             
            "INTERCONNECTORS_47_PREVIOUS_DAY",           
            "INTERCONNECTORS_48_PREVIOUS_DAY",
            "DEMAND_11_CURRENT_DAY",
            "DEMAND_19_CURRENT_DAY",
            "DEMAND_38_CURRENT_DAY",            
            "DEMAND_47_CURRENT_DAY",
            "DEMAND_48_CURRENT_DAY" 
        ]
    ].dropna()

    # Check for overlapping dates between X and target
    X, y = check_overlapping_dates(target, X)

    logger.info("Training an improved Huber regression model with 34 Feartures")

    # Adaptive grid search ranges based on previous insights.
    param_grid = {
        'epsilon': np.arange(1.2, 2.1, 0.2),  # More refined around promising regions
        'alpha': np.logspace(-4, -1, 4),  # Expanding to explore lower and higher alpha values
        'max_iter': [1000, 2000],  # Reduced variation after identifying an adequate range
        'tol': np.logspace(-5, -3, 3)  # Explored more precise tolerance levels
    }

    # Time series cross validation
    cv_method = TimeSeriesSplit(n_splits=5)
    # Using MAE as the main determinant, direct use within GridSearchCV
    scoring_metric = make_scorer(mean_absolute_error, greater_is_better=False)

    # Initialize the GridSearchCV object
    grid_search = GridSearchCV(
        estimator=HuberRegressor(),
        param_grid=param_grid,
        n_jobs=-1,
        cv=cv_method,  # Reduce the cv parameter as we are now iterating over TimeSeriesSplit object
        scoring=scoring_metric,
    )
    # Fit the grid search to find the best model
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    logger.info(f"Best Hyperparameters: {best_params}")

    # Extracting best model for final training
    best_model.fit(X, y)
    predictions = best_model.predict(X)

    # Assigning name to predictions for clarity
    predictions = pd.Series(predictions, index= X.index, name="Huber Model with 34 Features")  

    return best_model, predictions


def train_ransac(target, features):
    """
    Train an improved RANSAC regression model.
    This function trains a RANSAC regression model on the given data using 34 Features  
    The model's hyperparameters are fine-tuned for better performance.

    Args:
        target (pandas DataFrame): A DataFrame with the column LDZ for the gas demand actuals.
        features (pandas DataFrame): A DataFrame with 34 feature columns.

    Returns:
        tuple: A tuple containing the improved RANSACRegressor model and its predictions.
    """
    
    # Define the feature matrix
    X = features[
        [
            "TED_DA_FORECAST",
            "ACTUAL_D_SOFAR_ALL_BUT_WIND_AND_GT",
            "POWER_STATION_1_PREVIOUS_DAY",
            "POWER_STATION_16_PREVIOUS_DAY",          
            "POWER_STATION_33_PREVIOUS_DAY",
            "POWER_STATION_38_PREVIOUS_DAY",
            "POWER_STATION_43_PREVIOUS_DAY",
            "POWER_STATION_44_PREVIOUS_DAY",  
            "POWER_STATION_47_PREVIOUS_DAY",  
            "POWER_STATION_48_PREVIOUS_DAY",                    
            "DEMAND_9_NEXT_DAY",
            "DEMAND_10_NEXT_DAY",
            "DEMAND_19_NEXT_DAY",
            "DEMAND_38_NEXT_DAY",
            "DEMAND_43_NEXT_DAY",  
            "DEMAND_45_NEXT_DAY",          
            "DEMAND_46_NEXT_DAY",            
            "DEMAND_48_NEXT_DAY",
            "WIND_FORECAST",
            "WIND_11_PREVIOUS_DAY",
            "REST_1_PREVIOUS_DAY",            
            "REST_7_PREVIOUS_DAY",
            "REST_18_PREVIOUS_DAY",
            "REST_37_PREVIOUS_DAY",
            "INTERCONNECTORS_11_PREVIOUS_DAY",
            "INTERCONNECTORS_25_PREVIOUS_DAY",
            "INTERCONNECTORS_46_PREVIOUS_DAY",             
            "INTERCONNECTORS_47_PREVIOUS_DAY",           
            "INTERCONNECTORS_48_PREVIOUS_DAY",
            "DEMAND_11_CURRENT_DAY",
            "DEMAND_19_CURRENT_DAY",
            "DEMAND_38_CURRENT_DAY",            
            "DEMAND_47_CURRENT_DAY",
            "DEMAND_48_CURRENT_DAY" 
        ]
    ].dropna()

    logger.info("Improving RANSAC regression model training")   
  
    # Check for overlapping dates between X and target
    X, y = check_overlapping_dates(target, X)

    # Predefined hyperparameters for the RANSAC algorithm
    param_grid = {
        'min_samples': [0.2, 0.5, 0.8],  # Fraction of total number of samples
        'max_trials': [100, 200, 500],  # Adjusted range for max_trials
        'residual_threshold': [1.0, 2.0, 3.0],  # Optimized residual_threshold values
        'loss': ['absolute_error', 'squared_error'],  # Including different loss functions
    }

    # Time series cross-validation method
    cv_method = TimeSeriesSplit(n_splits=5)
    
    # Define the mean absolute error scorer
    scoring_metric = make_scorer(mean_absolute_error, greater_is_better=False)

    # Initialize and perform the grid search
    grid_search = GridSearchCV(
        estimator=RANSACRegressor(),
        param_grid=param_grid,
        n_jobs=-1,
        cv=cv_method,
        scoring=scoring_metric,
    )
    
    # Fit the GridSearchCV object to the data
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    logger.info(f"Best Hyperparameters for RANSAC: {best_params}")


    # Final model training with best parameters
    best_model.fit(X, y)

    # Make predictions using cross-validation
    predictions = tss_cross_val_predict(X, y, best_model)

    # Set the name of the predictions
    predictions.name = "RANSAC Model with 34 Features"

    return best_model, predictions

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
    # daily window
    nsplits = abs(round((X.index.min() - X.index.max()).days / 2))
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
