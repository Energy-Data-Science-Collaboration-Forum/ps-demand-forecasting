import joblib
import logging
from datetime import datetime as dt
import pandas as pd

from src.prepare_data import (
    prepare_gas_demand_actuals,
    prepare_electricity_features,
)
from src.train import train_glm_63, train_gam, train_glm_63_f, train_huber, train_ransac
from src.evaluate import evaluate_models

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)

FORMAT = "%Y%m%d_%H%M%S"

FEATURES = {
    "TED": "data/elexon_ted_forecast_sample.csv",
    "WIND": "data/elexon_wind_forecast_sample.csv",
    "ACTUAL_D_SOFAR_ALL_BUT_WIND_GT": "data/elexon_electricity_actuals_sample.csv",
    "ELECTRICITY_ACTUALS": "data/elexon_electricity_actuals_sample.csv",
}
ACTUALS = {"GAS": "data/gas_actuals_sample.csv"}

logger.info("Preprocessing actual gas demand")
gas_demand_actuals = prepare_gas_demand_actuals(ACTUALS["GAS"])
ps_demand_actuals = gas_demand_actuals[["PS"]]

logger.info("Preparing features")
electricity_features = prepare_electricity_features(FEATURES)


# This code trains a generalized linear model (GLM) with 3 features.
# The model is trained on the `electricity_features` dataset,
# and the predictions are saved to the `data/ps_model_<timestamp>.joblib` file.
ps_63_model, ps_63_cv_predictions = train_glm_63(
    ps_demand_actuals, electricity_features
)
joblib.dump(ps_63_model, "data/ps_63_model.joblib")


# This code trains a generalized additive model (GAM) to predict gas demand.
# The model is trained on the `electricity_features` dataset,
# and the predictions are saved to the `data/ps_model_<timestamp>.joblib` file.
ps_gam_model, ps_gam_predictions = train_gam(ps_demand_actuals, electricity_features)
joblib.dump(ps_gam_model, "data/ps_gam_model.joblib")


# This code trains a GLM with original features and added features.
# The added features are the day of the week, the hour of the day, and the month of the year.
# The model is trained on the `electricity_features` dataset,
# and the predictions are saved to the `data/ps_model_<timestamp>.joblib` file.
ps_63_f_model, ps_63_f_cv_predictions = train_glm_63_f(
    ps_demand_actuals, electricity_features
)
joblib.dump(ps_63_f_model, f"data/ps_model_{dt.now().strftime(format=FORMAT)}.joblib")


# This code trains a RANSAC model.
# The RANSAC model is a robust regression model that is also not sensitive to outliers.
# The model is trained on the `electricity_features` dataset,
# and the predictions are saved to the `data/ps_model_<timestamp>.joblib` file.
ps_ransac_model, ps_ransac_cv_predictions = train_ransac(
    ps_demand_actuals, electricity_features
)
joblib.dump(ps_ransac_model, f"data/ps_model_{dt.now().strftime(format=FORMAT)}.joblib")


# This code trains a Huber model.
# The Huber model is a robust regression model that is more sensitive to outliers than the Theil-Sen or RANSAC models.
# The model is trained on the `electricity_features` dataset,
# and the predictions are saved to the `data/ps_model_<timestamp>.joblib` file.
ps_huber_model, ps_huber_cv_predictions = train_huber(
    ps_demand_actuals, electricity_features
)
joblib.dump(ps_huber_model, f"data/ps_model_{dt.now().strftime(format=FORMAT)}.joblib")


# This code creates an ensemble model by averaging the predictions of the GLM_63_Added_Ftrs and Huber models.
ps_ensemble_63f_huber_cv_predictions = (0.5*ps_63_f_cv_predictions) + (0.5*ps_huber_cv_predictions)
ps_ensemble_63f_huber_cv_predictions.name = "Ensemble:GLM_63 & Huber Models-34 Features"


# This code adds the ensemble model predictions to the dataframe of all predictions.
all_predictions = pd.concat(
    [
        ps_63_cv_predictions.to_frame(),
        ps_gam_predictions,
        ps_63_f_cv_predictions.to_frame(),
        ps_huber_cv_predictions.to_frame(),
        ps_ransac_cv_predictions.to_frame(),
        ps_ensemble_63f_huber_cv_predictions.to_frame()        
    ],
    axis=1,
)

# Assuming evaluate_models function, all_predictions, ps_demand_actuals are defined elsewhere.
# Evaluate the models' performance
# The MAE is the average difference between the predicted and actual values.
# The MAPE is the average percentage error between the predicted and actual values.
model_performance = evaluate_models(all_predictions, ps_demand_actuals)

# Identify the MAE of the GLM_63 NGT Model for reference. Adjust the model name as per the actual naming convention if needed
reference_mae = model_performance.loc[model_performance['MODEL'] == 'GLM_63 NGT Model with 3 Features - Baseline Model', 'MAE'].values[0]

# Calculate MAE Deviation Percentage
model_performance['MAE_Deviation %'] = ((model_performance['MAE'] - reference_mae) / reference_mae) * 100

# This code sorts the models by MAE.
model_performance = model_performance.sort_values(by='MAE')

# The models are then ranked by their MAE and MAPE scores.
model_performance['MAE Ranking'] = model_performance['MAE'].rank(method='min').astype(int)
model_performance['MAPE Ranking'] = model_performance['MAPE'].rank(method='min').astype(int)

# Save the model performance results to a CSV file
filename = f"data/model_performance_{dt.now().strftime(format=FORMAT)}.csv"
model_performance.to_csv(filename, index=False)


# This code prints the model performance results.
print(model_performance)
