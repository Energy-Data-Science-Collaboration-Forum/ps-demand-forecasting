import joblib
import logging
from datetime import datetime as dt
import pandas as pd

from src.prepare_data import (
    prepare_gas_demand_actuals,
    prepare_electricity_features,
)
from src.train import train_glm_63, train_gam
from src.evaluate import evaluate_models

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)

FORMAT = "%Y%m%d_%H%M%S"

FEATURES = {
    "TED": "data/elexon_ted_forecast.csv",
    "WIND": "data/elexon_wind_forecast.csv",
    "ACTUAL_D_SOFAR_ALL_BUT_WIND_GT": "data/elexon_electricity_actuals.csv",
    "ELECTRICITY_ACTUALS": "data/elexon_electricity_actuals.csv",
}
ACTUALS = {"GAS": "data/gas_actuals.csv"}

logger.info("Preprocessing actual gas demand")
gas_demand_actuals = prepare_gas_demand_actuals(ACTUALS["GAS"])
ps_demand_actuals = gas_demand_actuals[["PS"]]

logger.info("Preparing features")
electricity_features = prepare_electricity_features(FEATURES)

ps_63_model, ps_63_cv_predictions = train_glm_63(
    ps_demand_actuals, electricity_features
)
joblib.dump(ps_63_model, "data/ps_63_model.joblib")

ps_gam_model, ps_gam_predictions = train_gam(ps_demand_actuals, electricity_features)
joblib.dump(ps_gam_model, "data/ps_gam_model.joblib")

all_predictions = pd.concat(
    [
        ps_63_cv_predictions.to_frame(),
        ps_gam_predictions
    ],
    axis=1,
)

model_performance = evaluate_models(all_predictions, ps_demand_actuals)
model_performance.to_csv(
    "data/model_performance.csv", index=False
)

print(model_performance)
