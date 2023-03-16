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

ps_63_model, ps_63_cv_predictions = train_glm_63(
    ps_demand_actuals, electricity_features
)
joblib.dump(ps_63_model, f"data/ps_model_{dt.now().strftime(format=FORMAT)}.joblib")

ps_gam_model, ps_gam_predictions = train_gam(ps_demand_actuals, electricity_features)
joblib.dump(ps_gam_model, f"data/ps_model_{dt.now().strftime(format=FORMAT)}.joblib")

all_predictions = pd.concat(
    [
        ps_63_cv_predictions.to_frame(),
        ps_gam_predictions
    ],
    axis=1,
)

model_performance = evaluate_models(all_predictions, ps_demand_actuals)
model_performance.to_csv(
    f"data/model_performance_{dt.now().strftime(format=FORMAT)}.csv", index=False
)

print(model_performance)
