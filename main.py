import joblib
from datetime import datetime as dt
from src.prepare_data import (
    prepare_gas_demand_actuals,
    prepare_electricity_features,
)
from src.train import train
from src.evaluate import evaluate_models

FORMAT = "%Y%m%d_%H%M%S"

FEATURES = {"TED":"data/elexon_ted_forecast_20221125_112613.csv", 
"WIND":"data/elexon_wind_forecast_20221125_112621.csv", "ACTUAL_D_SOFAR_ALL_BUT_WIND_GT":"data/elexon_electricity_actuals_20221122_221240.csv"}
ACTUALS = {"GAS": "data/gas_actuals_20221118_214136.csv"}

gas_demand_actuals = prepare_gas_demand_actuals(ACTUALS["GAS"])

electricity_features = prepare_electricity_features(FEATURES)

ps_model, ps_cv_predictions = train(gas_demand_actuals[["PS"]], electricity_features)
joblib.dump(ps_model, f"data/ps_model_{dt.now().strftime(format=FORMAT)}.joblib")

model_performance = evaluate_models(
    ps_cv_predictions, gas_demand_actuals
)
model_performance.to_csv(
    f"data/model_performance_{dt.now().strftime(format=FORMAT)}.csv", index=False
)

print(model_performance)
