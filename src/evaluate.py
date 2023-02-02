import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


def evaluate_models(predictions, actuals):
    combined = pd.concat([predictions, actuals["PS"]], axis=1)
    result = (
        combined.dropna()
        .melt(
            id_vars=["PS"],
            value_vars=predictions.columns,
            value_name="PREDICTIONS",
            var_name="MODEL",
        )
        .groupby("MODEL")
        .apply(lambda df: calculate_metrics(df))
        .reset_index()        
    )

    return result


def calculate_metrics(input_data):
    mae = mean_absolute_error(input_data["PS"], input_data["PREDICTIONS"])
    mape = mean_absolute_percentage_error(input_data["PS"], input_data["PREDICTIONS"])

    return pd.Series({"MAE":mae, "MAPE":mape})