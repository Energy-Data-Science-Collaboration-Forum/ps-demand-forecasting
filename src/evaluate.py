import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


def evaluate_models(ps_predictions, actuals):
    ps_predictions.name = "PS_PREDICTIONS"
    combined = pd.concat([ps_predictions, actuals], axis=1)

    combined = combined.dropna()

    result = pd.DataFrame(
        {
            "MODEL": ["PS"],
            "MAE": [
                mean_absolute_error(combined["PS"], combined["PS_PREDICTIONS"]),
            ],
            "MAPE": [
                mean_absolute_percentage_error(
                    combined["PS"], combined["PS_PREDICTIONS"]
                ),
            ],
        }
    )

    return result
