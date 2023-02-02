import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from src.evaluate import evaluate_models


def test_evaluate_models_one_model():
    mock_predictions = pd.DataFrame(
        {"GLM":[1, 2, 3, 4, 5, 6]}, index=pd.date_range("2022-08-01", "2022-08-06")
    )
    actuals = pd.DataFrame(
        {"PS": np.arange(7,1, step=-1)}, index=pd.date_range("2022-08-01", "2022-08-06")
    )

    result = evaluate_models(mock_predictions, actuals)
    desired_result = pd.DataFrame(
        {"MODEL": ["GLM"], "MAE": [3.0], "MAPE": [0.765079]}
    )
    assert_frame_equal(result, desired_result)


def test_evaluate_models_two_models():
    mock_predictions = pd.DataFrame(
        {"GLM":[1, 2, 3, 4, 5, 6], "DIFF":[4, 5, 3, 1, 8, 6]}, index=pd.date_range("2022-08-01", "2022-08-06")
    )
    actuals = pd.DataFrame(
        {"PS": np.arange(7,1, step=-1)}, index=pd.date_range("2022-08-01", "2022-08-06")
    )

    result = evaluate_models(mock_predictions, actuals)
    desired_result = pd.DataFrame(
        {"MODEL": ["DIFF", "GLM"], "MAE": [3.0, 3.0], "MAPE": [0.901984, 0.765079]}
    )
    assert_frame_equal(result, desired_result)
