import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal

import src.train
from src.train import train_glm_63


def test_train_glm():
    target = pd.DataFrame(
        {
            "PS": [
                52.16314,
                53.94938,
                61.38836,
                59.02986,
                53.926,
                49.71774,
                47.03101,
                49.43077,
                53.47305,
                54.79746,
                54.57056,
                51.4812,
                51.66534,
                52.27608,
                46.27135,
                48.5113,
                56.47527,
                57.09211,
                61.32952,
                63.85277,
                58.93696,
            ]
        },
        index=pd.DatetimeIndex(
            pd.date_range("2022-06-11", "2022-07-01"), name="GAS_DAY"
        ),
    )

    features = pd.DataFrame(
        {
            "TED_DA_FORECAST": [
                9.01,
                3.91,
                6.54,
                6.0,
                15.09,
                4.73,
                8.65,
                5.96,
                11.03,
                3.14,
                12.26,
                7.03,
                11.62,
                6.24,
                7.13,
                4.16,
                9.14,
                9.35,
                4.16,
                9.14,
                9.35,
            ],
            "WIND_FORECAST": [
                9.01,
                3.91,
                6.54,
                6.0,
                15.09,
                4.73,
                8.65,
                5.96,
                11.03,
                3.14,
                12.26,
                7.03,
                11.62,
                6.24,
                7.13,
                4.16,
                9.14,
                9.35,
                4.16,
                9.14,
                9.35,
            ],
            "ACTUAL_D_SOFAR_ALL_BUT_WIND_AND_GT": [
                9.01,
                3.91,
                6.54,
                6.0,
                15.09,
                4.73,
                8.65,
                5.96,
                11.03,
                3.14,
                12.26,
                7.03,
                11.62,
                6.24,
                7.13,
                4.16,
                9.14,
                9.35,
                4.16,
                9.14,
                9.35,
            ],
        },
        index=pd.DatetimeIndex(
            pd.date_range("2022-06-11", "2022-07-01"), name="GAS_DAY"
        ),
    )

    _, result = train_glm_63(target, features)
    desired_result = pd.Series(
        [
            53.63526375,
            53.37458276,
            53.68013042,
            53.62958443,
            53.79826036,
            52.74959288,
            52.75708101,
            52.28189509,
            52.74959288,
            52.75708101,
        ],
        index=pd.DatetimeIndex(
            pd.date_range("2022-06-22", "2022-07-01"), name="GAS_DAY"
        ),
        name="GLM_63",
    )
    assert_series_equal(result, desired_result)
