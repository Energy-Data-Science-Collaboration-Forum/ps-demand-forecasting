import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import src.prepare_data
from src.prepare_data import (
    prepare_gas_demand_actuals,
    prepare_ted_forecast,
    prepare_wind_forecast,
    prepare_electricity_actuals,
    prepare_actual_sofar,
    prepare_gen_previous_gas_day,
    aggregate_generation_data,
    prepare_hourly_wind_forecast,
    prepare_ted_half_hourly_forecast,
)


def test_prepare_gas_demand_actuals(monkeypatch):
    mock_data = pd.DataFrame(
        {
            "ApplicableFor": [pd.to_datetime("2022-01-10")] * 5,
            "Value": [1.0] * 5,
            "TYPE": [
                "NTS Volume Offtaken, Industrial Offtake Total",
                "NTS Volume Offtaken, Interconnector Exports Total",
                "NTS Volume Offtaken, LDZ Offtake Total",
                "NTS Volume Offtaken, Powerstations Total",
                "NTS Volume Offtaken, Storage Injection Total",
            ],
        }
    )

    def mock_read_csv(fp, parse_dates):
        return mock_data

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)

    result = prepare_gas_demand_actuals("")

    desired_result = pd.DataFrame(
        {
            "INDUSTRIAL": [1.0],
            "INTERCONNECTOR": [1.0],
            "LDZ": [1.0],
            "PS": [1.0],
            "STORAGE": [1.0],
        },
        index=pd.DatetimeIndex([pd.to_datetime("2022-01-10")], name="GAS_DAY"),
    )

    assert_frame_equal(result, desired_result)


def test_prepare_ted_forecast(monkeypatch):
    date_format = "%Y-%m-%dT%H:%M:%SZ"
    dates = pd.date_range("2021-11-11", "2021-11-12", freq="30min", inclusive="left")
    dates = dates.strftime(date_format)

    mock_data = pd.DataFrame(
        {
            "startTime": dates,
            "publishTime": ["2021-11-08T10:00:00Z"] * 48,
            "demand": [10.0] * 48,
            "settlementPeriod": range(1, 49),
        }
    )

    # add data that is too recent so will be cutoff
    dup = mock_data.copy()
    dup["publishTime"] = dup["startTime"]
    mock_data = pd.concat([mock_data, dup])

    # add data with a slightly later created date
    dup = mock_data.copy()
    dup["publishTime"] = (
        pd.to_datetime(dup["publishTime"]) + pd.Timedelta("30 minutes")
    ).dt.strftime(date_format)
    dup["demand"] = dup["demand"] + 10
    mock_data = pd.concat([mock_data, dup])

    # add data with not enough settlement periods
    dup2 = dup.copy()
    dup2["startTime"] = (
        pd.to_datetime(dup2["startTime"]) + pd.Timedelta("1 day")
    ).dt.strftime(date_format)
    dup2 = dup2.iloc[:-1, :]
    mock_data = pd.concat([mock_data, dup2])

    def mock_read_csv(fp):
        return mock_data

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)

    result = prepare_ted_forecast(None)

    desired_result = pd.DataFrame(
        {"TED_DA_FORECAST": [20.0]},
        index=pd.DatetimeIndex(["2021-11-11"], freq="D", name="GAS_DAY"),
    )

    assert_frame_equal(result, desired_result)


def test_prepare_wind_forecast(monkeypatch):
    date_format = "%Y-%m-%dT%H:%M:%SZ"
    dates = pd.date_range("2021-11-11", "2021-11-12", freq="1h", inclusive="left")
    dates = dates.strftime(date_format)

    mock_data = pd.DataFrame(
        {
            "startTime": dates,
            "publishTime": ["2021-11-08T10:00:00Z"] * 24,
            "generation": [10.0] * 24,
        }
    )

    # add data that is too recent so will be cutoff
    dup = mock_data.copy()
    dup["publishTime"] = dup["startTime"]
    mock_data = pd.concat([mock_data, dup])

    # add data with a slightly later created date
    dup = mock_data.copy()
    dup["publishTime"] = (
        pd.to_datetime(dup["publishTime"]) + pd.Timedelta("30 minutes")
    ).dt.strftime(date_format)
    dup["generation"] = dup["generation"] + 10
    mock_data = pd.concat([mock_data, dup])

    # add data with not enough settlement periods
    dup2 = dup.copy()
    dup2["startTime"] = (
        pd.to_datetime(dup2["startTime"]) + pd.Timedelta("1 day")
    ).dt.strftime(date_format)
    dup2 = dup2.iloc[:-1, :]
    mock_data = pd.concat([mock_data, dup2])

    def mock_read_csv(fp):
        return mock_data

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)

    result = prepare_wind_forecast(None)

    desired_result = pd.DataFrame(
        {"WIND_FORECAST": [20.0]},
        index=pd.DatetimeIndex(["2021-11-11"], freq="D", name="GAS_DAY"),
    )

    assert_frame_equal(result, desired_result)


def test_prepare_electricity_actuals(monkeypatch):
    date_format = "%Y-%m-%dT%H:%M:%SZ"
    dates = pd.date_range("2021-11-11", "2021-11-13", freq="30min", inclusive="left")
    dates = dates.strftime(date_format)

    mock_data = pd.DataFrame(
        {
            "startTime": dates,
            "publishTime": ["2021-11-08T10:00:00Z"] * 48 * 2,
            "settlementPeriod": list(range(1, 49)) * 2,
            "COAL": [1.0] * 48 * 2,
            "WIND": [1.0] * 48 * 2,
            "CCGT": [1.0] * 48 * 2,
        }
    )
    mock_data = mock_data.melt(
        id_vars=["startTime", "publishTime", "settlementPeriod"],
        var_name="fuelType",
        value_name="generation",
    )

    def mock_read_csv(fp):
        return mock_data

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    result = prepare_electricity_actuals(None)

    desired_result = pd.DataFrame(
        {"CCGT": [1.0], "COAL": [1.0], "WIND": [1.0]},
        index=pd.DatetimeIndex(
            [pd.to_datetime("2021-11-11")], name="GAS_DAY", freq="D"
        ),
    )
    assert_frame_equal(result, desired_result)


def test_prepare_electricity_actuals_duplicates(monkeypatch):
    date_format = "%Y-%m-%dT%H:%M:%SZ"
    dates = pd.date_range("2021-11-11", "2021-11-13", freq="30min", inclusive="left")
    dates = dates.strftime(date_format)

    mock_data = pd.DataFrame(
        {
            "startTime": dates,
            "publishTime": ["2021-11-08T10:00:00Z"] * 48 * 2,
            "settlementPeriod": list(range(1, 49)) * 2,
            "COAL": [1.0] * 48 * 2,
            "WIND": [1.0] * 48 * 2,
            "CCGT": [1.0] * 48 * 2,
        }
    )

    # add duplicate with different created_on date
    dup = mock_data[mock_data["settlementPeriod"] == 1].copy()
    dup["publishTime"] = (
        pd.to_datetime(dup["publishTime"]) + pd.Timedelta("30 minutes")
    ).dt.strftime(date_format)
    dup["COAL"] = 10.0
    mock_data = pd.concat([mock_data, dup])

    mock_data = mock_data.melt(
        id_vars=["startTime", "publishTime", "settlementPeriod"],
        var_name="fuelType",
        value_name="generation",
    )

    def mock_read_csv(fp):
        return mock_data

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    result = prepare_electricity_actuals(None)

    desired_result = pd.DataFrame(
        {"CCGT": [1.0], "COAL": [1.1875], "WIND": [1.0]},
        index=pd.DatetimeIndex(
            [pd.to_datetime("2021-11-11")], name="GAS_DAY", freq="D"
        ),
    )
    assert_frame_equal(result, desired_result)


def test_prepare_electricity_actuals_cutoff(monkeypatch):
    date_format = "%Y-%m-%dT%H:%M:%SZ"
    dates = pd.date_range(
        "2021-11-11 05:00", "2021-11-11 15:00", freq="30min", inclusive="left"
    )
    dates = dates.strftime(date_format)

    mock_data = pd.DataFrame(
        {
            "startTime": dates,
            "publishTime": ["2021-11-11T10:00:00Z"] * len(dates),
            "settlementPeriod": list(range(1, len(dates) + 1)),
            "COAL": [1.0] * len(dates),
            "WIND": [1.0] * len(dates),
            "CCGT": [1.0] * len(dates),
        }
    )
    # add duplicate with created_on date that is too late
    dup = mock_data[mock_data["settlementPeriod"] == 1].copy()
    dup["publishTime"] = (
        pd.to_datetime(dup["publishTime"]) + pd.Timedelta("4 hours")
    ).dt.strftime(date_format)
    dup["COAL"] = 10.0
    dup["WIND"] = 10.0
    dup["CCGT"] = 10.0
    mock_data = pd.concat([mock_data, dup])

    mock_data = mock_data.melt(
        id_vars=["startTime", "publishTime", "settlementPeriod"],
        var_name="fuelType",
        value_name="generation",
    )

    def mock_read_csv(fp):
        return mock_data

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    result = prepare_electricity_actuals(None, actual_on_day_available_only=True)

    desired_result = pd.DataFrame(
        {"CCGT": [1.0], "COAL": [1.0], "WIND": [1.0]},
        index=pd.DatetimeIndex(
            [pd.to_datetime("2021-11-11")], name="GAS_DAY", freq="D"
        ),
    )
    assert_frame_equal(result, desired_result)


def test_prepare_actual_d_sofar_all_but_wind_gt(monkeypatch):
    mock_data = pd.DataFrame(
        {
            "COAL": range(10),
            "WIND": range(10, 20),
            "NUCLEAR": range(20, 30),
            "CCGT": range(10),
            "OCGT": range(10),
        },
        index=pd.date_range("2022-08-01", "2022-08-10"),
    )

    def mock_gas_actuals(fp, actual_on_day_available_only):
        return mock_data

    monkeypatch.setattr(
        src.prepare_data,
        "prepare_electricity_actuals",
        mock_gas_actuals,
    )

    result = prepare_actual_sofar(None)

    desired_result = pd.DataFrame(
        {
            "ACTUAL_D_SOFAR_ALL_BUT_WIND_AND_GT": [
                20.0,
                22.0,
                24.0,
                26.0,
                28.0,
                30.0,
                32.0,
                34.0,
                36.0,
                38.0,
                np.NaN,
            ]
        },
        index=pd.DatetimeIndex(
            pd.date_range("2022-08-02", "2022-08-12"), name="GAS_DAY", freq="D"
        ),
    )

    assert_frame_equal(result, desired_result)


def test_prepare_gen_previous_gas_day(monkeypatch):
    mock_data = pd.DataFrame(
        {
            "settlementDate": ["2021-01-11"] * 48
            + ["2021-01-12"] * 48
            + ["2021-01-13"] * 48
            + ["2021-01-14"] * 48,
            "publishTime": ["2021-01-13T12:00:00Z"] * 192,
            "settlementPeriod": list(range(1, 49)) * 4,
            "COAL": [1.0] * 192,
            "WIND": [1.5] * 192,
            "CCGT": [1.0] * 48 + [2.0] * 48 + [3.0] * 48 + [4.0] * 48,
        }
    )

    mock_data["INTELEC"] = mock_data["INTEW"] = mock_data["INTFR"] = mock_data[
        "INTIFA2"
    ] = mock_data["INTIRL"] = mock_data["INTNED"] = mock_data["INTNEM"] = mock_data[
        "INTNSL"
    ] = mock_data[
        "OIL"
    ] = mock_data[
        "NUCLEAR"
    ] = mock_data[
        "NPSHYD"
    ] = mock_data[
        "PS"
    ] = mock_data[
        "BIOMASS"
    ] = mock_data[
        "OCGT"
    ] = mock_data[
        "OTHER"
    ] = mock_data[
        "COAL"
    ]

    mock_data["startTime"] = pd.to_datetime(mock_data["settlementDate"]) + pd.Timedelta(
        "30 min"
    ) * (mock_data["settlementPeriod"] - 1)
    mock_data["startTime"] = mock_data["startTime"].dt.strftime(
        date_format="%Y-%m-%dT%H:%M:%SZ"
    )

    mock_data = mock_data.melt(
        id_vars=["settlementDate", "publishTime", "settlementPeriod", "startTime"],
        value_name="generation",
        var_name="fuelType",
    )

    def mock_read_csv(fp):
        return mock_data

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    # Test function
    output = prepare_gen_previous_gas_day(None)

    # Expected output
    expected_output = np.concatenate(
        [
            np.ones((3, 48)) * 8,
            (np.ones((10, 3)) * np.array([3, 4, 5])).T,
            (np.ones((38, 3)) * np.array([2, 3, 4])).T,
            np.ones((3, 48)) * 7.0,
            np.ones((3, 48)) * 1.5,
        ],
        axis=1,
    )

    expected_output = pd.DataFrame(
        expected_output,
        columns=["INTERCONNECTORS_" + str(i) for i in range(1, 49)]
        + ["POWER_STATION_" + str(i) for i in range(1, 49)]
        + ["REST_" + str(i) for i in range(1, 49)]
        + ["WIND_" + str(i) for i in range(1, 49)],
        index=pd.DatetimeIndex(
            ["2021-01-13", "2021-01-14", "2021-01-15"], name="GAS_DAY"
        ),
    )

    # Assert that the output is as expected
    assert_frame_equal(output, expected_output)


def test_aggregate_generation_data():
    mock_data = pd.DataFrame(
        {
            "GAS_DAY": ["2021-01-11"] * 48,
            "SETTLEMENT_PERIOD": range(1, 49),
            "CCGT": [1] * 48,
            "OIL": [2] * 48,
            "COAL": [3] * 48,
            "NUCLEAR": [4] * 48,
            "WIND": [5] * 48,
            "PS": [6] * 48,
            "NPSHYD": [7] * 48,
            "OCGT": [8] * 48,
            "OTHER": [9] * 48,
            "INTFR": [10] * 48,
            "INTIRL": [11] * 48,
            "INTNED": [12] * 48,
            "INTEW": [13] * 48,
            "BIOMASS": [14] * 48,
            "INTNEM": [15] * 48,
            "INTELEC": [16] * 48,
            "INTIFA2": [17] * 48,
            "INTNSL": [18] * 48,
        },
    )

    result = aggregate_generation_data(mock_data)

    desired_result = pd.DataFrame(
        {
            "GAS_DAY": ["2021-01-11"] * 48,
            "SETTLEMENT_PERIOD": range(1, 49),
            "WIND": [5] * 48,
            "INTERCONNECTORS": [112] * 48,
            "POWER_STATION": [9] * 48,
            "REST": [45] * 48,
        },
    )

    assert_frame_equal(result, desired_result)


def test_prepare_hourly_wind_forecast(monkeypatch):
    mock_data = pd.DataFrame(
        {
            "startTime": ["2021-01-11"] * 24,
            "generation": [10] * 24,
            "publishTime": ["2021-01-08T10:00:00Z"] * 24,
            "sp": range(0, 24),
        }
    )

    mock_data["startTime"] = (
        pd.to_datetime(mock_data["startTime"])
        + pd.Timedelta("1 hour") * mock_data["sp"]
    )
    mock_data["startTime"] = mock_data["startTime"].dt.strftime(
        date_format="%Y-%m-%dT%H:%M:%SZ"
    )

    # add data that is too recent so will be cutoff
    dup = mock_data.copy()
    dup["publishTime"] = (
        pd.to_datetime(dup["publishTime"])
        .dt.floor(freq="H")
        .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    mock_data = mock_data.append(dup)

    # add data with a slightly later created date
    dup = mock_data.copy()
    dup["publishTime"] = (
        pd.to_datetime(dup["publishTime"]) + pd.Timedelta("30 minutes")
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    dup["generation"] = dup["generation"] + 10
    mock_data = mock_data.append(dup)

    # add data with not enough settlement periods
    dup2 = dup.copy()
    dup2["startTime"] = "2021-01-12"
    dup2["startTime"] = (
        pd.to_datetime(dup2["startTime"]) + pd.Timedelta("1 hour") * dup2["sp"]
    )
    dup2["startTime"] = dup2["startTime"].dt.strftime(date_format="%Y-%m-%dT%H:%M:%SZ")
    dup2 = dup2.iloc[:-1, :]
    mock_data = mock_data.append(dup2).drop(columns="sp")

    def mock_read_csv(fp):
        return mock_data

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)

    result = prepare_hourly_wind_forecast(None)
    desired_result = np.array([[20]] * 24).T
    desired_result = pd.DataFrame(
        desired_result,
        columns=["WIND_FORECAST_" + str(i) for i in range(1, 49, 2)],
        index=pd.DatetimeIndex(["2021-01-11"], name="GAS_DAY"),
    )

    assert_frame_equal(result, desired_result, check_dtype=False)


def test_get_ted_half_hourly_forecast(monkeypatch):
    mock_data = pd.DataFrame(
        {
            "settlementDate": ["2021-01-10"] * 48 + ["2021-01-11"] * 48,
            "publishTime": ["2021-01-08T10:00:00Z"] * 96,
            "settlementPeriod": list(range(1, 49)) * 2,
            "demand": [10.0] * 48 + [20.0] * 48,
            "dataset": ["dummy"] * 96,
        }
    )

    mock_data["startTime"] = pd.to_datetime(mock_data["settlementDate"]) + pd.Timedelta(
        "30 min"
    ) * (mock_data["settlementPeriod"] - 1)
    mock_data["startTime"] = mock_data["startTime"].dt.strftime(
        date_format="%Y-%m-%dT%H:%M:%SZ"
    )

    # add data that is too recent so will be cutoff
    dup = mock_data.copy()
    dup["publishTime"] = dup["settlementDate"] + "T00:00:00Z"
    mock_data = pd.concat([mock_data, dup])

    # add data with a slightly later created date
    dup = mock_data.copy()
    dup["publishTime"] = (
        pd.to_datetime(dup["publishTime"]) + pd.Timedelta("30 minutes")
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    dup["demand"] = dup["demand"] + 10
    mock_data = pd.concat([mock_data, dup])

    # add data with not enough settlement periods
    dup2 = dup.copy()
    dup2["settlementDate"] = "2021-01-12"
    dup2 = dup2.iloc[:-1, :]
    dup2["startTime"] = pd.to_datetime(dup2["settlementDate"]) + pd.Timedelta(
        "30 min"
    ) * (dup2["settlementPeriod"] - 1)
    dup2["startTime"] = dup2["startTime"].dt.strftime(date_format="%Y-%m-%dT%H:%M:%SZ")
    mock_data = pd.concat([mock_data, dup2])

    def mock_read_csv(fp):
        return mock_data

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)

    result = prepare_ted_half_hourly_forecast(None, days=0)

    desired_result = np.concatenate(
        [
            (np.ones((10, 2)) * np.array([30, 20])),
            (np.ones((38, 2)) * np.array([20, 30])),
        ]
    ).T
    desired_result = pd.DataFrame(
        desired_result,
        columns=["DEMAND_" + str(i) for i in range(1, 49)],
        index=pd.DatetimeIndex(["2021-01-10", "2021-01-11"], name="GAS_DAY"),
    )

    assert_frame_equal(result, desired_result)


def test_get_ted_half_hourly_forecast_previous_day(monkeypatch):
    mock_data = pd.DataFrame(
        {
            "settlementDate": ["2021-01-10"] * 48 + ["2021-01-11"] * 48,
            "publishTime": ["2021-01-08T10:00:00Z"] * 96,
            "settlementPeriod": list(range(1, 49)) * 2,
            "demand": [10.0] * 48 + [20.0] * 48,
            "dataset": ["dummy"] * 96,
        }
    )
    mock_data["startTime"] = pd.to_datetime(mock_data["settlementDate"]) + pd.Timedelta(
        "30 min"
    ) * (mock_data["settlementPeriod"] - 1)
    mock_data["startTime"] = mock_data["startTime"].dt.strftime(
        date_format="%Y-%m-%dT%H:%M:%SZ"
    )

    # add data that is too recent so will be cutoff
    dup = mock_data.copy()
    dup["publishTime"] = dup["settlementDate"] + "T00:00:00Z"
    mock_data = pd.concat([mock_data, dup])

    # add data with a slightly later created date
    dup = mock_data.copy()
    dup["publishTime"] = (
        pd.to_datetime(dup["publishTime"]) + pd.Timedelta("30 minutes")
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    dup["demand"] = dup["demand"] + 10
    mock_data = pd.concat([mock_data, dup])

    # add data with not enough settlement periods
    dup2 = dup.copy()
    dup2["settlementDate"] = "2021-01-12"
    dup2 = dup2.iloc[:-1, :]
    dup2["startTime"] = pd.to_datetime(dup2["settlementDate"]) + pd.Timedelta(
        "30 min"
    ) * (dup2["settlementPeriod"] - 1)
    dup2["startTime"] = dup2["startTime"].dt.strftime(date_format="%Y-%m-%dT%H:%M:%SZ")
    mock_data = pd.concat([mock_data, dup2])

    def mock_read_csv(fp):
        return mock_data

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)

    result = prepare_ted_half_hourly_forecast(None, days=1)

    desired_result = np.array([[30.0] * 10 + [20.0] * 38])
    desired_result = pd.DataFrame(
        desired_result,
        columns=["DEMAND_" + str(i) for i in range(1, 49)],
        index=pd.DatetimeIndex(["2021-01-11"], name="GAS_DAY"),
    )

    assert_frame_equal(result, desired_result)
