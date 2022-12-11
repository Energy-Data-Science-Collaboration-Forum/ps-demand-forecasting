import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import src.prepare_data
from src.prepare_data import prepare_gas_demand_actuals, prepare_ted_forecast, prepare_wind_forecast, prepare_electricity_actuals, prepare_actual_sofar


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
    dates = pd.date_range("2021-11-11", "2021-11-12", freq="30min", inclusive='left')
    dates = dates.strftime(date_format)

    mock_data = pd.DataFrame(
        {
            "startTime": dates,
            "publishTime": ["2021-11-08T10:00:00Z"] * 48,
            "demand": [10.0] * 48,
            "settlementPeriod":range(1,49)
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
    dup2["startTime"] = (pd.to_datetime(dup2["startTime"]) + pd.Timedelta("1 day")
    ).dt.strftime(date_format)
    dup2 = dup2.iloc[:-1, :]
    mock_data = pd.concat([mock_data, dup2])

    def mock_read_csv(fp):
        return mock_data

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)

    result = prepare_ted_forecast(None)

    desired_result = pd.DataFrame({"TED_DA_FORECAST": [20.0]},
        index=pd.DatetimeIndex(["2021-11-11"], freq="D", name="GAS_DAY"),)

    assert_frame_equal(result, desired_result)


def test_prepare_wind_forecast(monkeypatch):
    date_format = "%Y-%m-%dT%H:%M:%SZ"
    dates = pd.date_range("2021-11-11", "2021-11-12", freq="1h", inclusive='left')
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
    dup2["startTime"] = (pd.to_datetime(dup2["startTime"]) + pd.Timedelta("1 day")
    ).dt.strftime(date_format)
    dup2 = dup2.iloc[:-1, :]
    mock_data = pd.concat([mock_data, dup2])

    def mock_read_csv(fp):
        return mock_data

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)

    result = prepare_wind_forecast(None)

    desired_result = pd.DataFrame({"WIND_FORECAST": [20.0]},
        index=pd.DatetimeIndex(["2021-11-11"], freq="D", name="GAS_DAY"),)

    assert_frame_equal(result, desired_result)


def test_get_electricity_actuals(monkeypatch):
    date_format = "%Y-%m-%dT%H:%M:%SZ"
    dates = pd.date_range("2021-11-11", "2021-11-13", freq="30min", inclusive='left')
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
    mock_data = mock_data.melt(id_vars=['startTime', "publishTime","settlementPeriod"], 
        var_name="fuelType", value_name="generation")
    
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


def test_get_electricity_actuals_duplicates(monkeypatch):
    date_format = "%Y-%m-%dT%H:%M:%SZ"
    dates = pd.date_range("2021-11-11", "2021-11-13", freq="30min", inclusive='left')
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

    mock_data = mock_data.melt(id_vars=['startTime', "publishTime","settlementPeriod"], 
        var_name="fuelType", value_name="generation")

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


def test_get_electricity_actuals_cutoff(monkeypatch):
    date_format = "%Y-%m-%dT%H:%M:%SZ"
    dates = pd.date_range("2021-11-11 05:00", "2021-11-11 15:00", freq="30min", inclusive='left')
    dates = dates.strftime(date_format)

    mock_data = pd.DataFrame(
            {
                "startTime": dates,
                "publishTime": ["2021-11-11T10:00:00Z"] * len(dates),
                "settlementPeriod": list(range(1, len(dates)+1)),
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

    mock_data = mock_data.melt(id_vars=['startTime', "publishTime","settlementPeriod"], 
        var_name="fuelType", value_name="generation")

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



def test_get_actual_d_sofar_all_but_wind_gt(monkeypatch):
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
