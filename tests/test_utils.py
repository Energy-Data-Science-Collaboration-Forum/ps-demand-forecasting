import datetime as dt
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from src.utils import (
    remove_incomplete_settlement_periods,
    cutoff_forecast,
    infer_gas_day,
    remove_zero_ccgt,
    flatten_data,
    fill_46_settlement_period,
    remove_50_settlement_period,
)


def test_infer_gas_day():

    # date in the format year-month-day
    today = dt.date.today()

    # if greater than hardwired 10 settlement periods
    gas_day = infer_gas_day(11, today)
    assert gas_day == today

    # if smaller than hardwired 10 settlement periods
    gas_day = infer_gas_day(5, today)
    assert gas_day == today - dt.timedelta(days=1)

    # if equal to hardwired 10 settlement periods
    gas_day = infer_gas_day(10, today)
    assert gas_day == today - dt.timedelta(days=1)

    # if negative (impossible value)
    gas_day = infer_gas_day(-1, today)
    assert gas_day == today - dt.timedelta(days=1)

    # if zero settlement period
    gas_day = infer_gas_day(0, today)
    assert gas_day == today - dt.timedelta(days=1)

    # date with time detail
    now = dt.datetime.now()
    gas_day = infer_gas_day(0, now)
    assert gas_day == now - dt.timedelta(days=1)


def test_cutoff_forecast_parameterised():
    my_data = pd.DataFrame(
        {
            "GAS_DAY": ["2021-11-01", "2021-11-03", "2021-11-03"],
            "CREATED_ON": ["2021-11-01 09:00", "2021-11-01 10:00", "2021-11-01 13:00"],
        }
    )
    my_data["GAS_DAY"] = pd.to_datetime(my_data["GAS_DAY"])
    my_data["CREATED_ON"] = pd.to_datetime(my_data["CREATED_ON"])
    result = cutoff_forecast(my_data, days=2, hour=10)

    expected = pd.DataFrame(
        {
            "GAS_DAY": [pd.to_datetime("2021-11-03")],
            "CREATED_ON": [pd.to_datetime("2021-11-01 10:00")],
        },
        index=pd.Index([1], dtype="int64"),
    )

    assert_frame_equal(result, expected)


def test_cutoff_forecast_empty(caplog):
    my_data = pd.DataFrame(
        {
            "GAS_DAY": ["2021-11-01", "2021-11-01", "2021-11-01"],
            "CREATED_ON": ["2021-11-01 09:00", "2021-11-01 10:00", "2021-10-31 13:00"],
        }
    )
    my_data["GAS_DAY"] = pd.to_datetime(my_data["GAS_DAY"])
    my_data["CREATED_ON"] = pd.to_datetime(my_data["CREATED_ON"])
    result = cutoff_forecast(my_data)

    assert result.shape[0] == 0
    assert len(caplog.records) == 1
    assert (
        caplog.records[0].getMessage()
        == "cutoff_forecast: Data cutoff returned no data with difference of day: 1, hour: 12"
    )


def test_cutoff_forecast_normal():
    my_data = pd.DataFrame(
        {
            "GAS_DAY": ["2021-11-01", "2021-11-02"],
            "CREATED_ON": ["2021-11-01 09:00", "2021-11-01 10:00"],
        }
    )
    my_data["GAS_DAY"] = pd.to_datetime(my_data["GAS_DAY"])
    my_data["CREATED_ON"] = pd.to_datetime(my_data["CREATED_ON"])
    result = cutoff_forecast(my_data)

    expected = pd.DataFrame(
        {
            "GAS_DAY": [pd.to_datetime("2021-11-02")],
            "CREATED_ON": [pd.to_datetime("2021-11-01 10:00")],
        },
        index=pd.Index([1], dtype="int64"),
    )

    assert_frame_equal(result, expected)


def test_cutoff_forecast_time_of_day_fix():
    my_data = pd.DataFrame(
        {
            "GAS_DAY": ["2021-11-01", "2021-11-03", "2021-11-03"],
            "CREATED_ON": ["2021-11-01 09:00", "2021-11-01 13:00", "2021-11-02 13:00"],
        }
    )
    my_data["GAS_DAY"] = pd.to_datetime(my_data["GAS_DAY"])
    my_data["CREATED_ON"] = pd.to_datetime(my_data["CREATED_ON"])
    result = cutoff_forecast(my_data)

    expected = pd.DataFrame(
        {
            "GAS_DAY": [pd.to_datetime("2021-11-03")],
            "CREATED_ON": [pd.to_datetime("2021-11-01 13:00")],
        },
        index=pd.Index([1], dtype="int64"),
    )

    assert_frame_equal(result, expected)


def test_remove_incomplete_settlement_periods_complete(caplog):
    dummy_data = pd.DataFrame(
        {
            "GAS_DAY": ["2021-11-01"] * 48,
            "SETTLEMENT_PERIOD": list(range(1, 49)),
        }
    )

    result = remove_incomplete_settlement_periods("Dummy", dummy_data)

    desired_result = pd.DataFrame(
        {
            "GAS_DAY": ["2021-11-01"] * 48,
            "SETTLEMENT_PERIOD": range(1, 49),
        }
    )

    assert_frame_equal(desired_result, result)
    assert len(caplog.records) == 0


def test_remove_incomplete_settlement_periods(caplog):
    dummy_data = pd.DataFrame(
        {
            "GAS_DAY": ["2021-11-01"] * 48 + ["2021-11-02"] * 47,
            "SETTLEMENT_PERIOD": list(range(1, 49)) + list(range(1, 48)),
        }
    )

    result = remove_incomplete_settlement_periods("Dummy", dummy_data)

    desired_result = pd.DataFrame(
        {
            "GAS_DAY": ["2021-11-01"] * 48,
            "SETTLEMENT_PERIOD": range(1, 49),
        }
    )

    assert_frame_equal(desired_result, result)

    assert len(caplog.records) == 1
    assert (
        caplog.records[0].getMessage()
        == "Dummy has 1 days with missing Settlement Periods at ['2021-11-02']. These will be dropped."
    )


def test_remove_incomplete_settlement_periods_dst():
    dummy_data = pd.DataFrame(
        {
            "GAS_DAY": ["2021-11-01"] * 46 + ["2021-11-02"] * 50,
            "SETTLEMENT_PERIOD": list(range(1, 47)) + list(range(1, 51)),
        }
    )

    result = remove_incomplete_settlement_periods("Dummy", dummy_data)

    desired_result = pd.DataFrame(
        {
            "GAS_DAY": ["2021-11-01"] * 46 + ["2021-11-02"] * 50,
            "SETTLEMENT_PERIOD": list(range(1, 47)) + list(range(1, 51)),
        }
    )

    assert_frame_equal(desired_result, result)


def test_remove_incomplete_settlement_periods_complete_hourly(caplog):
    dummy_data = pd.DataFrame(
        {
            "GAS_DAY": ["2021-11-01"] * 24,
            "SETTLEMENT_PERIOD": list(range(1, 49, 2)),
        }
    )

    result = remove_incomplete_settlement_periods("Dummy", dummy_data, True)

    desired_result = pd.DataFrame(
        {
            "GAS_DAY": ["2021-11-01"] * 24,
            "SETTLEMENT_PERIOD": list(range(1, 49, 2)),
        }
    )

    assert_frame_equal(desired_result, result)
    assert len(caplog.records) == 0


def test_remove_incomplete_settlement_periods_hourly(caplog):
    dummy_data = pd.DataFrame(
        {
            "GAS_DAY": ["2021-11-01"] * 24 + ["2021-11-02"] * 22,
            "SETTLEMENT_PERIOD": list(range(1, 49, 2)) + list(range(1, 45, 2)),
        }
    )

    result = remove_incomplete_settlement_periods("Dummy", dummy_data, True)

    desired_result = pd.DataFrame(
        {
            "GAS_DAY": ["2021-11-01"] * 24,
            "SETTLEMENT_PERIOD": range(1, 49, 2),
        }
    )

    assert_frame_equal(desired_result, result)

    assert len(caplog.records) == 1
    assert (
        caplog.records[0].getMessage()
        == "Dummy has 1 days with missing Settlement Periods at ['2021-11-02']. These will be dropped."
    )


def test_remove_incomplete_settlement_periods_dst_hourly():
    dummy_data = pd.DataFrame(
        {
            "GAS_DAY": ["2021-11-01"] * 23 + ["2021-11-02"] * 25,
            "SETTLEMENT_PERIOD": list(range(1, 47, 2)) + list(range(1, 51, 2)),
        }
    )

    result = remove_incomplete_settlement_periods("Dummy", dummy_data, True)

    desired_result = pd.DataFrame(
        {
            "GAS_DAY": ["2021-11-01"] * 23 + ["2021-11-02"] * 25,
            "SETTLEMENT_PERIOD": list(range(1, 47, 2)) + list(range(1, 51, 2)),
        }
    )

    assert_frame_equal(desired_result, result)


def test_remove_zero_ccgt(caplog):

    dummy_data = pd.DataFrame({"One": [1, 2, 3], "CCGT": [1, 2, 0]})

    result = remove_zero_ccgt("Dummy", dummy_data)

    desired_result = pd.DataFrame({"One": [1, 2], "CCGT": [1, 2]})

    assert_frame_equal(result, desired_result)

    assert len(caplog.records) == 1
    assert (
        caplog.records[0].getMessage()
        == "Dummy has 1 Settlement Periods with 0 value CCGT, dropped those Settlement Periods"
    )


def test_flatten_data():
    mock_data = pd.DataFrame(
        {
            "GAS_DAY": ["2021-01-11"] * 48 + ["2021-01-12"] * 48,
            "SETTLEMENT_PERIOD": list(range(1, 49)) * 2,
            "WIND": [11] * 48 + [22] * 48,
        },
    )

    result = flatten_data(mock_data)

    desired_result = pd.DataFrame(
        index=["2021-01-11", "2021-01-12"],
        columns=[f"WIND_{number}" for number in range(1, 49)],
    )
    desired_result.index.name = "GAS_DAY"
    desired_result.loc["2021-01-11"] = 11
    desired_result.loc["2021-01-12"] = 22
    desired_result = desired_result.astype(
        {"WIND_{number}".format(number=number): np.int64 for number in range(1, 49)}
    )

    assert_frame_equal(result, desired_result)


def test_fill_46_settlement_period():
    # Test data
    mock_gen_data = pd.DataFrame(
        {
            "ELEC_DAY": ["2021-01-11"] * 46,
            "GAS_DAY": ["2021-01-11"] * 46,
            "CREATED_ON": ["2021-01-10 12:00:00"] * 46,
            "SETTLEMENT_PERIOD": range(1, 47),
            "COAL": [1.0] * 46,
            "WIND": [1.0] * 46,
            "CCGT": [1.0] + [3, 4] + [1.0] * 43,
        }
    )

    # Expected output
    expected_output = pd.DataFrame(
        {
            "ELEC_DAY": ["2021-01-11"] * 48,
            "SETTLEMENT_PERIOD": range(1, 49),
            "GAS_DAY": ["2021-01-11"] * 48,
            "CREATED_ON": ["2021-01-10 12:00:00"] * 48,
            "COAL": [1.0] * 48,
            "WIND": [1.0] * 48,
            "CCGT": [1.0] + [3, 3, 4, 4] + [1.0] * 43,
        }
    )

    # Test function
    output = fill_46_settlement_period(mock_gen_data)

    # Assert that the output is as expected
    assert_frame_equal(output, expected_output)


def test_remove_50_settlement_period():
    # Test data
    mock_gen_data = pd.DataFrame(
        {
            "ELEC_DAY": ["2021-01-11"] * 50,
            "GAS_DAY": ["2021-01-11"] * 50,
            "CREATED_ON": ["2021-01-10 12:00:00"] * 50,
            "SETTLEMENT_PERIOD": range(1, 51),
            "COAL": [1.0] * 50,
            "WIND": [1.0] * 50,
            "CCGT": [1.0] * 2 + [6, 5, 2, 4] + [6.0] * 44,
        }
    )

    # Expected output
    expected_output = pd.DataFrame(
        {
            "ELEC_DAY": ["2021-01-11"] * 48,
            "SETTLEMENT_PERIOD": range(1, 49),
            "GAS_DAY": ["2021-01-11"] * 48,
            "CREATED_ON": ["2021-01-10 12:00:00"] * 48,
            "COAL": [1.0] * 48,
            "WIND": [1.0] * 48,
            "CCGT": [1.0] * 2 + [4.0, 4.5] + [6.0] * 44,
        }
    )

    # Test function
    output = remove_50_settlement_period(mock_gen_data)

    # Assert that the output is as expected
    assert_frame_equal(output, expected_output)
