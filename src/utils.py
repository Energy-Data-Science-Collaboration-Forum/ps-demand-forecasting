import logging
import pandas as pd
import datetime as dt

logger = logging.getLogger(__name__)


def infer_gas_day(sp, elec_day, actual_datetime=None):
    """
    Gas Days start and end at 05:00:00.

    UK Electricity Days start and end at 00:00:00 and are split into 48
    Settlement Periods (SPS).

    Method
    Use Elec_datetime if we have it and map to gas_day, otherwise
    maps a record with a given SP to the appropriate GAS_DAY.

    Args:
        sp (integer): [description]
        elec_day (datetime): electricity day in datetime format
        actual_datetime (datetime, optional): [description]. Defaults to None.

    Returns:
        gas_day: datetime
    """
    # if we have elec_datetime then use it, if not then infer from everything else
    if actual_datetime is not None:
        gas_day = actual_datetime + dt.timedelta(hours=-5)
        gas_day = gas_day.date()
        return gas_day

    try:
        if 0 < sp <= 48:
            pass
    except:
        logger.warning("Settlement period needs to be between 1 and 48, it's " + str(sp))
        return pd.NaT

    if sp <= 10:
        gas_day = elec_day - dt.timedelta(days=1)
    else:
        gas_day = elec_day

    return gas_day


def remove_incomplete_settlement_periods(
    name, my_data, hourly_settlement_periods=False
):
    """
    For electricity data, checks that all settlement periods are there for a given day (assumes no duplicate entries for a given day)

    Args:
        name (str): Name of the dataset for logging purposes
        my_data (dataframe): Dataframe of elec data, with GAS_DAY and SETTLEMENT_PERIOD as minimum
        hourly_settlement_periods (bool, optional): Whether or not the settlement periods are only recorded on the whole hour. Default False.

    Returns:
        my_data: given dataframe with gas days removed that have missing settlement peridos
    """

    # settlement periods are numbers from 1 to 48
    # except during the switch between daylight savings time and no daylight savings time
    # in March there are two less periods (46) and in October there are two more (50)
    sp_sums = my_data.groupby("GAS_DAY")["SETTLEMENT_PERIOD"].sum()

    # the sum of the settlement periods must equal 0.5 * 48 * (48 + 1) = 1176
    # see https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss#Anecdotes
    settlement_periods_sums = [1176, 1081, 1275]  # [48 periods, 46 periods, 50 periods]
    # NB: there is a risk that we're actually missing periods 47 and 48 but we've not come across that yet
    if hourly_settlement_periods:
        # unless it's only the settlement periods at the whole hour
        # these are all the odd numbered settlement periods
        # in which case the sum must be equal to 24 * 24 = 576
        # https://www.vedantu.com/maths/sum-of-odd-numbers
        settlement_periods_sums = [576, 529, 625]

    gas_days_missing = sp_sums[~sp_sums.isin(settlement_periods_sums)].index

    if len(gas_days_missing) > 0:
        logger.warning(
            f"{name} has {len(gas_days_missing)} days with missing Settlement Periods at {gas_days_missing.values}. These will be dropped."
        )
        my_data = my_data[~my_data["GAS_DAY"].isin(gas_days_missing)]

    return my_data


def cutoff_forecast(df, days=1, hour=12):
    """
    Based on CREATED_ON only keep data available a certain number of days before, at a certain time (hours), in relation to GAS_DAY
    i.e. get data that was available at same time that we want to predict for

    Args:
        df (dataframe): input dataframe with GAS_DAY and CREATED_ON
        days (int, optional): days before prediction day data should be available e.g. for day ahead days = 1. Defaults to 1.
        hour (int, optional): hour at which data should be available e.g. 12 is midday. Defaults to 12.

    Returns:
        dataframe: same data, but for available days and at time hours
    """

    cutoff_dates = df["GAS_DAY"].dt.date - dt.timedelta(days=days)
    cutoff_dates = pd.to_datetime(cutoff_dates) + dt.timedelta(hours=hour)

    # only keep data available created the day before we predict for, unless CREATED_ON is empty
    df = df[df["CREATED_ON"] <= cutoff_dates]

    if len(df) == 0:
        logger.critical(
            f"cutoff_forecast: Data cutoff returned no data with difference of day: {days}, hour: {hour}"
        )

    return df


def remove_zero_ccgt(name, elec_actuals):
    """Remove records where the value for CCGT was zero as this indicates an error in the data collection

    Args:
        name (str): Name of the dataset for logging purposes
        elec_actuals (DataFrame): A data frame with at least a column for CCGT

    Returns:
        DataFrame: A data frame with the zero observations removed
    """
    
    mask = elec_actuals["CCGT"] <= 0
    if sum(mask) > 0:
        logger.warning(f"{name} has {sum(mask)} Settlement Periods with 0 value CCGT, dropped those Settlement Periods")
        elec_actuals = elec_actuals[~mask]
    
    return elec_actuals