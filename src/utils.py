import logging
import numpy as np
import pandas as pd
import datetime as dt
from pandas.api.types import is_numeric_dtype

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
        logger.warning(
            "Settlement period needs to be between 1 and 48, it's " + str(sp)
        )
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
        logger.warning(
            f"{name} has {sum(mask)} Settlement Periods with 0 value CCGT, dropped those Settlement Periods"
        )
        elec_actuals = elec_actuals[~mask]

    return elec_actuals


def flatten_data(df):
    """
    Flattens a pandas DataFrame by pivoting it and combining multi-index columns into single level.

    Args:
    df (pandas.DataFrame): input DataFrame to be flattened, must have GAS_DAY and SETTLEMENT_PERIOD as columns.

    Returns:
    pandas.DataFrame: output flattened DataFrame.
    """
    # pivot data to wide format with GAS_DAY as index and SETTLEMENT_PERIOD as columns
    df = df.pivot_table(index="GAS_DAY", columns="SETTLEMENT_PERIOD").copy()
    # combine multi-index columns into single level
    df.columns = ["_".join(map(str, col)) for col in df.columns.values]
    return df


def fill_46_settlement_period(gen, hourly_settlement_periods=False):
    """
    Function to fill in missing settlement periods (SPs) of 3 and 4 on clock change days.

    Args:
        gen (pandas DataFrame): Raw dataframe that contains the electricity generation data.
        hourly_settlement_periods (bool): When True sets number of SPs at 23 otherwise at 46. Default is False.

    Returns:
        gen (pandas DataFrame): Wrangled dataframe with missing SPs (due to clock change) filled in.

    """
    count = 23 if hourly_settlement_periods else 46

    gen_46_sp = (
        gen[gen.groupby("ELEC_DAY")["SETTLEMENT_PERIOD"].transform("count") == count]
        .reset_index(drop=True)
        .copy()
    )

    gen_46_sp.loc[:, "SETTLEMENT_PERIOD"] = gen_46_sp.loc[:, "SETTLEMENT_PERIOD"].apply(
        lambda x: x + 2 if x > 2 else x
    )
    # Set GAS_DAY for SP 11 and 12 (SP9 and 10 originally hence date was wrong) to nan
    # so that it can be filled in with the next available GAS_DAY
    gen_46_sp.loc[gen_46_sp.SETTLEMENT_PERIOD.isin([11, 12]), "GAS_DAY"] = np.nan
    gen_46_sp["GAS_DAY"] = gen_46_sp["GAS_DAY"].fillna(method="bfill")

    sp_range = range(1, 48, 2) if hourly_settlement_periods else range(1, 49)
    # create new multi-index
    iterables = [gen_46_sp.ELEC_DAY.unique(), sp_range]
    new_index = pd.MultiIndex.from_product(
        iterables, names=["ELEC_DAY", "SETTLEMENT_PERIOD"]
    )

    gen_46_sp = (
        gen_46_sp.groupby(["ELEC_DAY", "SETTLEMENT_PERIOD"])
        .last()
        .reindex(new_index)  # create placeholders of missing SPs of 3 and 4
        .fillna(method="ffill", limit=1)  # fill SP3 with previous SP2
        .fillna(method="bfill")  # fill SP4 with next SP5
        .reset_index()
    )

    gen = (
        gen[~gen.ELEC_DAY.isin(gen_46_sp.ELEC_DAY.unique())]
        .merge(gen_46_sp, how="outer")
        .sort_values(["ELEC_DAY", "SETTLEMENT_PERIOD"], ascending=True)
        .reset_index(drop=True)
    )

    return gen


def remove_50_settlement_period(gen):
    """
    Function to reduce 50 settlement periods to 48 settlement periods on clock change days.

    Args:
        gen (pandas DataFrame): Raw dataframe that contains the electricity generation data.

    Returns:
        gen (pandas DataFrame): Wrangled dataframe with duplicate SPs (due to clock change) removed.

    """
    gen_50_sp = gen[
        gen.groupby("ELEC_DAY")["SETTLEMENT_PERIOD"].transform("count") == 50
    ].copy()
    if not gen_50_sp.empty:
        gen_50_sp.loc[:, "SETTLEMENT_PERIOD"] = gen_50_sp.loc[
            :, "SETTLEMENT_PERIOD"
        ].apply(lambda x: x - 2 if x > 4 else x)
        gen_50_sp.loc[gen_50_sp.SETTLEMENT_PERIOD.isin([9, 10]), "GAS_DAY"] = np.nan
        gen_50_sp["GAS_DAY"] = gen_50_sp["GAS_DAY"].fillna(method="ffill")

        gen_50_sp = (
            gen_50_sp.groupby(["ELEC_DAY", "SETTLEMENT_PERIOD"])
            .agg(
                [lambda x: x.mean() if is_numeric_dtype(x) else x.iloc[0]]
            )  # take mean or get the first value if not numeric
            .droplevel(1, axis=1)  # drop lambda column name
            .reset_index()
        )

        gen = (
            gen[~gen.ELEC_DAY.isin(gen_50_sp.ELEC_DAY.unique())]
            .merge(gen_50_sp, how="outer")
            .sort_values(["ELEC_DAY", "SETTLEMENT_PERIOD"], ascending=True)
            .reset_index(drop=True)
        )
    return gen


def fix_missing_values(input_data):
    """Remove missing values from the beginning and end of the data and fill the remaining ones

    Args:
        input_data (DataFrame): A data frame with or without missing values

    Returns:
        DataFrame: A data frame with no missing values
    """
    # filter out missing values at the begin and end
    input_data = input_data.dropna()

    # now complete the timeseries
    input_data = input_data.reindex(pd.date_range(input_data.index.min(), input_data.index.max()))
    # our best guess is yesterday's value
    input_data = input_data.fillna(method='ffill')

    return input_data