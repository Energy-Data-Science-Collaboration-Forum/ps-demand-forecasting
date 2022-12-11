import datetime as dt
import pandas as pd

from src.utils import cutoff_forecast, remove_incomplete_settlement_periods, infer_gas_day, remove_zero_ccgt


def prepare_electricity_features(file_paths):
    """Gather all the features necessary to predict powerstations gas demand

    Args:
        file_paths (dict): A dictionary of feature names and file paths

    Returns:
        pandas DataFrame: A DataFrame with feature data in the columns and Gas Day on the index
    """
    features = []

    features.append(prepare_ted_forecast(file_paths["TED"]))
    features.append(prepare_wind_forecast(file_paths["WIND"]))
    features.append(prepare_actual_sofar(file_paths["ACTUAL_D_SOFAR_ALL_BUT_WIND_GT"]))

    features = pd.concat(features, axis=1).dropna()

    return features


def prepare_ted_forecast(file_path):
    """Read the TED forecast from the given file path and apply the necessary processing

    Args:
        file_path (str): The full file path to the TED forecast data

    Returns:
        pandas DataFrame: A dataframe with TED forecast data and Gas Day on the index
    """
    name = "TED_DA_FORECAST"

    demand = pd.read_csv(file_path)
    
    demand = demand.rename(
        columns={"startTime":"ELEC_DATETIME", "publishTime":"CREATED_ON", 
        "demand":"DEMAND", "settlementPeriod":"SETTLEMENT_PERIOD"}
    )
    
    for col in ["ELEC_DATETIME", "CREATED_ON"]:
        demand[col] = pd.to_datetime(demand[col]).dt.tz_convert("Europe/London").dt.tz_localize(None)

    demand["GAS_DAY"] = demand["ELEC_DATETIME"].apply(lambda edt: infer_gas_day(None, None, actual_datetime=edt))
    demand["GAS_DAY"] = pd.to_datetime(demand["GAS_DAY"])
    ## keep only available data at 12am the day before we predict for
    demand = cutoff_forecast(demand)

    ## keep latest for a given settlement period for a given day
    demand = demand.sort_values(
        ["ELEC_DATETIME", "CREATED_ON"],
        ascending=True,
    ).groupby(["ELEC_DATETIME"]).last().reset_index()

    demand = remove_incomplete_settlement_periods(name, demand)
    demand = demand.groupby(["GAS_DAY"])["DEMAND"].mean()
    demand = demand.reindex(pd.date_range(demand.index.min(), demand.index.max()))
    demand.index.name = "GAS_DAY"
    demand.name = name
    return demand.to_frame()


def prepare_wind_forecast(file_path):
    windf = pd.read_csv(file_path)

    windf = windf.rename(
        columns={"startTime":"ELEC_DATETIME", "publishTime":"CREATED_ON", 
        "generation":"DEMAND"}
    )
    
    for col in ["ELEC_DATETIME", "CREATED_ON"]:
        windf[col] = pd.to_datetime(windf[col]).dt.tz_convert("Europe/London").dt.tz_localize(None)

    windf["GAS_DAY"] = windf["ELEC_DATETIME"].apply(lambda edt: infer_gas_day(None, None, actual_datetime=edt))
    windf["GAS_DAY"] = pd.to_datetime(windf["GAS_DAY"])

    windf = cutoff_forecast(windf)

    windf = windf.sort_values(["ELEC_DATETIME", "CREATED_ON"], ascending=True)
    windf = windf.groupby(["ELEC_DATETIME"]).last().reset_index()
    
    # need settlement period to know which days are incomplete
    # we know the wind forecast is hourly so it will be all odd numbers
    windf['SETTLEMENT_PERIOD'] = windf['ELEC_DATETIME'].dt.hour * 2 + 1
    windf = remove_incomplete_settlement_periods(
        "WIND_FORECAST", windf, hourly_settlement_periods=True
    )

    windf = windf.groupby(["GAS_DAY"])["DEMAND"].mean()

    windf = windf.reindex(pd.date_range(windf.index.min(), windf.index.max()))
    windf.index.name = "GAS_DAY"
    windf.name = "WIND_FORECAST"

    return windf.to_frame()


def prepare_actual_sofar(filepath):
    """Read the electricity actuals from the given file path and apply the necessary processing
    to generate the actual so far feature.

    Args:
        file_path (str): The full file path to the electricity actuals data

    Returns:
        pandas DataFrame: A dataframe with the actual so far value for all electricity generation minus wind and GT 
        and Gas Day on the index
    """
    elec_actuals = prepare_electricity_actuals(
        filepath, actual_on_day_available_only=True
    ).sort_index(ascending=True)

    result = elec_actuals.drop(columns=["WIND", "CCGT", "OCGT"]).sum(axis=1)

    shift = 1
    new_index = pd.date_range(
        start=result.index.min(),
        end=result.index.max() + dt.timedelta(days=shift),
        freq="D",
        name="GAS_DAY",
    )
    result = result.reindex(new_index)
    result = result.shift(shift, freq="D")
    result.name = "ACTUAL_D_SOFAR_ALL_BUT_WIND_AND_GT"
    return result.to_frame()


def prepare_electricity_actuals(filepath, actual_on_day_available_only=False):
    """Read the actual electricity generation from the given file path and apply the necessary processing

    Args:
        file_path (str): The full file path to the TED forecast data

    Returns:
        pandas DataFrame: A dataframe with actual electricity data (for fuel type on the columns) and Gas Day on the index
    """
    name = "ELECTRICITY ACTUALS"

    gen = pd.read_csv(filepath)

    gen = gen.rename(
        columns={"startTime":"ELEC_DATETIME", "publishTime":"CREATED_ON", 
        "fuelType":"FUEL_TYPE", "settlementPeriod":"SETTLEMENT_PERIOD"}
    )
    for col in ["ELEC_DATETIME", "CREATED_ON"]:
        gen[col] = pd.to_datetime(gen[col]).dt.tz_convert("Europe/London").dt.tz_localize(None)

    gen["GAS_DAY"] = gen["ELEC_DATETIME"].apply(lambda edt: infer_gas_day(None, None, actual_datetime=edt))
    gen["GAS_DAY"] = pd.to_datetime(gen["GAS_DAY"])

    gen = (
        gen.drop_duplicates()
        .pivot(index=["ELEC_DATETIME", "GAS_DAY", "SETTLEMENT_PERIOD", "CREATED_ON"], columns='FUEL_TYPE', values='generation')
        .reset_index()
    )
    gen.columns.name = None

    gen = remove_zero_ccgt(name, gen)

    if actual_on_day_available_only:
        gen = cutoff_forecast(gen, days=0)

    # take latest values
    gen = gen.sort_values(
        ["ELEC_DATETIME", "CREATED_ON"], ascending=True
    )
    gen = gen.groupby(["ELEC_DATETIME"]).last().reset_index()

    if not actual_on_day_available_only:
        # if we're only looking at what's available on the day then settlement periods will always be incomplete
        gen = remove_incomplete_settlement_periods(name, gen)

    # aggregate to gas day: take the average over all settlement periods
    gen = (
        gen.fillna(0)
        .drop(columns=["ELEC_DATETIME", "SETTLEMENT_PERIOD"])
        .groupby(["GAS_DAY"])
        .mean(numeric_only=True)
    )

    gen = gen.reindex(pd.date_range(gen.index.min(), gen.index.max()))
    gen.index.name = "GAS_DAY"

    return gen


def prepare_gas_demand_actuals(file_path):
    """Read the gas demand actuals from the given file path and apply the necessary processing.
    The data is reported separately for Interconnectors, Powerstations, Industrials, Storage and LDZ.

    Args:
        file_path (str): The full file path to the gas demand actuals data

    Returns:
        pandas DataFrame: A dataframe with gas demand actuals data and Gas Day on the index
    """
    result = pd.read_csv(file_path, parse_dates=["ApplicableFor"])

    result = result.rename(columns={"ApplicableFor": "GAS_DAY"})
    result = result[["GAS_DAY", "TYPE", "Value"]]
    result["TYPE"] = (
        result["TYPE"]
        .str.replace("NTS Volume Offtaken, ", "")
        .str.replace(" Total", "")
        .str.replace("Industrial Offtake", "Industrial")
        .str.replace("Powerstations", "Powerstation")
        .str.upper()
    )
    result["GAS_DAY"] = result["GAS_DAY"].dt.tz_localize(None)

    demand = result.pivot(index="GAS_DAY", columns="TYPE", values="Value").rename(
        {
            "INTERCONNECTOR EXPORTS": "INTERCONNECTOR",
            "STORAGE INJECTION": "STORAGE",
            "POWERSTATION": "PS",
            "LDZ OFFTAKE": "LDZ",
        },
        axis=1,
    )

    demand.columns.name = None

    demand = demand.fillna(0).sort_index(ascending=True)

    return demand
